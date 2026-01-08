#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py  (Python 3.9+)

Evaluation script compatible with the OPT training pipeline that saves:
  - replica_XXX_<TAG>.weights.h5
  - replica_XXX_<TAG>_meta.npz

This script:
  1) Loads dataset_<TAG>.npz
  2) Loads all replica weights + meta
  3) Rebuilds the same model (subclassed) and loads weights
  4) Extracts CFF surfaces on the xi grid:
        ImH(xi), ReH_DR(xi), ReH_tot(xi), gate(xi), delta(xi), corr(xi)
  5) Plots:
        - Histograms of C0, gate_mean, corr_rms
        - Mean±std bands vs xi
        - Per-bin XS(phi), BSA(phi) comparisons with data + mean curve (+ optional band)
  6) Writes CSV/NPZ summary files

Run:
  python evaluate_gated_dr_pv_opt.py

Notes:
- Truth overlay: this script tries to read truth_<TAG>.json AND/OR truth arrays from NPZ
  if present (e.g., C0_truth, ImH_truth, ReH_truth, y_true).
- If truth is absent or has unexpected keys, the script will still run (truth overlay skipped).
"""

import glob
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs


# =========================
# USER CONFIG
# =========================

VERSION_DIR = "output"
TAG = "v_1"  # must match your dataset/training tag

DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")
TRUTH_JSON = os.path.join(VERSION_DIR, "data", f"truth_{TAG}.json")  # optional

REPLICAS_DIR = os.path.join(VERSION_DIR, "replicas_gated_dr_pv_opt")
WEIGHTS_GLOB = os.path.join(REPLICAS_DIR, f"replica_*_{TAG}.weights.h5")

OUT_DIR = os.path.join(VERSION_DIR, "eval_gated_dr_pv_opt", TAG)

# bkm settings (must match training/generator)
BEAM_ENERGY = 5.75
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# nuisance CFFs (must match training/generator)
CFF_E  = complex(2.217354372014208, 0.0)
CFF_HT = complex(1.409393726454478, 1.57736440256014)
CFF_ET = complex(144.4101642020152, 0.0)

# Plot options
PLOT_ENSEMBLE_BAND = True   # propagate all replicas through BKM to get ±1σ band
MAX_REPLICAS_FOR_BAND = 50  # cap for speed if you have many replicas

PLOT_TRUTH_CURVES = True    # use y_true from NPZ if available (or try truth JSON)
HIST_BINS = "auto"

# =========================
# END USER CONFIG
# =========================

_FLOATX = tf.float32
PI = float(np.pi)


# -------------------------
# IO helpers
# -------------------------
def _safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def load_npz_dict(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

def load_json_optional(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    xB = np.asarray(xB, dtype=float)
    return xB / (2.0 - xB)

def group_by_kinematics(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    decimals: int = 10,
) -> Dict[Tuple[float, float, float], Dict[str, np.ndarray]]:
    """
    Group pointwise arrays into unique (t, xB, Q2) bins with vectors over phi.
    """
    t_arr, xB_arr, Q2_arr, phi_arr = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    key_arr = np.round(np.stack([t_arr, xB_arr, Q2_arr], axis=1), decimals=decimals)

    bins: Dict[Tuple[float, float, float], Dict[str, np.ndarray]] = {}
    for (t_r, xB_r, Q2_r) in np.unique(key_arr, axis=0):
        mask = np.all(key_arr == np.array([t_r, xB_r, Q2_r]), axis=1)
        phi = phi_arr[mask]
        order = np.argsort(phi)
        bins[(float(t_r), float(xB_r), float(Q2_r))] = {
            "phi": phi[order].astype(np.float32),
            "xs": y[mask, 0][order].astype(np.float32),
            "bsa": y[mask, 1][order].astype(np.float32),
            "xs_err": yerr[mask, 0][order].astype(np.float32),
            "bsa_err": yerr[mask, 1][order].astype(np.float32),
        }
    return bins

def assert_common_phi_grid(bins: Dict[Tuple[float, float, float], Dict[str, np.ndarray]], atol: float = 1e-6) -> np.ndarray:
    keys = list(bins.keys())
    phi0 = bins[keys[0]]["phi"]
    for k in keys[1:]:
        phik = bins[k]["phi"]
        if len(phik) != len(phi0) or np.max(np.abs(phik - phi0)) > atol:
            raise ValueError("Expected a common phi grid across all bins.")
    return phi0


# -------------------------
# PV kernel + model (must match training)
# -------------------------
def build_pv_kernel_trapezoid(xi_nodes: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi_nodes, dtype=float).reshape(-1)
    B = xi.size

    w = np.zeros(B, dtype=float)
    for j in range(B):
        if j == 0:
            w[j] = 0.5 * (xi[1] - xi[0])
        elif j == B - 1:
            w[j] = 0.5 * (xi[B - 1] - xi[B - 2])
        else:
            w[j] = 0.5 * (xi[j + 1] - xi[j - 1])

    Xi = xi.reshape(-1, 1)
    Xj = xi.reshape(1, -1)
    diff = Xi - Xj
    summ = Xi + Xj

    K = (1.0 / PI) * w.reshape(1, -1) * (1.0 / diff - 1.0 / summ)
    np.fill_diagonal(K, 0.0)
    return K.astype(np.float32)

class PVDispersionLayer(tf.keras.layers.Layer):
    def __init__(self, xi_nodes: np.ndarray, K_matrix: np.ndarray, name: str = "pv_dr"):
        super().__init__(name=name)
        xi = np.asarray(xi_nodes, dtype=np.float32).reshape(-1)
        self.xi_nodes = tf.constant(xi, dtype=_FLOATX)
        self.B = int(xi.size)

        K_matrix = np.asarray(K_matrix, dtype=np.float32)
        if K_matrix.shape != (self.B, self.B):
            raise ValueError(f"K_matrix must have shape ({self.B},{self.B}), got {K_matrix.shape}")
        self.K = tf.Variable(K_matrix, trainable=False, dtype=_FLOATX, name="K")

    def call(self, imh: tf.Tensor, c0: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(self.K, imh) + c0

class TrainableScalar(tf.keras.layers.Layer):
    def __init__(self, init_value: float = 0.0, name: str = "trainable_scalar"):
        super().__init__(name=name)
        self.init_value = float(init_value)

    def build(self, input_shape):
        self.scalar = self.add_weight(
            name="value",
            shape=(),
            initializer=tf.constant_initializer(self.init_value),
            trainable=True,
            dtype=_FLOATX,
        )

    def call(self, x):
        b = tf.shape(x)[0]
        return tf.ones((b,), dtype=_FLOATX) * self.scalar

class GatedDRModelOPT(tf.keras.Model):
    """
    Must match the OPT training architecture:
      trunk: Dense(64) -> Dense(64)
      heads: imh_raw, delta_raw, gate_logit
      PV layer uses fixed K
      scaling: imh_scale * tanh(imh_raw), delta_scale * tanh(delta_raw)
    """
    def __init__(self, xi_nodes: np.ndarray, K_matrix: np.ndarray, imh_scale: float, delta_scale: float):
        super().__init__(name="GatedDRModel_OPT")

        self.imh_scale = float(imh_scale)
        self.delta_scale = float(delta_scale)

        # layers must match names used in training
        self.pv = PVDispersionLayer(xi_nodes=xi_nodes, K_matrix=K_matrix, name="pv_dr")
        self.c0_layer = TrainableScalar(init_value=0.0, name="C0")

        self.d1 = tf.keras.layers.Dense(64, activation="relu", name="dense")
        self.d2 = tf.keras.layers.Dense(64, activation="relu", name="dense_1")

        self.h_imh = tf.keras.layers.Dense(1, activation="linear", name="imh_raw")
        self.h_delta = tf.keras.layers.Dense(1, activation="linear", name="delta_raw")
        self.h_gate = tf.keras.layers.Dense(1, activation="linear", name="gate_logit")

    def call(self, kin: tf.Tensor, training: bool = False):
        x = self.d1(kin, training=training)
        x = self.d2(x, training=training)

        imh_raw = tf.squeeze(self.h_imh(x, training=training), axis=1)
        delta_raw = tf.squeeze(self.h_delta(x, training=training), axis=1)
        gate_logit = tf.squeeze(self.h_gate(x, training=training), axis=1)

        imh = tf.constant(self.imh_scale, dtype=_FLOATX) * tf.tanh(imh_raw)
        delta = tf.constant(self.delta_scale, dtype=_FLOATX) * tf.tanh(delta_raw)
        gate = tf.sigmoid(gate_logit)

        c0_vec = self.c0_layer(kin)
        c0 = c0_vec[0]

        reh_dr = self.pv(imh, c0)
        return reh_dr, imh, delta, gate, c0


# -------------------------
# bkm10 forward (evaluation only)
# -------------------------
def forward_bkm10_single_bin(
    reh: float,
    imh: float,
    t: float,
    xB: float,
    Q2: float,
    phi_rad: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    kin = BKM10Inputs(
        lab_kinematics_k=float(BEAM_ENERGY),
        squared_Q_momentum_transfer=float(Q2),
        x_Bjorken=float(xB),
        squared_hadronic_momentum_transfer_t=float(t),
    )
    cfg = {
        "kinematics": kin,
        "cff_inputs": CFFInputs(
            compton_form_factor_h=complex(float(reh), float(imh)),
            compton_form_factor_h_tilde=CFF_HT,
            compton_form_factor_e=CFF_E,
            compton_form_factor_e_tilde=CFF_ET,
        ),
        "target_polarization": float(TARGET_POLARIZATION),
        "lepton_beam_polarization": float(LEPTON_BEAM_POLARIZATION),
        "using_ww": bool(USING_WW),
    }
    xsecs = DifferentialCrossSection(configuration=cfg, verbose=False, debugging=False)
    xs = np.asarray(xsecs.compute_cross_section(phi_rad).real, dtype=float)
    bsa = np.asarray(xsecs.compute_bsa(phi_rad).real, dtype=float)
    return xs, bsa


# -------------------------
# Plot helpers
# -------------------------
def plot_hist(values: np.ndarray, truth: Optional[float], xlabel: str, title: str, outpath: str) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mu = float(np.mean(values))
    sig = float(np.std(values, ddof=0))

    plt.figure()
    plt.hist(values, bins=HIST_BINS, edgecolor="black", alpha=0.75)
    plt.axvline(mu, label="mean")
    plt.axvline(mu - sig, linestyle=":", label=r"$\pm 1\sigma$")
    plt.axvline(mu + sig, linestyle=":")

    if truth is not None and np.isfinite(truth):
        plt.axvline(float(truth), color="red", linestyle="--", linewidth=2, label="truth")

    plt.xlabel(xlabel)
    plt.ylabel("Replica count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return mu, sig

def plot_band_vs_xi(
    xi: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ylabel: str,
    title: str,
    outpath: str,
    truth: Optional[np.ndarray] = None,
) -> None:
    xi = np.asarray(xi, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    order = np.argsort(xi)
    x = xi[order]
    m = mean[order]
    s = std[order]

    plt.figure()
    plt.plot(x, m, label="ensemble mean")
    plt.fill_between(x, m - s, m + s, alpha=0.25, label=r"ensemble $\pm 1\sigma$")

    if truth is not None:
        truth = np.asarray(truth, dtype=float)[order]
        plt.plot(x, truth, linestyle="--", linewidth=2, label="truth")

    plt.xlabel(r"$\xi$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_data_with_curve(
    phi_deg: np.ndarray,
    y_obs: np.ndarray,
    y_err: np.ndarray,
    y_mean: np.ndarray,
    title: str,
    ylabel: str,
    outpath: str,
    y_std: Optional[np.ndarray] = None,
    y_truth: Optional[np.ndarray] = None,
) -> None:
    idx = np.argsort(phi_deg)
    x = phi_deg[idx]
    y = y_obs[idx]
    ye = y_err[idx]
    ym = y_mean[idx]

    plt.figure()
    plt.errorbar(x, y, yerr=ye, fmt="o", capsize=2, label="pseudodata")
    plt.plot(x, ym, label="inferred mean")

    if y_std is not None:
        ys = np.asarray(y_std, dtype=float)[idx]
        plt.fill_between(x, ym - ys, ym + ys, alpha=0.25, label=r"ensemble $\pm 1\sigma$")

    if y_truth is not None:
        yt = np.asarray(y_truth, dtype=float)[idx]
        plt.plot(x, yt, linestyle="--", linewidth=2, label="truth")

    plt.xlabel(r"$\phi$ (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def main() -> None:
    _safe_mkdir(OUT_DIR)

    d = load_npz_dict(DATA_NPZ)
    X = d["x"].astype(np.float32)                 # (N,4): t, xB, Q2, phi(rad)
    y_central = d["y_central"].astype(np.float32) # (N,2): XS, BSA
    y_sigma = d["y_sigma"].astype(np.float32)     # (N,2): XS_err, BSA_err

    bins_data = group_by_kinematics(X, y_central, y_sigma)
    phi_grid = assert_common_phi_grid(bins_data)
    Nphi = len(phi_grid)

    # Sort bins by xi (consistent with training)
    keys_sorted = sorted(bins_data.keys(), key=lambda k: xi_from_xB(k[1]))
    B = len(keys_sorted)

    t_bins = np.array([k[0] for k in keys_sorted], dtype=float)
    xB_bins = np.array([k[1] for k in keys_sorted], dtype=float)
    Q2_bins = np.array([k[2] for k in keys_sorted], dtype=float)
    xi_bins = xi_from_xB(xB_bins).astype(float)

    phi_deg = np.degrees(phi_grid.astype(float))

    print(f"Dataset bins: B={B}, Nphi={Nphi}, xi range=[{xi_bins.min():.6g},{xi_bins.max():.6g}]")

    # Optional truth arrays from NPZ
    y_true_bins = None
    if PLOT_TRUTH_CURVES and ("y_true" in d):
        y_true = d["y_true"].astype(np.float32)
        bins_true = group_by_kinematics(X, y_true, y_sigma*0.0)
        y_true_bins = bins_true
        print("Found y_true in NPZ -> will overlay truth curves (dashed).")

    C0_truth = float(d["C0_truth"]) if ("C0_truth" in d) else None
    ImH_truth = d["ImH_truth"].astype(float) if ("ImH_truth" in d) else None
    ReH_truth = d["ReH_truth"].astype(float) if ("ReH_truth" in d) else None

    # Optional truth json (robustly used only for extra info)
    truth_json = load_json_optional(TRUTH_JSON)
    if truth_json is not None:
        print(f"Loaded truth JSON: {TRUTH_JSON}")
    else:
        print("Truth JSON not found or unreadable (ok).")

    # Load replica weights list
    weight_paths = sorted(glob.glob(WEIGHTS_GLOB))
    if not weight_paths:
        raise FileNotFoundError(f"No replica weights found: {WEIGHTS_GLOB}")
    print(f"Found {len(weight_paths)} replica weight files.")

    # Load meta from the first replica (assume common xi/K/scales across replicas)
    first_base = weight_paths[0].replace(".weights.h5", "")
    first_meta = first_base + "_meta.npz"
    if not os.path.exists(first_meta):
        raise FileNotFoundError(f"Missing meta file for first replica: {first_meta}")

    meta0 = np.load(first_meta, allow_pickle=True)
    xi_meta = meta0["xi_bins"].astype(float).reshape(-1)
    K_used = meta0["K_used"].astype(np.float32) if "K_used" in meta0.files else meta0["K_used".lower()]  # defensive

    # Pull scales from meta config if present
    imh_scale = 8.0
    delta_scale = 2.0
    if "config" in meta0.files:
        cfg = meta0["config"].item()
        imh_scale = float(cfg.get("IMH_SCALE", imh_scale))
        delta_scale = float(cfg.get("DELTA_REH_SCALE", delta_scale))

    # Validate xi ordering matches dataset
    if xi_meta.shape[0] != B:
        print("WARNING: xi_bins length in meta does not match dataset B. Using dataset xi for plotting; model uses meta xi.")
    else:
        max_dxi = float(np.max(np.abs(xi_meta - xi_bins)))
        print(f"xi consistency check: max|xi_meta - xi_dataset| = {max_dxi:.3e}")
        if max_dxi > 1e-5:
            print("WARNING: xi mismatch is larger than expected. If large, Hard-DR closure may not match.")

    # Build evaluation model
    # Note: model kin input is (B,3) with bins ordered by xi increasing
    kin_bins = np.stack([t_bins, xB_bins, Q2_bins], axis=1).astype(np.float32)
    kin_tf = tf.constant(kin_bins, dtype=_FLOATX)

    model = GatedDRModelOPT(xi_nodes=xi_meta, K_matrix=K_used, imh_scale=imh_scale, delta_scale=delta_scale)
    _ = model(kin_tf, training=False)  # build variables

    # Collect per-replica outputs
    C0_list = []
    gate_mean_list = []
    corr_rms_list = []

    ImH_list = []
    ReH_DR_list = []
    ReH_tot_list = []
    gate_list = []
    delta_list = []
    corr_list = []

    # Per-bin data arrays for plotting
    xs_obs = np.stack([bins_data[k]["xs"] for k in keys_sorted], axis=0).astype(float)      # (B,Nphi)
    bsa_obs = np.stack([bins_data[k]["bsa"] for k in keys_sorted], axis=0).astype(float)
    xs_err = np.stack([bins_data[k]["xs_err"] for k in keys_sorted], axis=0).astype(float)
    bsa_err = np.stack([bins_data[k]["bsa_err"] for k in keys_sorted], axis=0).astype(float)

    for wp in weight_paths:
        base = wp.replace(".weights.h5", "")
        meta_path = base + "_meta.npz"
        if not os.path.exists(meta_path):
            print(f"WARNING: missing meta for {os.path.basename(wp)} -> skipping")
            continue

        # Load weights into the same model object
        model.load_weights(wp)

        reh_dr, imh, delta, gate, c0 = model(kin_tf, training=False)

        reh_dr = reh_dr.numpy().astype(float)
        imh = imh.numpy().astype(float)
        delta = delta.numpy().astype(float)
        gate = gate.numpy().astype(float)
        c0 = float(c0.numpy())

        corr = (1.0 - gate) * delta
        reh_tot = reh_dr + corr

        C0_list.append(c0)
        gate_mean_list.append(float(np.mean(gate)))
        corr_rms_list.append(float(np.sqrt(np.mean(corr**2))))

        ImH_list.append(imh)
        ReH_DR_list.append(reh_dr)
        ReH_tot_list.append(reh_tot)
        gate_list.append(gate)
        delta_list.append(delta)
        corr_list.append(corr)

    # Convert to arrays
    C0_arr = np.asarray(C0_list, dtype=float)
    gate_mean_arr = np.asarray(gate_mean_list, dtype=float)
    corr_rms_arr = np.asarray(corr_rms_list, dtype=float)

    ImH_arr = np.asarray(ImH_list, dtype=float)          # (R,B)
    ReH_DR_arr = np.asarray(ReH_DR_list, dtype=float)    # (R,B)
    ReH_tot_arr = np.asarray(ReH_tot_list, dtype=float)  # (R,B)
    gate_arr = np.asarray(gate_list, dtype=float)        # (R,B)
    delta_arr = np.asarray(delta_list, dtype=float)      # (R,B)
    corr_arr = np.asarray(corr_list, dtype=float)        # (R,B)

    R = ImH_arr.shape[0]
    print(f"Loaded {R} replicas (after skipping missing meta).")

    # Ensemble stats vs xi
    def mean_std(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.mean(a, axis=0), np.std(a, axis=0, ddof=0)

    ImH_mu, ImH_sig = mean_std(ImH_arr)
    ReH_DR_mu, ReH_DR_sig = mean_std(ReH_DR_arr)
    ReH_tot_mu, ReH_tot_sig = mean_std(ReH_tot_arr)
    gate_mu, gate_sig = mean_std(gate_arr)
    corr_mu, corr_sig = mean_std(corr_arr)

    # Save summary CSV / NPZ
    try:
        import pandas as pd
        pd.DataFrame({
            "replica": np.arange(1, R + 1),
            "C0": C0_arr,
            "gate_mean": gate_mean_arr,
            "corr_rms": corr_rms_arr,
        }).to_csv(os.path.join(OUT_DIR, f"replica_summary_{TAG}.csv"), index=False)
        print("Wrote:", os.path.join(OUT_DIR, f"replica_summary_{TAG}.csv"))
    except Exception:
        pass

    np.savez_compressed(
        os.path.join(OUT_DIR, f"ensemble_cffs_{TAG}.npz"),
        xi_bins=xi_bins.astype(np.float32),
        C0_values=C0_arr.astype(np.float32),
        gate_mean_values=gate_mean_arr.astype(np.float32),
        corr_rms_values=corr_rms_arr.astype(np.float32),
        ImH_values=ImH_arr.astype(np.float32),
        ReH_DR_values=ReH_DR_arr.astype(np.float32),
        ReH_tot_values=ReH_tot_arr.astype(np.float32),
        gate_values=gate_arr.astype(np.float32),
        delta_values=delta_arr.astype(np.float32),
        corr_values=corr_arr.astype(np.float32),
        ImH_mean=ImH_mu.astype(np.float32),
        ImH_std=ImH_sig.astype(np.float32),
        ReH_DR_mean=ReH_DR_mu.astype(np.float32),
        ReH_DR_std=ReH_DR_sig.astype(np.float32),
        ReH_tot_mean=ReH_tot_mu.astype(np.float32),
        ReH_tot_std=ReH_tot_sig.astype(np.float32),
        gate_mean_vs_xi=gate_mu.astype(np.float32),
        gate_std_vs_xi=gate_sig.astype(np.float32),
        corr_mean_vs_xi=corr_mu.astype(np.float32),
        corr_std_vs_xi=corr_sig.astype(np.float32),
    )
    print("Wrote:", os.path.join(OUT_DIR, f"ensemble_cffs_{TAG}.npz"))

    # Histograms
    plot_hist(C0_arr, C0_truth, xlabel=r"$C_0$", title="Replica histogram: C0", outpath=os.path.join(OUT_DIR, f"hist_C0_{TAG}.png"))
    plot_hist(gate_mean_arr, None, xlabel=r"$\langle g \rangle$", title="Replica histogram: gate mean", outpath=os.path.join(OUT_DIR, f"hist_gateMean_{TAG}.png"))
    plot_hist(corr_rms_arr, None, xlabel=r"$\mathrm{RMS}[(1-g)\Delta\Re H]$", title="Replica histogram: correction RMS", outpath=os.path.join(OUT_DIR, f"hist_corrRMS_{TAG}.png"))

    # Bands vs xi
    plot_band_vs_xi(
        xi_bins, ImH_mu, ImH_sig,
        ylabel=r"$\Im \mathcal{H}$",
        title=r"Ensemble $\Im \mathcal{H}(\xi)$",
        outpath=os.path.join(OUT_DIR, f"band_ImH_{TAG}.png"),
        truth=ImH_truth,
    )
    plot_band_vs_xi(
        xi_bins, ReH_tot_mu, ReH_tot_sig,
        ylabel=r"$\Re \mathcal{H}_{\rm tot}$",
        title=r"Ensemble $\Re \mathcal{H}_{\rm tot}(\xi)$",
        outpath=os.path.join(OUT_DIR, f"band_ReHtot_{TAG}.png"),
        truth=ReH_truth,
    )
    plot_band_vs_xi(
        xi_bins, ReH_DR_mu, ReH_DR_sig,
        ylabel=r"$\Re \mathcal{H}_{\rm DR}$",
        title=r"Ensemble $\Re \mathcal{H}_{\rm DR}(\xi)$",
        outpath=os.path.join(OUT_DIR, f"band_ReHDR_{TAG}.png"),
        truth=None,
    )
    plot_band_vs_xi(
        xi_bins, corr_mu, corr_sig,
        ylabel=r"$(1-g)\Delta\Re \mathcal{H}$",
        title=r"Ensemble effective correction $(1-g)\Delta\Re \mathcal{H}(\xi)$",
        outpath=os.path.join(OUT_DIR, f"band_corr_{TAG}.png"),
        truth=None,
    )
    plot_band_vs_xi(
        xi_bins, gate_mu, gate_sig,
        ylabel=r"$g(\xi)$",
        title=r"Ensemble gate $g(\xi)$",
        outpath=os.path.join(OUT_DIR, f"band_gate_{TAG}.png"),
        truth=None,
    )

    # Observable curves: compute inferred mean curves from mean CFFs
    xs_mean = np.zeros((B, Nphi), dtype=float)
    bsa_mean = np.zeros((B, Nphi), dtype=float)

    for b in range(B):
        xs_m, bsa_m = forward_bkm10_single_bin(
            reh=float(ReH_tot_mu[b]),
            imh=float(ImH_mu[b]),
            t=float(t_bins[b]),
            xB=float(xB_bins[b]),
            Q2=float(Q2_bins[b]),
            phi_rad=phi_grid.astype(float),
        )
        xs_mean[b, :] = xs_m
        bsa_mean[b, :] = bsa_m

    # Optional band: propagate each replica through BKM
    xs_std = None
    bsa_std = None
    if PLOT_ENSEMBLE_BAND:
        n_use = min(R, int(MAX_REPLICAS_FOR_BAND))
        xs_all = np.zeros((n_use, B, Nphi), dtype=float)
        bsa_all = np.zeros((n_use, B, Nphi), dtype=float)

        for r_use in range(n_use):
            for b in range(B):
                xs_r, bsa_r = forward_bkm10_single_bin(
                    reh=float(ReH_tot_arr[r_use, b]),
                    imh=float(ImH_arr[r_use, b]),
                    t=float(t_bins[b]),
                    xB=float(xB_bins[b]),
                    Q2=float(Q2_bins[b]),
                    phi_rad=phi_grid.astype(float),
                )
                xs_all[r_use, b, :] = xs_r
                bsa_all[r_use, b, :] = bsa_r

        xs_std = np.std(xs_all, axis=0, ddof=0)   # (B,Nphi)
        bsa_std = np.std(bsa_all, axis=0, ddof=0)
        print(f"Computed ensemble bands using {n_use} replicas.")

    # Per-bin plots
    for b in range(B):
        xi_b = float(xi_bins[b])
        title_tag = f"bin {b:02d}: t={t_bins[b]:.3g}, xB={xB_bins[b]:.4g}, Q2={Q2_bins[b]:.3g}, xi={xi_b:.4g}"

        xs_truth = None
        bsa_truth = None
        if y_true_bins is not None:
            xs_truth = y_true_bins[keys_sorted[b]]["xs"].astype(float)
            bsa_truth = y_true_bins[keys_sorted[b]]["bsa"].astype(float)

        plot_data_with_curve(
            phi_deg=phi_deg,
            y_obs=xs_obs[b, :],
            y_err=xs_err[b, :],
            y_mean=xs_mean[b, :],
            y_std=(xs_std[b, :] if xs_std is not None else None),
            y_truth=xs_truth,
            ylabel="XS",
            title="XS vs phi | " + title_tag,
            outpath=os.path.join(OUT_DIR, f"xs_bin{b:02d}_{TAG}.png"),
        )
        plot_data_with_curve(
            phi_deg=phi_deg,
            y_obs=bsa_obs[b, :],
            y_err=bsa_err[b, :],
            y_mean=bsa_mean[b, :],
            y_std=(bsa_std[b, :] if bsa_std is not None else None),
            y_truth=bsa_truth,
            ylabel="BSA",
            title="BSA vs phi | " + title_tag,
            outpath=os.path.join(OUT_DIR, f"bsa_bin{b:02d}_{TAG}.png"),
        )

    print("\nWrote evaluation outputs to:", OUT_DIR)
    print("Key files:")
    print(f"  - replica_summary_{TAG}.csv")
    print(f"  - ensemble_cffs_{TAG}.npz")
    print(f"  - hist_C0_{TAG}.png, hist_gateMean_{TAG}.png, hist_corrRMS_{TAG}.png")
    print(f"  - band_ImH_{TAG}.png, band_ReHtot_{TAG}.png, band_ReHDR_{TAG}.png, band_corr_{TAG}.png, band_gate_{TAG}.png")
    print("  - xs_binXX_*.png, bsa_binXX_*.png")


if __name__ == "__main__":
    main()
