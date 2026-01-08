#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py  (Python 3.9+)

Evaluation script for FiLM-gated non-bias DR training output.

Expects trained replicas:
  <VERSION_DIR>/replicas_film_dr_nobias/replica_XXX_<TAG>.weights.h5
  <VERSION_DIR>/replicas_film_dr_nobias/replica_XXX_<TAG>_meta.npz

Also expects dataset:
  <VERSION_DIR>/data/dataset_<TAG>.npz
Optionally:
  <VERSION_DIR>/data/truth_<TAG>.json (not required; NPZ truth preferred)

Outputs plots and summary tables under:
  <VERSION_DIR>/eval_film_dr_nobias/<TAG>/
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
TAG = "v_1"

DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")
TRUTH_JSON = os.path.join(VERSION_DIR, "data", f"truth_{TAG}.json")  # optional

MODELS_DIR = os.path.join(VERSION_DIR, "replicas_film_dr_nobias")
WEIGHTS_GLOB = os.path.join(MODELS_DIR, f"replica_*_{TAG}.weights.h5")

OUT_DIR = os.path.join(VERSION_DIR, "eval_film_dr_nobias", TAG)

# bkm10 settings (must match generator/training)
BEAM_ENERGY = 5.75
Q2_DEFAULT = 1.82
T_DEFAULT = -0.17
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# nuisance CFFs (must match generator/training)
CFF_E  = complex(2.217354372014208, 0.0)
CFF_HT = complex(1.409393726454478, 1.57736440256014)
CFF_ET = complex(144.4101642020152, 0.0)

# plot / speed options
PLOT_ENSEMBLE_BAND = True
MAX_REPLICAS_FOR_BAND = 50
HIST_BINS = "auto"

# =========================
# END USER CONFIG
# =========================

_FLOATX = tf.float32
PI = float(np.pi)


# -------------------------
# Helpers
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
# Model (must match training exactly)
# -------------------------
class PVDispersionLayer(tf.keras.layers.Layer):
    def __init__(self, K_matrix: np.ndarray, name: str = "pv_dr"):
        super().__init__(name=name)
        K_matrix = np.asarray(K_matrix, dtype=np.float32)
        self.K = tf.Variable(K_matrix, trainable=False, dtype=_FLOATX, name="K")

    def call(self, imh: tf.Tensor, c0: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(self.K, imh) + c0

class TrainableScalar(tf.keras.layers.Layer):
    def __init__(self, init_value: float = 0.0, name: str = "C0"):
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

class FiLMGateDRModel(tf.keras.Model):
    def __init__(
        self,
        K_matrix: np.ndarray,
        imh_scale: float,
        reh_scale: float,
        gate_width: int,
        film_ctx_width: int,
        film_gamma_scale: float,
        film_beta_scale: float,
        alpha_bias_init: float,
        seed: Optional[int] = None,
    ):
        super().__init__(name="FiLMGateDRModel")

        init = tf.keras.initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=seed)

        self.imh_scale = float(imh_scale)
        self.reh_scale = float(reh_scale)
        self.film_gamma_scale = float(film_gamma_scale)
        self.film_beta_scale = float(film_beta_scale)

        self.pv = PVDispersionLayer(K_matrix=K_matrix, name="pv_dr")
        self.c0_layer = TrainableScalar(init_value=0.0, name="C0")

        self.tr1 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init, name="trunk_dense1")
        self.tr2 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init, name="trunk_dense2")

        self.h_imh = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init, name="imh_raw")
        self.h_reh = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init, name="reh_free_raw")

        self.g1 = tf.keras.layers.Dense(gate_width, activation="relu", kernel_initializer=init, name="gate_hidden")

        self.ctx1 = tf.keras.layers.Dense(film_ctx_width, activation="relu", kernel_initializer=init, name="film_ctx")

        self.to_gamma = tf.keras.layers.Dense(
            gate_width, activation="linear",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="film_gamma_raw"
        )
        self.to_beta = tf.keras.layers.Dense(
            gate_width, activation="linear",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="film_beta_raw"
        )
        self.to_alpha = tf.keras.layers.Dense(
            1, activation="linear",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Constant(float(alpha_bias_init)),
            name="alpha_logit"
        )

    def call(self, kin: tf.Tensor, training: bool = False):
        # kin: (B,3) = (t,xB,Q2)
        xB = kin[:, 1]
        xi = xB / (2.0 - xB)

        h = self.tr1(kin, training=training)
        h = self.tr2(h, training=training)

        imh_raw = tf.squeeze(self.h_imh(h, training=training), axis=1)
        reh_raw = tf.squeeze(self.h_reh(h, training=training), axis=1)

        imh = tf.constant(self.imh_scale, dtype=_FLOATX) * tf.tanh(imh_raw)
        reh_free = tf.constant(self.reh_scale, dtype=_FLOATX) * tf.tanh(reh_raw)

        c0_vec = self.c0_layer(kin)
        c0 = c0_vec[0]

        reh_dr = self.pv(imh, c0)

        gh = self.g1(h, training=training)

        ctx = tf.stack([xi, imh, reh_dr], axis=1)
        ctxh = self.ctx1(ctx, training=training)

        gamma_raw = self.to_gamma(ctxh, training=training)
        beta_raw = self.to_beta(ctxh, training=training)

        gamma = 1.0 + tf.constant(self.film_gamma_scale, dtype=_FLOATX) * gamma_raw
        beta = tf.constant(self.film_beta_scale, dtype=_FLOATX) * beta_raw

        gh_mod = gamma * gh + beta

        alpha_logit = tf.squeeze(self.to_alpha(gh_mod, training=training), axis=1)
        alpha = tf.sigmoid(alpha_logit)

        return reh_free, imh, reh_dr, alpha, c0, gamma_raw, beta_raw


# -------------------------
# Plot helpers
# -------------------------
def plot_hist(values: np.ndarray, xlabel: str, title: str, outpath: str, truth: Optional[float] = None) -> None:
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
    idx = np.argsort(xi)

    x = xi[idx]
    m = mean[idx]
    s = std[idx]

    plt.figure()
    plt.plot(x, m, label="ensemble mean")
    plt.fill_between(x, m - s, m + s, alpha=0.25, label=r"ensemble $\pm 1\sigma$")

    if truth is not None:
        tt = np.asarray(truth, dtype=float)[idx]
        plt.plot(x, tt, linestyle="--", linewidth=2, label="truth")

    plt.xlabel(r"$\xi$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_data_curve(
    phi_deg: np.ndarray,
    y_obs: np.ndarray,
    y_err: np.ndarray,
    y_mean: np.ndarray,
    ylabel: str,
    title: str,
    outpath: str,
    y_std: Optional[np.ndarray] = None,
    y_truth: Optional[np.ndarray] = None,
) -> None:
    idx = np.argsort(phi_deg)
    x = phi_deg[idx]
    y = np.asarray(y_obs, dtype=float)[idx]
    ye = np.asarray(y_err, dtype=float)[idx]
    ym = np.asarray(y_mean, dtype=float)[idx]

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
    X = d["x"].astype(np.float32)
    y_central = d["y_central"].astype(np.float32)
    y_sigma = d["y_sigma"].astype(np.float32)

    bins_data = group_by_kinematics(X, y_central, y_sigma)
    phi_grid = assert_common_phi_grid(bins_data)
    phi_deg = np.degrees(phi_grid.astype(float))

    keys_sorted = sorted(bins_data.keys(), key=lambda k: xi_from_xB(k[1]))
    B = len(keys_sorted)
    Nphi = len(phi_grid)

    t_bins = np.array([k[0] for k in keys_sorted], dtype=float)
    xB_bins = np.array([k[1] for k in keys_sorted], dtype=float)
    Q2_bins = np.array([k[2] for k in keys_sorted], dtype=float)
    xi_bins = xi_from_xB(xB_bins)

    print(f"Dataset bins: B={B}, Nphi={Nphi}, xi range=[{xi_bins.min():.6g},{xi_bins.max():.6g}]")

    # Truth arrays from NPZ if available
    C0_truth = float(d["C0_truth"]) if "C0_truth" in d else None
    ImH_truth = d["ImH_truth"].astype(float) if "ImH_truth" in d else None
    ReH_truth = d["ReH_truth"].astype(float) if "ReH_truth" in d else None
    ReH_DR_truth = d["ReH_DR_truth"].astype(float) if "ReH_DR_truth" in d else None
    Delta_viol_truth = d["Delta_viol_truth"].astype(float) if "Delta_viol_truth" in d else None

    y_true_bins = None
    if "y_true" in d:
        y_true = d["y_true"].astype(np.float32)
        bins_true = group_by_kinematics(X, y_true, y_sigma * 0.0)
        y_true_bins = bins_true
        print("Found y_true in NPZ -> overlay truth curves enabled.")

    # Load replicas
    weight_paths = sorted(glob.glob(WEIGHTS_GLOB))
    if not weight_paths:
        raise FileNotFoundError(f"No replica weights found: {WEIGHTS_GLOB}")
    print(f"Found {len(weight_paths)} replica weight files.")

    # Build model from first replica meta
    first_base = weight_paths[0].replace(".weights.h5", "")
    first_meta_path = first_base + "_meta.npz"
    if not os.path.exists(first_meta_path):
        raise FileNotFoundError(f"Missing meta file: {first_meta_path}")

    meta0 = np.load(first_meta_path, allow_pickle=True)
    K_used = meta0["K_used"].astype(np.float32)
    cfg = meta0["config"].item() if "config" in meta0.files else {}

    IMH_SCALE = float(cfg.get("IMH_SCALE", 8.0))
    REH_SCALE = float(cfg.get("REH_SCALE", 6.0))
    GATE_WIDTH = int(cfg.get("GATE_WIDTH", 32))
    FILM_CTX_WIDTH = int(cfg.get("FILM_CTX_WIDTH", 16))
    FILM_GAMMA_SCALE = float(cfg.get("FILM_GAMMA_SCALE", 0.05))
    FILM_BETA_SCALE = float(cfg.get("FILM_BETA_SCALE", 0.05))
    ALPHA_BIAS_INIT = float(cfg.get("ALPHA_BIAS_INIT", -6.0))

    model = FiLMGateDRModel(
        K_matrix=K_used,
        imh_scale=IMH_SCALE,
        reh_scale=REH_SCALE,
        gate_width=GATE_WIDTH,
        film_ctx_width=FILM_CTX_WIDTH,
        film_gamma_scale=FILM_GAMMA_SCALE,
        film_beta_scale=FILM_BETA_SCALE,
        alpha_bias_init=ALPHA_BIAS_INIT,
    )

    kin_bins = np.stack([t_bins, xB_bins, Q2_bins], axis=1).astype(np.float32)
    kin_tf = tf.constant(kin_bins, dtype=_FLOATX)
    _ = model(kin_tf, training=False)  # build vars

    # Per-replica extraction
    C0_list = []
    alpha_mean_list = []
    selected_list = []
    best1_list = []
    best2_list = []
    improv_list = []

    ImH_list = []
    ReH_free_list = []
    ReH_DR_list = []
    ReH_pred_list = []
    alpha_list = []

    for wp in weight_paths:
        base = wp.replace(".weights.h5", "")
        meta_path = base + "_meta.npz"
        if not os.path.exists(meta_path):
            print(f"WARNING: missing meta for {os.path.basename(wp)} -> skipping")
            continue
        meta = np.load(meta_path, allow_pickle=True)
        cfg_r = meta["config"].item() if "config" in meta.files else {}

        selected = str(cfg_r.get("selected", "film_gate"))
        best1 = float(cfg_r.get("best_val_stage1", np.nan))
        best2 = float(cfg_r.get("best_val_stage2", np.nan))
        improv = best1 - best2

        model.load_weights(wp)
        reh_free, imh, reh_dr, alpha, c0, _, _ = model(kin_tf, training=False)

        reh_free = reh_free.numpy().astype(float)
        imh = imh.numpy().astype(float)
        reh_dr = reh_dr.numpy().astype(float)
        alpha = alpha.numpy().astype(float)
        c0 = float(c0.numpy())

        if selected == "free":
            reh_pred = reh_free
        else:
            reh_pred = reh_free + alpha * (reh_dr - reh_free)

        C0_list.append(c0)
        alpha_mean_list.append(float(np.mean(alpha)))
        selected_list.append(selected)
        best1_list.append(best1)
        best2_list.append(best2)
        improv_list.append(improv)

        ImH_list.append(imh)
        ReH_free_list.append(reh_free)
        ReH_DR_list.append(reh_dr)
        ReH_pred_list.append(reh_pred)
        alpha_list.append(alpha)

    C0_arr = np.asarray(C0_list, dtype=float)
    alpha_mean_arr = np.asarray(alpha_mean_list, dtype=float)
    best1_arr = np.asarray(best1_list, dtype=float)
    best2_arr = np.asarray(best2_list, dtype=float)
    improv_arr = np.asarray(improv_list, dtype=float)

    ImH_arr = np.asarray(ImH_list, dtype=float)            # (R,B)
    ReH_free_arr = np.asarray(ReH_free_list, dtype=float)
    ReH_DR_arr = np.asarray(ReH_DR_list, dtype=float)
    ReH_pred_arr = np.asarray(ReH_pred_list, dtype=float)
    alpha_arr = np.asarray(alpha_list, dtype=float)

    R = ImH_arr.shape[0]
    print(f"Loaded {R} replicas (after skipping missing meta).")

    # Selection stats
    frac_free = float(np.mean([s == "free" for s in selected_list])) if selected_list else np.nan
    frac_gate = float(np.mean([s != "free" for s in selected_list])) if selected_list else np.nan
    print(f"Selection fractions: free={frac_free:.3f}, film_gate={frac_gate:.3f}")

    # Ensemble stats
    def mean_std(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.mean(a, axis=0), np.std(a, axis=0, ddof=0)

    ImH_mu, ImH_sig = mean_std(ImH_arr)
    ReH_pred_mu, ReH_pred_sig = mean_std(ReH_pred_arr)
    ReH_DR_mu, ReH_DR_sig = mean_std(ReH_DR_arr)
    alpha_mu, alpha_sig = mean_std(alpha_arr)

    # Save summary CSV
    try:
        import pandas as pd
        pd.DataFrame({
            "replica_file": [os.path.basename(p) for p in weight_paths[:R]],
            "selected": selected_list,
            "C0": C0_arr,
            "alpha_mean": alpha_mean_arr,
            "best_val_stage1": best1_arr,
            "best_val_stage2": best2_arr,
            "improvement": improv_arr,
        }).to_csv(os.path.join(OUT_DIR, f"replica_summary_{TAG}.csv"), index=False)
        print("Wrote:", os.path.join(OUT_DIR, f"replica_summary_{TAG}.csv"))
    except Exception:
        pass

    # Histograms
    plot_hist(C0_arr, xlabel=r"$C_0$", title="Replica histogram: C0", outpath=os.path.join(OUT_DIR, f"hist_C0_{TAG}.png"), truth=C0_truth)
    plot_hist(alpha_mean_arr, xlabel=r"$\langle \alpha \rangle$", title="Replica histogram: mean alpha", outpath=os.path.join(OUT_DIR, f"hist_alphaMean_{TAG}.png"))
    plot_hist(improv_arr[np.isfinite(improv_arr)], xlabel="best_val_stage1 - best_val_stage2", title="Replica histogram: val improvement", outpath=os.path.join(OUT_DIR, f"hist_valImprovement_{TAG}.png"))

    # Bands vs xi
    plot_band_vs_xi(xi_bins, ImH_mu, ImH_sig, ylabel=r"$\Im \mathcal{H}$", title=r"Ensemble $\Im \mathcal{H}(\xi)$", outpath=os.path.join(OUT_DIR, f"band_ImH_{TAG}.png"), truth=ImH_truth)
    plot_band_vs_xi(xi_bins, ReH_pred_mu, ReH_pred_sig, ylabel=r"$\Re \mathcal{H}_{\rm pred}$", title=r"Ensemble $\Re \mathcal{H}_{\rm pred}(\xi)$", outpath=os.path.join(OUT_DIR, f"band_ReHpred_{TAG}.png"), truth=ReH_truth)
    plot_band_vs_xi(xi_bins, ReH_DR_mu, ReH_DR_sig, ylabel=r"$\Re \mathcal{H}_{\rm DR}$", title=r"Ensemble $\Re \mathcal{H}_{\rm DR}(\xi)$", outpath=os.path.join(OUT_DIR, f"band_ReHDR_{TAG}.png"), truth=ReH_DR_truth)
    plot_band_vs_xi(xi_bins, alpha_mu, alpha_sig, ylabel=r"$\alpha(\xi)$", title=r"Ensemble gate $\alpha(\xi)$", outpath=os.path.join(OUT_DIR, f"band_alpha_{TAG}.png"))

    # Plot truth violation if present
    if Delta_viol_truth is not None:
        plt.figure()
        plt.plot(xi_bins, Delta_viol_truth, linestyle="--", linewidth=2)
        plt.xlabel(r"$\xi$")
        plt.ylabel(r"$\Delta_{\rm viol}(\xi)$ (truth)")
        plt.title("Truth DR violation term")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"truth_delta_viol_{TAG}.png"), dpi=200)
        plt.close()

    # Observable curves using ensemble-mean CFFs
    xs_obs = np.stack([bins_data[k]["xs"] for k in keys_sorted], axis=0).astype(float)      # (B,Nphi)
    bsa_obs = np.stack([bins_data[k]["bsa"] for k in keys_sorted], axis=0).astype(float)
    xs_err = np.stack([bins_data[k]["xs_err"] for k in keys_sorted], axis=0).astype(float)
    bsa_err = np.stack([bins_data[k]["bsa_err"] for k in keys_sorted], axis=0).astype(float)

    xs_mean = np.zeros((B, Nphi), dtype=float)
    bsa_mean = np.zeros((B, Nphi), dtype=float)

    for b in range(B):
        xs_m, bsa_m = forward_bkm10_single_bin(
            reh=float(ReH_pred_mu[b]),
            imh=float(ImH_mu[b]),
            t=float(t_bins[b]),
            xB=float(xB_bins[b]),
            Q2=float(Q2_bins[b]),
            phi_rad=phi_grid.astype(float),
        )
        xs_mean[b, :] = xs_m
        bsa_mean[b, :] = bsa_m

    # Optional ensemble band by propagating replicas
    xs_std = None
    bsa_std = None
    if PLOT_ENSEMBLE_BAND and R > 1:
        n_use = min(R, int(MAX_REPLICAS_FOR_BAND))
        xs_all = np.zeros((n_use, B, Nphi), dtype=float)
        bsa_all = np.zeros((n_use, B, Nphi), dtype=float)
        for r_use in range(n_use):
            for b in range(B):
                xs_r, bsa_r = forward_bkm10_single_bin(
                    reh=float(ReH_pred_arr[r_use, b]),
                    imh=float(ImH_arr[r_use, b]),
                    t=float(t_bins[b]),
                    xB=float(xB_bins[b]),
                    Q2=float(Q2_bins[b]),
                    phi_rad=phi_grid.astype(float),
                )
                xs_all[r_use, b, :] = xs_r
                bsa_all[r_use, b, :] = bsa_r
        xs_std = np.std(xs_all, axis=0, ddof=0)
        bsa_std = np.std(bsa_all, axis=0, ddof=0)
        print(f"Computed ensemble bands using {n_use} replicas.")

    # Per-bin plots
    for b in range(B):
        title_tag = f"bin {b:02d}: t={t_bins[b]:.3g}, xB={xB_bins[b]:.4g}, Q2={Q2_bins[b]:.3g}, xi={xi_bins[b]:.4g}"

        xs_truth = None
        bsa_truth = None
        if y_true_bins is not None:
            xs_truth = y_true_bins[keys_sorted[b]]["xs"].astype(float)
            bsa_truth = y_true_bins[keys_sorted[b]]["bsa"].astype(float)

        plot_data_curve(
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
        plot_data_curve(
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
    print(f"  free fraction={frac_free:.3f}, film_gate fraction={frac_gate:.3f}")


if __name__ == "__main__":
    main()
