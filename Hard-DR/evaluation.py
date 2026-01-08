#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py  (Python 3.9+)

Evaluate Hard-DR closure fits.

This script is compatible with the Hard-DR generator/trainer pair:
  - dataset_<TAG>.npz contains MULTIPLE (t,xB,Q2) bins and a COMMON phi grid
  - truth_<TAG>.json contains "fixed_kinematics", "xB_nodes", "xi_nodes",
    "truth_H" (ImH_KM15, ReH_DR), and "hard_dr" (C0_truth)
  - trained replicas are usually saved as weights + meta:
        replicas_hard_dr/replica_XXX_<TAG>.weights.h5
        replicas_hard_dr/replica_XXX_<TAG>_meta.npz
    (but this script can also load .keras models if you chose to save them)

Outputs:
  - per-replica parameters saved to CSV
  - histogram of C0 across replicas
  - ImH(xi) and ReH(xi) mean±std vs truth
  - XS(phi), BSA(phi) plots per xB bin with inferred mean curve and optional ±1σ band

Why this differs from your old closure_evaluate.py:
  - Hard DR produces a *function* ImH(xi) plus C0, not single scalar ReH/ImH
  - truth JSON schema changed (no "kinematics" field)
  - replicas are often saved as weights+meta rather than full .keras
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
# USER CONFIG (edit here)
# =========================

VERSION_DIR = "output"
TAG = "v_1"

DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")
TRUTH_JSON = os.path.join(VERSION_DIR, "data", f"truth_{TAG}.json")

# Replica location (weights+meta is the default recommended for subclassed models)
REPLICA_DIR = os.path.join(VERSION_DIR, "replicas_hard_dr")

# If you saved weights-only:
WEIGHTS_GLOB = os.path.join(REPLICA_DIR, f"replica_*_{TAG}.weights.h5")
META_SUFFIX = "_meta.npz"

# If you saved full keras models (not recommended for subclassed models):
KERAS_GLOB = os.path.join(REPLICA_DIR, f"replica_*_{TAG}.keras")

OUT_DIR = os.path.join(VERSION_DIR, "eval_hard_dr", TAG)

# Nuisance CFFs: read from truth JSON if available, else hardcode
NUISANCE_SOURCE = "truth"  # "truth" or "hardcoded"
CFF_E_HARDCODED  = complex(2.217354372014208, 0.0)
CFF_HT_HARDCODED = complex(1.409393726454478, 1.57736440256014)
CFF_ET_HARDCODED = complex(144.4101642020152, 0.0)

# Plot options
PLOT_ENSEMBLE_BAND = True     # for XS/BSA curves per bin
PLOT_TRUTH_CURVE = True       # overlay truth curves in XS/BSA plots
MAKE_NODE_HISTS = False       # set True to also make per-xi histograms for ImH/ReH

HIST_BINS = "auto"

# =========================
# END USER CONFIG
# =========================

_FLOATX = tf.float32


# -------------------------
# Utility
# -------------------------
def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    xB = np.asarray(xB, dtype=float)
    return xB / (2.0 - xB)


def group_by_kinematics(
    X: np.ndarray,
    y_central: np.ndarray,
    y_sigma: np.ndarray,
    decimals: int = 10,
) -> Dict[Tuple[float, float, float], Dict[str, np.ndarray]]:
    """
    Group pointwise arrays into unique (t, xB, Q2) bins, each containing vectors over phi.

    Returns dict keyed by (t, xB, Q2) with:
      phi, xs, bsa, xs_err, bsa_err  all shape (Nphi,)
    """
    t_arr = X[:, 0]
    xB_arr = X[:, 1]
    Q2_arr = X[:, 2]
    phi_arr = X[:, 3]

    key_arr = np.stack([t_arr, xB_arr, Q2_arr], axis=1)
    key_arr = np.round(key_arr, decimals=decimals)

    bins: Dict[Tuple[float, float, float], Dict[str, np.ndarray]] = {}
    for (t_r, xB_r, Q2_r) in np.unique(key_arr, axis=0):
        mask = np.all(key_arr == np.array([t_r, xB_r, Q2_r]), axis=1)
        phi = phi_arr[mask]
        order = np.argsort(phi)
        bins[(float(t_r), float(xB_r), float(Q2_r))] = {
            "phi": phi[order].astype(np.float64),
            "xs": y_central[mask, 0][order].astype(np.float64),
            "bsa": y_central[mask, 1][order].astype(np.float64),
            "xs_err": y_sigma[mask, 0][order].astype(np.float64),
            "bsa_err": y_sigma[mask, 1][order].astype(np.float64),
        }
    return bins


def assert_common_phi_grid(
    bins: Dict[Tuple[float, float, float], Dict[str, np.ndarray]],
    atol: float = 1e-6,
) -> np.ndarray:
    """Ensure all bins share the same phi grid; return phi grid."""
    keys = list(bins.keys())
    phi0 = bins[keys[0]]["phi"]
    for k in keys[1:]:
        phik = bins[k]["phi"]
        if len(phik) != len(phi0) or np.max(np.abs(phik - phi0)) > atol:
            raise ValueError(
                "This Hard-DR evaluation expects a common phi grid across xB bins.\n"
                f"Bin {keys[0]} has Nphi={len(phi0)}; bin {k} has Nphi={len(phik)}."
            )
    return phi0


# -------------------------
# Truth parsing (robust to schema variants)
# -------------------------
def parse_truth(truth: Dict) -> Dict:
    """
    Return a normalized dict with:
      Ebeam, Q2, t,
      xB_nodes, xi_nodes,
      C0_truth,
      ImH_truth (per node), ReH_truth (per node),
      nuisance CFFs,
      bkm settings if present.
    """
    # Kinematics: support multiple schema variants
    if "kinematics" in truth:
        kin = truth["kinematics"]
        Ebeam = float(kin.get("beam_energy", kin.get("Ebeam")))
        Q2 = float(kin.get("q_squared", kin.get("Q2")))
        t = float(kin.get("t"))
    elif "fixed_kinematics" in truth:
        kin = truth["fixed_kinematics"]
        Ebeam = float(kin.get("Ebeam", kin.get("beam_energy")))
        Q2 = float(kin.get("Q2", kin.get("q_squared")))
        t = float(kin.get("t"))
    elif "kinematics_fixed" in truth:
        kin = truth["kinematics_fixed"]
        Ebeam = float(kin.get("beam_energy", kin.get("Ebeam")))
        Q2 = float(kin.get("q_squared", kin.get("Q2")))
        t = float(kin.get("t"))
    else:
        raise KeyError("Truth JSON missing kinematics fields (expected 'fixed_kinematics' or 'kinematics').")

    xB_nodes = np.asarray(truth.get("xB_nodes", []), dtype=float)
    xi_nodes = np.asarray(truth.get("xi_nodes", []), dtype=float)

    # Hard DR truth values
    C0_truth = None
    if "hard_dr" in truth and "C0_truth" in truth["hard_dr"]:
        C0_truth = float(truth["hard_dr"]["C0_truth"])

    ImH_truth = None
    ReH_truth = None
    if "truth_H" in truth:
        th = truth["truth_H"]
        if "ImH_KM15" in th:
            ImH_truth = np.asarray(th["ImH_KM15"], dtype=float)
        if "ReH_DR" in th:
            ReH_truth = np.asarray(th["ReH_DR"], dtype=float)

    nuisance = None
    if "nuisance_fixed" in truth:
        nf = truth["nuisance_fixed"]
        # generator may store complex as [re, im]
        def _cplx(v):
            return complex(float(v[0]), float(v[1]))
        nuisance = {
            "E": _cplx(nf["E"]),
            "Ht": _cplx(nf["Ht"]),
            "Et": _cplx(nf["Et"]),
        }

    settings = truth.get("bkm10_settings", None)

    return {
        "Ebeam": Ebeam,
        "Q2": Q2,
        "t": t,
        "xB_nodes": xB_nodes,
        "xi_nodes": xi_nodes,
        "C0_truth": C0_truth,
        "ImH_truth": ImH_truth,
        "ReH_truth": ReH_truth,
        "nuisance": nuisance,
        "settings": settings,
    }


# -------------------------
# Forward model (bkm10_lib)
# -------------------------
def make_xsecs_object(
    Ebeam: float,
    Q2: float,
    xB: float,
    t: float,
    cff_h: complex,
    cff_e: complex,
    cff_ht: complex,
    cff_et: complex,
    using_ww: bool,
    target_polarization: float,
    lepton_beam_polarization: float,
) -> DifferentialCrossSection:
    cfg = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k=float(Ebeam),
            squared_Q_momentum_transfer=float(Q2),
            x_Bjorken=float(xB),
            squared_hadronic_momentum_transfer_t=float(t),
        ),
        "cff_inputs": CFFInputs(
            compton_form_factor_h=cff_h,
            compton_form_factor_h_tilde=cff_ht,
            compton_form_factor_e=cff_e,
            compton_form_factor_e_tilde=cff_et,
        ),
        "target_polarization": float(target_polarization),
        "lepton_beam_polarization": float(lepton_beam_polarization),
        "using_ww": bool(using_ww),
    }
    return DifferentialCrossSection(configuration=cfg, verbose=False, debugging=False)


def forward_bkm10_bin(
    phi_rad: np.ndarray,
    Ebeam: float,
    Q2: float,
    xB: float,
    t: float,
    reh: float,
    imh: float,
    cff_e: complex,
    cff_ht: complex,
    cff_et: complex,
    using_ww: bool,
    target_polarization: float,
    lepton_beam_polarization: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xsecs = make_xsecs_object(
        Ebeam=Ebeam,
        Q2=Q2,
        xB=xB,
        t=t,
        cff_h=complex(float(reh), float(imh)),
        cff_e=cff_e,
        cff_ht=cff_ht,
        cff_et=cff_et,
        using_ww=using_ww,
        target_polarization=target_polarization,
        lepton_beam_polarization=lepton_beam_polarization,
    )
    xs = np.asarray(xsecs.compute_cross_section(phi_rad).real, dtype=float)
    bsa = np.asarray(xsecs.compute_bsa(phi_rad).real, dtype=float)
    return xs, bsa


# -------------------------
# Hard-DR model definition (must match training architecture)
# -------------------------
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


def make_imh_network(seed: Optional[int] = None) -> tf.keras.Model:
    init = tf.keras.initializers.RandomUniform(minval=-0.3, maxval=0.3, seed=seed)
    kin_in = tf.keras.Input(shape=(3,), name="kin")  # (t, xB, Q2)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init)(kin_in)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init)(x)
    imh = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init, name="ImH")(x)
    return tf.keras.Model(inputs=kin_in, outputs=imh, name="ImHNet")


class HardDRHModel(tf.keras.Model):
    def __init__(self, K: np.ndarray, seed: Optional[int] = None):
        super().__init__(name="HardDRHModel")
        self.imh_net = make_imh_network(seed=seed)
        self.c0_layer = TrainableScalar(init_value=0.0, name="C0")
        self.K = tf.constant(np.asarray(K, dtype=np.float32), dtype=_FLOATX)

    def call(self, kin: tf.Tensor, training: bool = False):
        # kin: (B,3) tensor of (t, xB, Q2) for all bins
        imh = tf.squeeze(self.imh_net(kin, training=training), axis=1)  # (B,)
        c0_vec = self.c0_layer(kin)  # (B,)
        c0 = c0_vec[0]
        reh = c0 + tf.linalg.matvec(self.K, imh)  # (B,)
        return reh, imh, c0


def load_replicas_weights_and_meta(weights_glob: str) -> Tuple[List[str], List[str]]:
    wpaths = sorted(glob.glob(weights_glob))
    if not wpaths:
        return [], []
    mpaths = []
    for wp in wpaths:
        base = wp.replace(".weights.h5", "")
        mp = base + META_SUFFIX
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Missing meta file for weights: {wp}\nExpected: {mp}")
        mpaths.append(mp)
    return wpaths, mpaths


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
    if truth is not None:
        plt.axvline(float(truth), color="red", linestyle="--", linewidth=2, label="truth")
    plt.xlabel(xlabel)
    plt.ylabel("Replica count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return mu, sig


def plot_band_vs_x(x: np.ndarray, mean: np.ndarray, std: np.ndarray, truth: Optional[np.ndarray],
                   xlabel: str, ylabel: str, title: str, outpath: str) -> None:
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    plt.figure()
    plt.plot(x, mean, label="ensemble mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, label=r"ensemble $\pm 1\sigma$")
    if truth is not None:
        plt.plot(x, np.asarray(truth, dtype=float), linestyle="--", linewidth=2, label="truth")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_phi_curve(
    phi_deg: np.ndarray,
    y_obs: np.ndarray,
    y_err: np.ndarray,
    y_mean: np.ndarray,
    y_std: Optional[np.ndarray],
    truth_curve: Optional[np.ndarray],
    title: str,
    ylabel: str,
    outpath: str,
) -> None:
    idx = np.argsort(phi_deg)
    x = phi_deg[idx]
    y = y_obs[idx]
    ye = y_err[idx]
    ym = y_mean[idx]

    plt.figure()
    plt.errorbar(x, y, yerr=ye, fmt="o", capsize=2, label="pseudodata")
    plt.plot(x, ym, label="inferred (ensemble mean)")
    if y_std is not None:
        ys = y_std[idx]
        plt.fill_between(x, ym - ys, ym + ys, alpha=0.25, label=r"ensemble $\pm 1\sigma$")
    if truth_curve is not None:
        yt = truth_curve[idx]
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

    d = load_npz(DATA_NPZ)
    X = d["x"].astype(np.float32)                 # (N,4) [t, xB, Q2, phi]
    y_central = d["y_central"].astype(np.float32) # (N,2) [XS,BSA]
    y_sigma = d["y_sigma"].astype(np.float32)     # (N,2) [XS_err,BSA_err]

    bins = group_by_kinematics(X, y_central, y_sigma)
    if len(bins) < 2:
        raise ValueError(f"Hard-DR eval expects multiple xB bins; found {len(bins)} bins.")

    phi_grid = assert_common_phi_grid(bins)
    phi_deg = np.degrees(phi_grid).astype(float)

    # Sort bins by xi (must match training convention)
    keys_sorted = sorted(bins.keys(), key=lambda k: xi_from_xB(k[1]))
    t_bins = np.array([k[0] for k in keys_sorted], dtype=float)
    xB_bins = np.array([k[1] for k in keys_sorted], dtype=float)
    Q2_bins = np.array([k[2] for k in keys_sorted], dtype=float)
    xi_bins = xi_from_xB(xB_bins)

    B = len(keys_sorted)
    Nphi = len(phi_grid)

    # Build data arrays (B,Nphi)
    xs_obs = np.stack([bins[k]["xs"] for k in keys_sorted], axis=0)
    bsa_obs = np.stack([bins[k]["bsa"] for k in keys_sorted], axis=0)
    xs_err = np.stack([bins[k]["xs_err"] for k in keys_sorted], axis=0)
    bsa_err = np.stack([bins[k]["bsa_err"] for k in keys_sorted], axis=0)

    # Truth (optional but recommended for closure)
    truth = load_json(TRUTH_JSON)
    tr = parse_truth(truth)

    # Nuisance CFFs
    if NUISANCE_SOURCE == "truth" and tr["nuisance"] is not None:
        cff_e = tr["nuisance"]["E"]
        cff_ht = tr["nuisance"]["Ht"]
        cff_et = tr["nuisance"]["Et"]
    else:
        cff_e = CFF_E_HARDCODED
        cff_ht = CFF_HT_HARDCODED
        cff_et = CFF_ET_HARDCODED

    # bkm settings (fall back to defaults if missing)
    settings = tr["settings"] or {}
    using_ww = bool(settings.get("using_ww", True))
    target_pol = float(settings.get("target_polarization", 0.0))
    beam_pol = float(settings.get("lepton_beam_polarization", 0.0))

    # Use Ebeam from truth if present, else infer from dataset/training config (not stored in NPZ usually)
    Ebeam = float(tr["Ebeam"])

    # Kernel K: prefer dataset value, else meta per replica
    if "K" in d:
        K = d["K"].astype(np.float32)
    else:
        raise KeyError("dataset NPZ does not contain K; please regenerate with the Hard-DR generator that stores K.")

    # Kinematics tensor for the model
    kin_bins = np.stack([t_bins, xB_bins, Q2_bins], axis=1).astype(np.float32)  # (B,3)
    kin_tf = tf.convert_to_tensor(kin_bins, dtype=_FLOATX)

    # ---- Load replicas ----
    wpaths, mpaths = load_replicas_weights_and_meta(WEIGHTS_GLOB)

    use_weights = len(wpaths) > 0
    use_keras = False
    if not use_weights:
        kpaths = sorted(glob.glob(KERAS_GLOB))
        if kpaths:
            use_keras = True
        else:
            raise FileNotFoundError(
                "No replicas found.\n"
                f"Tried weights glob: {WEIGHTS_GLOB}\n"
                f"Tried keras glob:   {KERAS_GLOB}"
            )

    reh_rep = []
    imh_rep = []
    c0_rep = []
    replica_ids = []

    if use_weights:
        print(f"Loading {len(wpaths)} replicas from weights+meta in {REPLICA_DIR}")
        for wp, mp in zip(wpaths, mpaths):
            meta = np.load(mp, allow_pickle=True)
            # Prefer K from meta if present (guarantees exact match to training)
            K_rep = meta["K"].astype(np.float32) if "K" in meta.files else K

            model = HardDRHModel(K=K_rep, seed=None)
            # Build model variables by calling once
            _ = model(kin_tf, training=False)
            model.load_weights(wp)

            reh, imh, c0 = model(kin_tf, training=False)
            reh_rep.append(reh.numpy().astype(float))
            imh_rep.append(imh.numpy().astype(float))
            c0_rep.append(float(c0.numpy()))

            replica_ids.append(os.path.basename(wp).replace(".weights.h5", ""))

    else:
        # Full .keras load (only if you saved that way)
        kpaths = sorted(glob.glob(KERAS_GLOB))
        print(f"Loading {len(kpaths)} replicas from .keras in {REPLICA_DIR}")
        for kp in kpaths:
            # Subclassed models often need safe_mode=False; if this fails, use weights-only saving.
            m = tf.keras.models.load_model(kp, compile=False, safe_mode=False)
            reh, imh, c0 = m(kin_tf, training=False)  # must return (reh,imh,c0)
            reh_rep.append(np.asarray(reh).astype(float))
            imh_rep.append(np.asarray(imh).astype(float))
            c0_rep.append(float(np.asarray(c0).item()))
            replica_ids.append(os.path.basename(kp).replace(".keras", ""))

    reh_rep = np.asarray(reh_rep, dtype=float)  # (R,B)
    imh_rep = np.asarray(imh_rep, dtype=float)  # (R,B)
    c0_rep = np.asarray(c0_rep, dtype=float)    # (R,)

    R = reh_rep.shape[0]
    assert reh_rep.shape == (R, B)

    # ---- Truth alignment (ensure same xB ordering) ----
    ImH_truth = tr["ImH_truth"]
    ReH_truth = tr["ReH_truth"]
    C0_truth = tr["C0_truth"]

    # If truth arrays exist but their xB ordering differs, reorder by matching xB nodes.
    # We'll match by nearest xB within tolerance.
    if ImH_truth is not None and len(ImH_truth) == B and tr["xB_nodes"] is not None and len(tr["xB_nodes"]) == B:
        xB_truth = np.asarray(tr["xB_nodes"], dtype=float)
        # Build permutation p such that xB_truth[p[i]] ~ xB_bins[i]
        p = []
        for xb in xB_bins:
            j = int(np.argmin(np.abs(xB_truth - xb)))
            p.append(j)
        p = np.asarray(p, dtype=int)
        ImH_truth = np.asarray(ImH_truth)[p]
        if ReH_truth is not None and len(ReH_truth) == B:
            ReH_truth = np.asarray(ReH_truth)[p]

    # ---- Save per-replica outputs ----
    out_csv = os.path.join(OUT_DIR, f"replica_hard_dr_params_{TAG}.csv")
    cols = {"replica": replica_ids, "C0": c0_rep}
    for i in range(B):
        cols[f"xB_{i:02d}"] = np.full(R, xB_bins[i])
        cols[f"xi_{i:02d}"] = np.full(R, xi_bins[i])
        cols[f"ImH_{i:02d}"] = imh_rep[:, i]
        cols[f"ReH_{i:02d}"] = reh_rep[:, i]
    import pandas as pd
    pd.DataFrame(cols).to_csv(out_csv, index=False)
    print(f"Saved per-replica DR parameters to: {out_csv}")

    # ---- Summary stats ----
    c0_mu = float(np.mean(c0_rep)); c0_sig = float(np.std(c0_rep, ddof=0))
    imh_mu = np.mean(imh_rep, axis=0); imh_sig = np.std(imh_rep, axis=0, ddof=0)
    reh_mu = np.mean(reh_rep, axis=0); reh_sig = np.std(reh_rep, axis=0, ddof=0)

    # ---- Plots: C0 histogram + bands vs xi ----
    plot_hist(
        c0_rep,
        C0_truth,
        xlabel=r"$C_0$ (subtraction constant)",
        title="Hard-DR closure: $C_0$ across replicas",
        outpath=os.path.join(OUT_DIR, f"hist_C0_{TAG}.png"),
    )

    plot_band_vs_x(
        x=xi_bins,
        mean=imh_mu,
        std=imh_sig,
        truth=ImH_truth if ImH_truth is not None else None,
        xlabel=r"$\xi$",
        ylabel=r"$\Im\,\mathcal{H}$",
        title=r"Hard-DR closure: $\Im\,\mathcal{H}(\xi)$",
        outpath=os.path.join(OUT_DIR, f"band_ImH_vs_xi_{TAG}.png"),
    )

    plot_band_vs_x(
        x=xi_bins,
        mean=reh_mu,
        std=reh_sig,
        truth=ReH_truth if ReH_truth is not None else None,
        xlabel=r"$\xi$",
        ylabel=r"$\Re\,\mathcal{H}$",
        title=r"Hard-DR closure: $\Re\,\mathcal{H}(\xi)$",
        outpath=os.path.join(OUT_DIR, f"band_ReH_vs_xi_{TAG}.png"),
    )

    # Optional: node-by-node histograms
    if MAKE_NODE_HISTS:
        for i in range(B):
            plot_hist(
                imh_rep[:, i],
                float(ImH_truth[i]) if ImH_truth is not None else None,
                xlabel=rf"$\Im\,\mathcal{{H}}(\xi_{{{i}}})$",
                title=rf"Hard-DR closure: $\Im\,\mathcal{{H}}$ at $\xi={xi_bins[i]:.4g}$",
                outpath=os.path.join(OUT_DIR, f"hist_ImH_node{i:02d}_{TAG}.png"),
            )
            plot_hist(
                reh_rep[:, i],
                float(ReH_truth[i]) if ReH_truth is not None else None,
                xlabel=rf"$\Re\,\mathcal{{H}}(\xi_{{{i}}})$",
                title=rf"Hard-DR closure: $\Re\,\mathcal{{H}}$ at $\xi={xi_bins[i]:.4g}$",
                outpath=os.path.join(OUT_DIR, f"hist_ReH_node{i:02d}_{TAG}.png"),
            )

    # ---- Pseudodata curves: per xB bin ----
    for b in range(B):
        # Ensemble mean curve
        xs_mean, bsa_mean = forward_bkm10_bin(
            phi_rad=phi_grid,
            Ebeam=Ebeam,
            Q2=float(Q2_bins[b]),
            xB=float(xB_bins[b]),
            t=float(t_bins[b]),
            reh=float(reh_mu[b]),
            imh=float(imh_mu[b]),
            cff_e=cff_e,
            cff_ht=cff_ht,
            cff_et=cff_et,
            using_ww=using_ww,
            target_polarization=target_pol,
            lepton_beam_polarization=beam_pol,
        )

        # Truth curve (optional)
        xs_truth = None
        bsa_truth = None
        if PLOT_TRUTH_CURVE and (ImH_truth is not None) and (ReH_truth is not None):
            xs_truth, bsa_truth = forward_bkm10_bin(
                phi_rad=phi_grid,
                Ebeam=Ebeam,
                Q2=float(Q2_bins[b]),
                xB=float(xB_bins[b]),
                t=float(t_bins[b]),
                reh=float(ReH_truth[b]),
                imh=float(ImH_truth[b]),
                cff_e=cff_e,
                cff_ht=cff_ht,
                cff_et=cff_et,
                using_ww=using_ww,
                target_polarization=target_pol,
                lepton_beam_polarization=beam_pol,
            )

        # Ensemble band (optional, propagate all replicas)
        xs_std = None
        bsa_std = None
        if PLOT_ENSEMBLE_BAND:
            xs_all = []
            bsa_all = []
            for r in range(R):
                xs_r, bsa_r = forward_bkm10_bin(
                    phi_rad=phi_grid,
                    Ebeam=Ebeam,
                    Q2=float(Q2_bins[b]),
                    xB=float(xB_bins[b]),
                    t=float(t_bins[b]),
                    reh=float(reh_rep[r, b]),
                    imh=float(imh_rep[r, b]),
                    cff_e=cff_e,
                    cff_ht=cff_ht,
                    cff_et=cff_et,
                    using_ww=using_ww,
                    target_polarization=target_pol,
                    lepton_beam_polarization=beam_pol,
                )
                xs_all.append(xs_r)
                bsa_all.append(bsa_r)
            xs_all = np.asarray(xs_all, dtype=float)   # (R,Nphi)
            bsa_all = np.asarray(bsa_all, dtype=float) # (R,Nphi)
            xs_std = np.std(xs_all, axis=0, ddof=0)
            bsa_std = np.std(bsa_all, axis=0, ddof=0)

        # Plots for this bin
        xb = float(xB_bins[b]); xi = float(xi_bins[b])
        plot_phi_curve(
            phi_deg=phi_deg,
            y_obs=xs_obs[b, :],
            y_err=xs_err[b, :],
            y_mean=xs_mean,
            y_std=xs_std,
            truth_curve=xs_truth,
            title=rf"XS vs $\phi$ (Hard-DR) at $x_B={xb:.3f}$, $\xi={xi:.3f}$",
            ylabel="XS",
            outpath=os.path.join(OUT_DIR, f"xs_bin{b:02d}_xB{xb:.3f}_{TAG}.png"),
        )

        plot_phi_curve(
            phi_deg=phi_deg,
            y_obs=bsa_obs[b, :],
            y_err=bsa_err[b, :],
            y_mean=bsa_mean,
            y_std=bsa_std,
            truth_curve=bsa_truth,
            title=rf"BSA vs $\phi$ (Hard-DR) at $x_B={xb:.3f}$, $\xi={xi:.3f}$",
            ylabel="BSA",
            outpath=os.path.join(OUT_DIR, f"bsa_bin{b:02d}_xB{xb:.3f}_{TAG}.png"),
        )

    # ---- Console summary ----
    print("\nHard-DR evaluation summary:")
    print(f"  replicas: R={R}, bins: B={B}, Nphi={Nphi}")
    print(f"  C0: mean={c0_mu:+.6g}, std={c0_sig:.6g}, truth={C0_truth if C0_truth is not None else 'N/A'}")
    if ImH_truth is not None:
        print("  ImH node-wise bias (mean - truth):")
        for i in range(B):
            print(f"    i={i:02d} xi={xi_bins[i]:.4f}  bias={imh_mu[i]-ImH_truth[i]:+.4g}  std={imh_sig[i]:.4g}")
    if ReH_truth is not None:
        print("  ReH node-wise bias (mean - truth):")
        for i in range(B):
            print(f"    i={i:02d} xi={xi_bins[i]:.4f}  bias={reh_mu[i]-ReH_truth[i]:+.4g}  std={reh_sig[i]:.4g}")

    print("\nWrote outputs to:", OUT_DIR)
    print("  - hist_C0_*.png")
    print("  - band_ImH_vs_xi_*.png")
    print("  - band_ReH_vs_xi_*.png")
    print("  - xs_bin*.png, bsa_bin*.png")
    print("  - replica_hard_dr_params_*.csv")


if __name__ == "__main__":
    main()
