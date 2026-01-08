#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""closure_evaluate.py  (Python 3.9+)

Evaluation for the closure pipeline:
  - Histograms of extracted (ReH, ImH) across trained replica models
  - Pseudodata plots:
      * XS(phi) with error bars + inferred curve(s)
      * BSA(phi) with error bars + inferred curve(s)

Inputs (default):
  <VERSION_DIR>/data/dataset_<TAG>.npz
  <VERSION_DIR>/data/truth_<TAG>.json          (truth for evaluation only)
  <VERSION_DIR>/replicas/*.keras              (trained replicas)

What is "inferred"?
  - Each replica produces a scalar estimate (ReH_r, ImH_r) by averaging the model
    output over the phi grid (fixed kinematics).
  - The ensemble mean (ReH_mean, ImH_mean) is used as the central inferred curve.
  - Optionally, we compute an ensemble ±1σ band of curves by propagating all replicas
    through bkm10_lib.

IMPORTANT:
  - This script does NOT affect training. It reads truth only for plotting the
    reference (red) line and optional truth observable curves.

Edit the USER CONFIG block below and run:
  python closure_evaluate.py
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

# Silence macOS LibreSSL warning (harmless)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*")

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
MODELS_GLOB = os.path.join(VERSION_DIR, "replicas", f"*_{TAG}.keras")

OUT_DIR = os.path.join(VERSION_DIR, "eval")

# Model output indices for CFFs (matches our training model: [ReH, ImH, ...])
REH_INDEX = 0
IMH_INDEX = 1

# Use nuisance CFFs from:
#   "truth"     -> read E, H~, E~ from TRUTH_JSON
#   "hardcoded" -> use the constants below (must match training)
NUISANCE_SOURCE = "truth"

# Hardcoded nuisance CFFs if NUISANCE_SOURCE = "hardcoded"
CFF_E_HARDCODED  = complex(2.217354372014208, 0)
CFF_HT_HARDCODED = complex(1.409393726454478, 1.57736440256014)
CFF_ET_HARDCODED = complex(144.4101642020152, 0)

# Plot options
PLOT_ENSEMBLE_BAND = True    # compute and show ±1σ band of XS/BSA curves across replicas
PLOT_TRUTH_CURVE = True      # overlay the truth XS/BSA curve from truth CFFs (dashed)

HIST_BINS = "auto"         # or an int

# =========================
# END USER CONFIG
# =========================


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


def load_models(model_glob: str) -> Tuple[List[tf.keras.Model], List[str]]:
    paths = sorted(glob.glob(model_glob))
    if not paths:
        raise FileNotFoundError(f"No models matched: {model_glob}")
    models: List[tf.keras.Model] = []
    for p in paths:
        models.append(tf.keras.models.load_model(p, compile=False, safe_mode=False))
    return models, paths


def extract_replica_cffs(models: List[tf.keras.Model], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-replica mean and std-over-phi for ReH/ImH."""
    x_tf = tf.convert_to_tensor(X.astype(np.float32), dtype=tf.float32)

    reh_mean = []
    imh_mean = []
    reh_phi_std = []
    imh_phi_std = []

    for m in models:
        y = m(x_tf, training=False).numpy()
        reh = y[:, REH_INDEX].astype(float)
        imh = y[:, IMH_INDEX].astype(float)
        reh_mean.append(float(np.mean(reh)))
        imh_mean.append(float(np.mean(imh)))
        reh_phi_std.append(float(np.std(reh, ddof=0)))
        imh_phi_std.append(float(np.std(imh, ddof=0)))

    return (np.asarray(reh_mean), np.asarray(imh_mean),
            np.asarray(reh_phi_std), np.asarray(imh_phi_std))


def make_xsecs_object(
    beam_energy: float,
    q2: float,
    xb: float,
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
            lab_kinematics_k=float(beam_energy),
            squared_Q_momentum_transfer=float(q2),
            x_Bjorken=float(xb),
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


def forward_bkm10(
    phi_rad: np.ndarray,
    kin: Dict,
    settings: Dict,
    reh: float,
    imh: float,
    cff_e: complex,
    cff_ht: complex,
    cff_et: complex,
) -> Tuple[np.ndarray, np.ndarray]:
    xsecs = make_xsecs_object(
        beam_energy=float(kin["beam_energy"]),
        q2=float(kin["q_squared"]),
        xb=float(kin["x_b"]),
        t=float(kin["t"]),
        cff_h=complex(float(reh), float(imh)),
        cff_e=cff_e,
        cff_ht=cff_ht,
        cff_et=cff_et,
        using_ww=bool(settings["using_ww"]),
        target_polarization=float(settings["target_polarization"]),
        lepton_beam_polarization=float(settings["lepton_beam_polarization"]),
    )
    xs = np.asarray(xsecs.compute_cross_section(phi_rad).real, dtype=float)
    bsa = np.asarray(xsecs.compute_bsa(phi_rad).real, dtype=float)
    return xs, bsa


def plot_histogram(values: np.ndarray, truth: Optional[float], xlabel: str, title: str, outpath: str) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mu = float(np.mean(values))
    sig = float(np.std(values, ddof=0))

    plt.figure()
    plt.hist(values, bins=HIST_BINS, edgecolor="black", alpha=0.75)

    # mean and ±1σ
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


def plot_pseudodata_with_curves(
    phi_deg: np.ndarray,
    y_obs: np.ndarray,
    y_err: np.ndarray,
    y_curve: np.ndarray,
    curve_label: str,
    title: str,
    ylabel: str,
    outpath: str,
    y_band_std: Optional[np.ndarray] = None,
    truth_curve: Optional[np.ndarray] = None,
) -> None:
    # Sort by phi for clean plotting
    idx = np.argsort(phi_deg)
    x = phi_deg[idx]
    y = y_obs[idx]
    ye = y_err[idx]
    yc = y_curve[idx]

    plt.figure()
    plt.errorbar(x, y, yerr=ye, fmt="o", capsize=2, label="pseudodata")

    plt.plot(x, yc, label=curve_label)

    if y_band_std is not None:
        ys = y_band_std[idx]
        plt.fill_between(x, yc - ys, yc + ys, alpha=0.25, label=r"ensemble $\pm 1\sigma$")

    if truth_curve is not None:
        yt = truth_curve[idx]
        plt.plot(x, yt, linestyle="--", label="truth")

    plt.xlabel(r"$\phi$ (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    _safe_mkdir(OUT_DIR)

    d = load_npz(DATA_NPZ)
    X = d["x"].astype(np.float32)                 # (N,4): t, xB, Q2, phi(rad)
    y_central = d["y_central"].astype(np.float32) # (N,2): XS, BSA
    y_sigma = d["y_sigma"].astype(np.float32)     # (N,2): XS_err, BSA_err

    phi_rad = X[:, 3].astype(float)
    phi_deg = np.degrees(phi_rad).astype(float)

    # Load truth for evaluation display (and optional truth curve)
    truth = load_json(TRUTH_JSON)
    kin = {
        "beam_energy": truth["kinematics"]["beam_energy"],
        "q_squared": truth["kinematics"]["q_squared"],
        "x_b": truth["kinematics"]["x_b"],
        "t": truth["kinematics"]["t"],
    }
    settings = truth["bkm10_settings"]
    truth_cffs = truth["km15_truth_cffs"]
    reh_true = float(truth_cffs["cff_real_h_km15"])
    imh_true = float(truth_cffs["cff_imag_h_km15"])

    # Nuisance CFFs
    if NUISANCE_SOURCE == "truth":
        cff_e = complex(float(truth_cffs["cff_real_e_km15"]), float(truth_cffs["cff_imag_e_km15"]))
        cff_ht = complex(float(truth_cffs["cff_real_ht_km15"]), float(truth_cffs["cff_imag_ht_km15"]))
        cff_et = complex(float(truth_cffs["cff_real_et_km15"]), float(truth_cffs["cff_imag_et_km15"]))
    elif NUISANCE_SOURCE == "hardcoded":
        cff_e = CFF_E_HARDCODED
        cff_ht = CFF_HT_HARDCODED
        cff_et = CFF_ET_HARDCODED
    else:
        raise ValueError("NUISANCE_SOURCE must be 'truth' or 'hardcoded'")

    # Load models and extract replica CFFs
    models, model_paths = load_models(MODELS_GLOB)
    print(f"Loaded {len(models)} replicas from: {os.path.dirname(MODELS_GLOB)}")
    print(f"First model: {os.path.basename(model_paths[0])}")

    reh_rep, imh_rep, reh_phi_std, imh_phi_std = extract_replica_cffs(models, X)

    # Save extracted values
    out_csv = os.path.join(OUT_DIR, f"replica_cffs_{TAG}.csv")
    import pandas as pd  # local import to keep top deps minimal
    pd.DataFrame({
        "model_file": [os.path.basename(p) for p in model_paths],
        "ReH": reh_rep,
        "ImH": imh_rep,
        "ReH_std_over_phi": reh_phi_std,
        "ImH_std_over_phi": imh_phi_std,
    }).to_csv(out_csv, index=False)
    print(f"Saved per-replica CFFs to: {out_csv}")

    # Histograms
    reh_mu, reh_sig = plot_histogram(
        reh_rep,
        reh_true,
        xlabel=r"$\Re H$ (extracted per replica)",
        title="Closure: ReH across replicas",
        outpath=os.path.join(OUT_DIR, f"hist_ReH_{TAG}.png"),
    )
    imh_mu, imh_sig = plot_histogram(
        imh_rep,
        imh_true,
        xlabel=r"$\Im H$ (extracted per replica)",
        title="Closure: ImH across replicas",
        outpath=os.path.join(OUT_DIR, f"hist_ImH_{TAG}.png"),
    )

    print("\nHistogram summary:")
    print(f"  ReH: mean={reh_mu:.6g}, std={reh_sig:.6g}, truth={reh_true:.6g}, bias={reh_mu-reh_true:.6g}")
    print(f"  ImH: mean={imh_mu:.6g}, std={imh_sig:.6g}, truth={imh_true:.6g}, bias={imh_mu-imh_true:.6g}")

    # Predicted curves from inferred (ensemble mean) CFFs
    xs_mean, bsa_mean = forward_bkm10(
        phi_rad=phi_rad,
        kin=kin,
        settings=settings,
        reh=reh_mu,
        imh=imh_mu,
        cff_e=cff_e,
        cff_ht=cff_ht,
        cff_et=cff_et,
    )

    # Optional truth curves
    xs_truth_curve = None
    bsa_truth_curve = None
    if PLOT_TRUTH_CURVE:
        xs_truth_curve, bsa_truth_curve = forward_bkm10(
            phi_rad=phi_rad,
            kin=kin,
            settings=settings,
            reh=reh_true,
            imh=imh_true,
            cff_e=cff_e if NUISANCE_SOURCE == "hardcoded" else complex(float(truth_cffs["cff_real_e_km15"]), float(truth_cffs["cff_imag_e_km15"])),
            cff_ht=cff_ht if NUISANCE_SOURCE == "hardcoded" else complex(float(truth_cffs["cff_real_ht_km15"]), float(truth_cffs["cff_imag_ht_km15"])),
            cff_et=cff_et if NUISANCE_SOURCE == "hardcoded" else complex(float(truth_cffs["cff_real_et_km15"]), float(truth_cffs["cff_imag_et_km15"])),
        )

    # Optional ensemble band (propagate all replicas through bkm10)
    xs_std = None
    bsa_std = None
    if PLOT_ENSEMBLE_BAND:
        xs_all = []
        bsa_all = []
        for reh, imh in zip(reh_rep, imh_rep):
            xs_r, bsa_r = forward_bkm10(
                phi_rad=phi_rad,
                kin=kin,
                settings=settings,
                reh=float(reh),
                imh=float(imh),
                cff_e=cff_e,
                cff_ht=cff_ht,
                cff_et=cff_et,
            )
            xs_all.append(xs_r)
            bsa_all.append(bsa_r)
        xs_all = np.asarray(xs_all, dtype=float)   # (nrep, N)
        bsa_all = np.asarray(bsa_all, dtype=float) # (nrep, N)
        xs_std = np.std(xs_all, axis=0, ddof=0)
        bsa_std = np.std(bsa_all, axis=0, ddof=0)

    # Pseudodata plots
    xs_obs = y_central[:, 0].astype(float)
    bsa_obs = y_central[:, 1].astype(float)
    xs_err = y_sigma[:, 0].astype(float)
    bsa_err = y_sigma[:, 1].astype(float)

    plot_pseudodata_with_curves(
        phi_deg=phi_deg,
        y_obs=xs_obs,
        y_err=xs_err,
        y_curve=xs_mean,
        curve_label=r"inferred (ensemble mean)\, $H$",
        title="Pseudodata XS vs $\phi$ with inferred curve",
        ylabel="XS",
        outpath=os.path.join(OUT_DIR, f"xs_fit_{TAG}.png"),
        y_band_std=xs_std,
        truth_curve=xs_truth_curve,
    )

    plot_pseudodata_with_curves(
        phi_deg=phi_deg,
        y_obs=bsa_obs,
        y_err=bsa_err,
        y_curve=bsa_mean,
        curve_label=r"inferred (ensemble mean)\, $H$",
        title="Pseudodata BSA vs $\phi$ with inferred curve",
        ylabel="BSA",
        outpath=os.path.join(OUT_DIR, f"bsa_fit_{TAG}.png"),
        y_band_std=bsa_std,
        truth_curve=bsa_truth_curve,
    )

    print("\nWrote plots to:", OUT_DIR)
    print("  - hist_ReH_{TAG}.png")
    print("  - hist_ImH_{TAG}.png")
    print("  - xs_fit_{TAG}.png")
    print("  - bsa_fit_{TAG}.png")
    print("\nNote: if ReH_std_over_phi/ImH_std_over_phi are not near 0, your model is leaking phi into the CFF prediction.")


if __name__ == "__main__":
    main()
