#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

KMI (no-DR) surface evaluation with TRUE vs MEAN surfaces on the same 3D plot,
including a transparent ±1σ band around the mean surface.

What this script produces
-------------------------
1) Bin-level (the 10 kinematic bins in the dataset):
   - CSV: bin_results_<TAG>.csv

2) Surface-level (random cloud of kinematics):
   - CSV: surface_results_<TAG>.csv
   - summary_<TAG>.json with RMSE/bias/coverage/pull stats
   - scatter and pull plots (still useful)

3) 3D surface plots (the main thing you asked for):
   For each fixed t-slice:
     - surface_ReH_t<...>_<TAG>.png  (truth wireframe + mean surface + ±1σ transparent band)
     - surface_ImH_t<...>_<TAG>.png

Important
---------
A 3D surface plot requires a 2D domain. Since the CFF surface depends on (xB,Q2,t),
we make plots at fixed t slices and show surfaces over (xB,Q2).

Truth
-----
Truth is defined as:
  - KM15 via gepard if dataset truth_source == "KM15"
  - Otherwise a toy truth (must match your generator toy formula)

Run
---
python evaluate.py
"""

import glob
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import tensorflow as tf

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# USER CONFIG
# =========================
VERSION_DIR = "output"
TAG = "v_1"

DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")
MODELS_DIR = os.path.join(VERSION_DIR, "replicas_kmi_no_dr")
WEIGHTS_GLOB = os.path.join(MODELS_DIR, f"replica_*_{TAG}.weights.h5")

OUT_DIR = os.path.join(VERSION_DIR, "eval_kmi_surface", TAG)

# How many random eval points for global surface cloud test
N_SURFACE_EVAL = 400
SURFACE_SEED = 20260109

# Physical cuts for random surface sampling (match generator philosophy)
W_MIN = 2.0
Y_MAX = 0.95

# Use how many replicas (None => all)
MAX_REPLICAS = None  # e.g. 50

# 3D surface plots (truth vs mean + band)
MAKE_3D_SURFACES = True
HEATMAP_T_SLICES = [-0.25, -0.40]  # choose representative t slices
GRID_N_XB = 45
GRID_N_Q2 = 45

# Ranges for the surface grid (auto from dataset bins + margin)
AUTO_RANGES = True
XB_RANGE = (0.18, 0.40)
Q2_RANGE = (1.2, 4.0)

# =========================
# END USER CONFIG
# =========================

M_PROTON = 0.9382720813  # GeV
_FLOATX = tf.float32


# ---------- basic helpers ----------
def _safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    xB = np.asarray(xB, dtype=float)
    return xB / (2.0 - xB)


def y_from_xB_Q2_E(xB: float, Q2: float, E: float) -> float:
    return float(Q2) / (2.0 * M_PROTON * float(xB) * float(E))


def W2_from_xB_Q2(xB: float, Q2: float) -> float:
    return float(M_PROTON**2 + float(Q2) * (1.0 / float(xB) - 1.0))


def is_physical_point(E: float, xB: float, Q2: float) -> bool:
    y = y_from_xB_Q2_E(xB, Q2, E)
    if not (0.0 < y < float(Y_MAX)):
        return False
    W2 = W2_from_xB_Q2(xB, Q2)
    if not (W2 > float(W_MIN) ** 2):
        return False
    return True


def build_features(t: np.ndarray, xB: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    xi = xi_from_xB(xB)
    return np.column_stack([t, xB, np.log(Q2), xi]).astype(np.float32)


# ---------- model (must match training) ----------
def make_cff_model(input_dim: int, feat_mean: np.ndarray, feat_std: np.ndarray,
                   reh_scale: float, imh_scale: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_dim,), name="features")
    mean_tf = tf.constant(feat_mean.reshape(1, -1), dtype=_FLOATX)
    std_tf = tf.constant(feat_std.reshape(1, -1), dtype=_FLOATX)

    x = tf.keras.layers.Lambda(lambda z: (z - mean_tf) / std_tf, name="standardize")(inp)
    x = tf.keras.layers.Dense(128, activation="relu", name="d1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="d2")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="d3")(x)

    raw = tf.keras.layers.Dense(2, activation="linear", name="raw")(x)
    reh = tf.keras.layers.Lambda(lambda r: float(reh_scale) * tf.tanh(r), name="ReH")(raw[:, 0])
    imh = tf.keras.layers.Lambda(lambda r: float(imh_scale) * tf.tanh(r), name="ImH")(raw[:, 1])
    out = tf.keras.layers.Concatenate(name="cff")([reh[:, None], imh[:, None]])
    return tf.keras.Model(inp, out)


# ---------- truth model ----------
def truth_H_toy(xB: float, Q2: float, t: float) -> complex:
    """
    Must match the toy truth formula used in your generator if TRUTH_SOURCE="toy".
    (This matches the kmi_generate_dataset_no_dr_v2.py toy by default.)
    """
    xi = xB / (2.0 - xB)
    reh = 2.0 + 1.0*(xi-0.20) + 0.25*np.log(Q2) + 0.3*(t+0.30)
    imh = 2.4 + 1.6*(xi-0.20) + 0.10*np.log(Q2) - 0.2*(t+0.30)
    return complex(float(reh), float(imh))


def truth_H_km15(E: float, xB: float, Q2: float, t: float) -> complex:
    import gepard as g
    from gepard.fits import th_KM15
    dp = g.DataPoint(
        xB=float(xB), t=float(t), Q2=float(Q2),
        phi=float(np.radians(10.0)),
        process="ep2epgamma",
        exptype="fixed target",
        in1energy=float(E),
        in1charge=-1,
        in1polarization=+1,
        observable="XS",
        fname="Trento",
    )
    return complex(float(th_KM15.ReH(dp)), float(th_KM15.ImH(dp)))


def get_truth_surface(E: float, truth_source: str,
                      xB: np.ndarray, Q2: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Re = np.zeros_like(xB, dtype=float)
    Im = np.zeros_like(xB, dtype=float)
    use_km15 = (str(truth_source).upper() == "KM15")
    if use_km15:
        for i in range(len(xB)):
            H = truth_H_km15(E, float(xB[i]), float(Q2[i]), float(t[i]))
            Re[i] = H.real
            Im[i] = H.imag
    else:
        for i in range(len(xB)):
            H = truth_H_toy(float(xB[i]), float(Q2[i]), float(t[i]))
            Re[i] = H.real
            Im[i] = H.imag
    return Re, Im


# ---------- metrics ----------
def summarize_errors(mu: np.ndarray, sig: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    mu = np.asarray(mu, dtype=float)
    sig = np.asarray(sig, dtype=float)
    truth = np.asarray(truth, dtype=float)

    err = mu - truth
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))

    sig_eff = np.where(sig > 0, sig, np.nan)
    pull = err / sig_eff
    pull = pull[np.isfinite(pull)]

    coverage = float(np.mean(np.abs(err) <= sig)) if np.all(sig >= 0) else float("nan")

    return dict(
        rmse=rmse,
        bias=bias,
        mae=mae,
        coverage_1sigma=coverage,
        pull_mean=float(np.mean(pull)) if pull.size else float("nan"),
        pull_std=float(np.std(pull, ddof=0)) if pull.size else float("nan"),
        n_pull=int(pull.size),
    )


# ---------- 3D surface plot ----------
def save_surface_compare_png(
    xB_grid: np.ndarray,
    Q2_grid: np.ndarray,
    Z_truth: np.ndarray,
    Z_mean: np.ndarray,
    Z_sigma: np.ndarray,
    title: str,
    zlabel: str,
    outpath: str,
):
    """
    3D comparison plot:
      - Truth surface: wireframe
      - Mean surface: solid
      - ±1σ band: two translucent surfaces
    NaNs are used to mask nonphysical points.
    """
    mask = np.isfinite(Z_mean) & np.isfinite(Z_truth) & np.isfinite(Z_sigma)
    if not np.any(mask):
        print(f"[WARN] No finite points to plot for: {outpath}")
        return

    Zt = np.where(mask, Z_truth, np.nan)
    Zm = np.where(mask, Z_mean, np.nan)
    Zu = np.where(mask, Z_mean + Z_sigma, np.nan)
    Zl = np.where(mask, Z_mean - Z_sigma, np.nan)

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # truth wireframe (doesn't occlude)
    ax.plot_wireframe(xB_grid, Q2_grid, Zt, rstride=1, cstride=1, linewidth=0.6)

    # mean surface
    ax.plot_surface(xB_grid, Q2_grid, Zm, linewidth=0, antialiased=True, alpha=0.85)

    # ±1σ band surfaces
    ax.plot_surface(xB_grid, Q2_grid, Zu, linewidth=0, antialiased=True, alpha=0.20)
    ax.plot_surface(xB_grid, Q2_grid, Zl, linewidth=0, antialiased=True, alpha=0.20)

    ax.set_xlabel(r"$x_B$")
    ax.set_ylabel(r"$Q^2\ \mathrm{(GeV^2)}$")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # helpful view
    ax.view_init(elev=22, azim=-135)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    _safe_mkdir(OUT_DIR)

    d = np.load(DATA_NPZ, allow_pickle=True)
    beam_energy = float(d["beam_energy"]) if "beam_energy" in d.files else 5.75
    truth_source = d["truth_source"].item() if "truth_source" in d.files else "toy"

    # bin definitions (10 points)
    t_bins = d["t_bins"].astype(float)
    xB_bins = d["xB_bins"].astype(float)
    Q2_bins = d["Q2_bins"].astype(float)
    B = len(xB_bins)

    # ranges
    if AUTO_RANGES:
        xb0, xb1 = float(np.min(xB_bins)), float(np.max(xB_bins))
        q20, q21 = float(np.min(Q2_bins)), float(np.max(Q2_bins))
        XB_RANGE_USE = (max(0.10, xb0 - 0.03), min(0.65, xb1 + 0.03))
        Q2_RANGE_USE = (max(0.8, q20 - 0.4), q21 + 0.4)
    else:
        XB_RANGE_USE = XB_RANGE
        Q2_RANGE_USE = Q2_RANGE

    # load replica weights
    weight_paths = sorted(glob.glob(WEIGHTS_GLOB))
    if not weight_paths:
        raise FileNotFoundError(f"No weights matched: {WEIGHTS_GLOB}")
    if MAX_REPLICAS is not None:
        weight_paths = weight_paths[:int(MAX_REPLICAS)]
    R = len(weight_paths)
    print(f"Loaded {R} replicas.")

    # meta from first replica
    base0 = weight_paths[0].replace(".weights.h5", "")
    meta0 = np.load(base0 + "_meta.npz", allow_pickle=True)
    feat_mean = meta0["feat_mean"].astype(np.float32)
    feat_std = meta0["feat_std"].astype(np.float32)
    reh_scale = float(meta0["REH_SCALE"])
    imh_scale = float(meta0["IMH_SCALE"])

    model = make_cff_model(4, feat_mean, feat_std, reh_scale, imh_scale)
    _ = model(tf.zeros((1, 4), dtype=_FLOATX), training=False)

    # --------- Surface cloud test (random points) ----------
    rng = np.random.default_rng(int(SURFACE_SEED))
    xB_list, Q2_list, t_list = [], [], []
    n_try = 0
    while len(xB_list) < int(N_SURFACE_EVAL):
        n_try += 1
        if n_try > 200000:
            raise RuntimeError("Failed to sample enough physical points. Relax ranges/cuts.")

        xBv = float(rng.uniform(*XB_RANGE_USE))
        logQ2 = float(rng.uniform(np.log(Q2_RANGE_USE[0]), np.log(Q2_RANGE_USE[1])))
        Q2v = float(np.exp(logQ2))

        # sample t from the observed range with margin
        tmin = float(np.min(t_bins)) - 0.05
        tmax = float(np.max(t_bins)) + 0.05
        tv = float(rng.uniform(tmin, tmax))
        if tv >= 0:
            continue

        if not is_physical_point(beam_energy, xBv, Q2v):
            continue

        xB_list.append(xBv); Q2_list.append(Q2v); t_list.append(tv)

    xB_s = np.array(xB_list, dtype=float)
    Q2_s = np.array(Q2_list, dtype=float)
    t_s = np.array(t_list, dtype=float)

    feat_s = build_features(t_s, xB_s, Q2_s)
    feat_s_tf = tf.constant(feat_s.astype(np.float32), dtype=_FLOATX)

    ReH_rep_s = np.zeros((R, len(xB_s)), dtype=float)
    ImH_rep_s = np.zeros((R, len(xB_s)), dtype=float)
    for r, wp in enumerate(weight_paths):
        model.load_weights(wp)
        cff = model(feat_s_tf, training=False).numpy().astype(float)
        ReH_rep_s[r, :] = cff[:, 0]
        ImH_rep_s[r, :] = cff[:, 1]

    ReH_mu_s = np.mean(ReH_rep_s, axis=0)
    ImH_mu_s = np.mean(ImH_rep_s, axis=0)
    ReH_sig_s = np.std(ReH_rep_s, axis=0, ddof=0)
    ImH_sig_s = np.std(ImH_rep_s, axis=0, ddof=0)

    ReH_true_s, ImH_true_s = get_truth_surface(
        E=beam_energy, truth_source=str(truth_source),
        xB=xB_s, Q2=Q2_s, t=t_s
    )

    # save surface CSV
    import pandas as pd
    surf_df = pd.DataFrame({
        "xB": xB_s,
        "xi": xi_from_xB(xB_s),
        "Q2": Q2_s,
        "t": t_s,
        "ReH_true": ReH_true_s,
        "ImH_true": ImH_true_s,
        "ReH_mu": ReH_mu_s,
        "ReH_sig": ReH_sig_s,
        "ImH_mu": ImH_mu_s,
        "ImH_sig": ImH_sig_s,
    })
    surf_csv = os.path.join(OUT_DIR, f"surface_results_{TAG}.csv")
    surf_df.to_csv(surf_csv, index=False)
    print("Wrote:", surf_csv)

    # summary JSON
    sum_ReH = summarize_errors(ReH_mu_s, ReH_sig_s, ReH_true_s)
    sum_ImH = summarize_errors(ImH_mu_s, ImH_sig_s, ImH_true_s)
    summary = dict(
        tag=TAG,
        truth_source=str(truth_source),
        beam_energy=float(beam_energy),
        n_replicas=int(R),
        n_surface_eval=int(len(xB_s)),
        ranges_used=dict(
            xB=[float(XB_RANGE_USE[0]), float(XB_RANGE_USE[1])],
            Q2=[float(Q2_RANGE_USE[0]), float(Q2_RANGE_USE[1])],
            t=[float(np.min(t_s)), float(np.max(t_s))],
        ),
        surface_metrics=dict(ReH=sum_ReH, ImH=sum_ImH),
    )
    sum_path = os.path.join(OUT_DIR, f"summary_{TAG}.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Wrote:", sum_path)

    # scatter and pull plots (still useful)
    plt.figure()
    plt.scatter(ReH_true_s, ReH_mu_s, s=10)
    lo = min(np.min(ReH_true_s), np.min(ReH_mu_s))
    hi = max(np.max(ReH_true_s), np.max(ReH_mu_s))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("ReH truth"); plt.ylabel("ReH mean")
    plt.title("Surface scatter: ReH truth vs mean")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"scatter_ReH_{TAG}.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(ImH_true_s, ImH_mu_s, s=10)
    lo = min(np.min(ImH_true_s), np.min(ImH_mu_s))
    hi = max(np.max(ImH_true_s), np.max(ImH_mu_s))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("ImH truth"); plt.ylabel("ImH mean")
    plt.title("Surface scatter: ImH truth vs mean")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"scatter_ImH_{TAG}.png"), dpi=200)
    plt.close()

    pull_ReH = (ReH_mu_s - ReH_true_s) / np.where(ReH_sig_s > 0, ReH_sig_s, np.nan)
    pull_ImH = (ImH_mu_s - ImH_true_s) / np.where(ImH_sig_s > 0, ImH_sig_s, np.nan)
    pull_ReH = pull_ReH[np.isfinite(pull_ReH)]
    pull_ImH = pull_ImH[np.isfinite(pull_ImH)]

    plt.figure()
    plt.hist(pull_ReH, bins="auto", edgecolor="black", alpha=0.75)
    plt.xlabel("(ReH_mean - ReH_true)/σ"); plt.ylabel("count")
    plt.title("Surface pull: ReH")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pull_ReH_{TAG}.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.hist(pull_ImH, bins="auto", edgecolor="black", alpha=0.75)
    plt.xlabel("(ImH_mean - ImH_true)/σ"); plt.ylabel("count")
    plt.title("Surface pull: ImH")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pull_ImH_{TAG}.png"), dpi=200)
    plt.close()

    # --------- 3D surface plots at fixed t slices ----------
    if MAKE_3D_SURFACES:
        xb0, xb1 = XB_RANGE_USE
        q20, q21 = Q2_RANGE_USE

        xb_1d = np.linspace(xb0, xb1, int(GRID_N_XB))
        q2_1d = np.exp(np.linspace(np.log(q20), np.log(q21), int(GRID_N_Q2)))  # log grid

        XB, Q2 = np.meshgrid(xb_1d, q2_1d)  # shapes (Ny, Nx)
        Ny, Nx = XB.shape

        for t_fix in HEATMAP_T_SLICES:
            # Build list of physical points in this slice
            pts = []
            ij = []
            for iy in range(Ny):
                for ix in range(Nx):
                    xBv = float(XB[iy, ix])
                    Q2v = float(Q2[iy, ix])
                    if is_physical_point(beam_energy, xBv, Q2v):
                        pts.append((float(t_fix), xBv, Q2v))
                        ij.append((iy, ix))

            if not pts:
                print(f"[WARN] No physical points for t={t_fix}")
                continue

            t_arr = np.array([p[0] for p in pts], dtype=float)
            xB_arr = np.array([p[1] for p in pts], dtype=float)
            Q2_arr = np.array([p[2] for p in pts], dtype=float)

            feat = build_features(t_arr, xB_arr, Q2_arr)
            feat_tf2 = tf.constant(feat.astype(np.float32), dtype=_FLOATX)

            # Replicas on grid points
            ReH_rep = np.zeros((R, len(pts)), dtype=float)
            ImH_rep = np.zeros((R, len(pts)), dtype=float)
            for r, wp in enumerate(weight_paths):
                model.load_weights(wp)
                cff = model(feat_tf2, training=False).numpy().astype(float)
                ReH_rep[r, :] = cff[:, 0]
                ImH_rep[r, :] = cff[:, 1]

            ReH_mean = np.mean(ReH_rep, axis=0)
            ReH_sig = np.std(ReH_rep, axis=0, ddof=0)
            ImH_mean = np.mean(ImH_rep, axis=0)
            ImH_sig = np.std(ImH_rep, axis=0, ddof=0)

            # Truth on grid points
            ReH_true, ImH_true = get_truth_surface(beam_energy, str(truth_source), xB_arr, Q2_arr, t_arr)

            # Fill maps
            ReH_truth_map = np.full((Ny, Nx), np.nan, dtype=float)
            ReH_mean_map  = np.full((Ny, Nx), np.nan, dtype=float)
            ReH_sig_map   = np.full((Ny, Nx), np.nan, dtype=float)

            ImH_truth_map = np.full((Ny, Nx), np.nan, dtype=float)
            ImH_mean_map  = np.full((Ny, Nx), np.nan, dtype=float)
            ImH_sig_map   = np.full((Ny, Nx), np.nan, dtype=float)

            for k, (iy, ix) in enumerate(ij):
                ReH_truth_map[iy, ix] = ReH_true[k]
                ReH_mean_map[iy, ix]  = ReH_mean[k]
                ReH_sig_map[iy, ix]   = ReH_sig[k]
                ImH_truth_map[iy, ix] = ImH_true[k]
                ImH_mean_map[iy, ix]  = ImH_mean[k]
                ImH_sig_map[iy, ix]   = ImH_sig[k]

            # filenames
            t_tag = f"{t_fix:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")

            save_surface_compare_png(
                xB_grid=XB, Q2_grid=Q2,
                Z_truth=ReH_truth_map,
                Z_mean=ReH_mean_map,
                Z_sigma=ReH_sig_map,
                title=rf"$\Re\mathcal{{H}}(x_B,Q^2)$ at $t={t_fix:.2f}$  (wire=truth, surface=mean, band=$\pm1\sigma$)",
                zlabel=r"$\Re\mathcal{H}$",
                outpath=os.path.join(OUT_DIR, f"surface_ReH_t{t_tag}_{TAG}.png"),
            )

            save_surface_compare_png(
                xB_grid=XB, Q2_grid=Q2,
                Z_truth=ImH_truth_map,
                Z_mean=ImH_mean_map,
                Z_sigma=ImH_sig_map,
                title=rf"$\Im\mathcal{{H}}(x_B,Q^2)$ at $t={t_fix:.2f}$  (wire=truth, surface=mean, band=$\pm1\sigma$)",
                zlabel=r"$\Im\mathcal{H}$",
                outpath=os.path.join(OUT_DIR, f"surface_ImH_t{t_tag}_{TAG}.png"),
            )

    print("\nWrote outputs to:", OUT_DIR)
    print("Main 3D surface plots:")
    for t_fix in HEATMAP_T_SLICES:
        t_tag = f"{t_fix:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        print(f"  surface_ReH_t{t_tag}_{TAG}.png")
        print(f"  surface_ImH_t{t_tag}_{TAG}.png")


if __name__ == "__main__":
    main()
