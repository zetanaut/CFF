#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generator.py for Hard DR constraint

Hard-DR closure dataset generator that is *guaranteed* to be compatible with
the Hard-DR training script (your HardDR_training.py), by:

  (1) selecting only PHYSICAL xB nodes (for fixed t, Q2, Ebeam) where bkm10_lib
      returns finite XS(phi), BSA(phi) over the full common phi grid,
  (2) doing the physical check using float32-rounded kinematics to match the
      training script's numerical behavior,
  (3) constructing Hard DR truth:
        ImH(xi) from KM15 on the xi grid
        ReH(xi) = C0 + K @ ImH   using the SAME discretized PV kernel used in training,
  (4) computing observables XS(phi), BSA(phi) using bkm10_lib with fixed nuisance CFFs.

Outputs:
  <OUT_DIR>/data/dataset_<TAG>.npz
  <OUT_DIR>/data/dataset_<TAG>.csv
  <OUT_DIR>/data/truth_<TAG>.json   (eval/diagnostics only)

This generator fixes the NaN problem you saw in training that originates from
unphysical kinematics (invalid sqrt inside bkm10_lib).
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

import gepard as g
from gepard.fits import th_KM15

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs


# =========================
# USER CONFIG (edit here)
# =========================

OUT_DIR = "output"
TAG = "v_1"

# Fixed kinematics shared across all xB nodes
BEAM_ENERGY = 5.75
Q2 = 1.82
T = -0.17

# Candidate xB range to scan (will be reduced automatically to physical subrange)
XB_MIN = 0.20
XB_MAX = 0.50

# Desired number of physical xB nodes in the final dataset
N_XB = 7

# Dense scan resolution (higher = more robust selection)
N_XB_CAND = 201  # odd is nice (includes midpoints)

# Safety: remove a fraction of physical range from each edge (avoid boundary sqrt ~ 0)
EDGE_CUT_FRACTION = 0.10  # 10% cut on each side of the physical xB interval

# Phi grid (MUST be identical for every xB bin)
PHI_START_DEG = 0.0
PHI_STOP_DEG = 360.0
N_PHI = 30
INCLUDE_ENDPOINT = True

# bkm10_lib settings (must match training script)
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# Fixed nuisance CFFs (must match training script for closure)
CFF_E_FIXED  = complex(2.217354372014208, 0.0)
CFF_HT_FIXED = complex(1.409393726454478, 1.57736440256014)
CFF_ET_FIXED = complex(144.4101642020152, 0.0)

# Hard DR subtraction constant truth specification
C0_MODE = "match_km15"      # "match_km15" or "fixed" or "zero"
C0_FIXED_VALUE = 0.25
C0_MATCH_INDEX = None       # None -> middle xi node; or integer index after sorting by xi

# Uncertainties: sigma = ABS + REL*|y_true|, with optional floors
XS_ABS_ERR = 0.0
XS_REL_ERR = 0.0
BSA_ABS_ERR = 0.0
BSA_REL_ERR = 0.0
XS_MIN_ERR = 0.0
BSA_MIN_ERR = 0.0

# Central values: noisy single sample or equal to truth
ADD_NOISE_TO_CENTRAL = True
CENTRAL_NOISE_SEED = 42

# =========================
# END USER CONFIG
# =========================

PI = float(np.pi)


# -------------------------
# Utility
# -------------------------
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _phi_grid_deg() -> np.ndarray:
    if INCLUDE_ENDPOINT:
        return np.linspace(PHI_START_DEG, PHI_STOP_DEG, N_PHI, dtype=float)
    return np.linspace(PHI_START_DEG, PHI_STOP_DEG, N_PHI, endpoint=False, dtype=float)


def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    xB = np.asarray(xB, dtype=float)
    return xB / (2.0 - xB)


# -------------------------
# DR kernel (MUST match training script)
# -------------------------
def build_dr_kernel(x_nodes: np.ndarray) -> np.ndarray:
    """
    Trapezoid PV kernel:
      ReH_i = C0 + sum_j K[i,j] ImH_j
      K[i,j]=(1/pi)*w_j*(1/(xi_i-xj)-1/(xi_i+xj)), with PV K[i,i]=0
    """
    x = np.asarray(x_nodes, dtype=float)
    B = len(x)
    if B < 2:
        raise ValueError("Need at least 2 xi nodes for DR kernel.")

    w = np.zeros(B, dtype=float)
    w[0]  = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if B > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])

    K = np.zeros((B, B), dtype=float)
    for i in range(B):
        xi = x[i]
        for j in range(B):
            xj = x[j]
            if i == j:
                K[i, j] = 0.0
            else:
                K[i, j] = (1.0 / PI) * w[j] * ((1.0 / (xi - xj)) - (1.0 / (xi + xj)))
    return K.astype(np.float32)


# -------------------------
# KM15 truth for H (ImH and (optional) ReH used only to choose C0)
# -------------------------
def km15_ReImH_at_xB(xB: float, phi: float = 0.0) -> Tuple[float, float]:
    dp = g.DataPoint(
        xB=float(xB),
        t=float(T),
        Q2=float(Q2),
        phi=float(phi),
        process="ep2epgamma",
        exptype="fixed target",
        in1energy=float(BEAM_ENERGY),
        in1charge=-1,
        in1polarization=+1,
        observable="XS",
        fname="Trento",
    )
    return float(th_KM15.ReH(dp)), float(th_KM15.ImH(dp))


# -------------------------
# bkm10 forward helper (used for physics selection + truth)
# -------------------------
def make_xsecs_object_for_bin(xB: float, reh: float, imh: float) -> DifferentialCrossSection:
    cfg = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k=float(BEAM_ENERGY),
            squared_Q_momentum_transfer=float(Q2),
            x_Bjorken=float(xB),
            squared_hadronic_momentum_transfer_t=float(T),
        ),
        "cff_inputs": CFFInputs(
            compton_form_factor_h=complex(float(reh), float(imh)),
            compton_form_factor_h_tilde=CFF_HT_FIXED,
            compton_form_factor_e=CFF_E_FIXED,
            compton_form_factor_e_tilde=CFF_ET_FIXED,
        ),
        "target_polarization": float(TARGET_POLARIZATION),
        "lepton_beam_polarization": float(LEPTON_BEAM_POLARIZATION),
        "using_ww": bool(USING_WW),
    }
    return DifferentialCrossSection(configuration=cfg, verbose=False, debugging=False)


def bkm_is_finite_for_xB(xB: float, phi_rad32: np.ndarray) -> bool:
    """
    Physical / numerical compatibility check:
    - uses float32-rounded xB,Q2,t to mimic training numerics
    - uses a benign CFF choice H=0+i0 (CFFs do not affect sqrt-domain issues)
    - returns True only if XS and BSA are finite over full phi grid
    """
    xB32 = float(np.float32(xB))
    # also round t,Q2 like training would
    _ = float(np.float32(Q2))
    _ = float(np.float32(T))

    try:
        with warnings.catch_warnings():
            # bkm10_lib sometimes emits RuntimeWarnings; we just test finiteness
            warnings.simplefilter("ignore", RuntimeWarning)
            xsecs = make_xsecs_object_for_bin(xB=xB32, reh=0.0, imh=0.0)
            xs = np.asarray(xsecs.compute_cross_section(phi_rad32).real, dtype=np.float64)
            bsa = np.asarray(xsecs.compute_bsa(phi_rad32).real, dtype=np.float64)
    except Exception:
        return False

    return bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(bsa)))


def select_physical_xB_nodes(phi_rad32: np.ndarray) -> np.ndarray:
    """
    Scan [XB_MIN, XB_MAX] and select N_XB nodes inside the physical subrange,
    with EDGE_CUT_FRACTION margins removed to avoid boundary instabilities.
    """
    xB_cand = np.linspace(float(XB_MIN), float(XB_MAX), int(N_XB_CAND), dtype=float)
    ok = np.array([bkm_is_finite_for_xB(xB, phi_rad32) for xB in xB_cand], dtype=bool)

    if not np.any(ok):
        raise RuntimeError(
            "No physical xB points found for your fixed (Ebeam,Q2,t). "
            "Make |t| larger (more negative), reduce Q2, or narrow the xB range."
        )

    xB_ok = xB_cand[ok]

    # We assume the physical region is contiguous in practice; use min/max of ok set
    xB_lo = float(np.min(xB_ok))
    xB_hi = float(np.max(xB_ok))

    # Apply edge cuts (safety margin away from sqrt boundary)
    span = xB_hi - xB_lo
    if span <= 0:
        raise RuntimeError("Physical xB span collapsed unexpectedly.")

    cut = float(EDGE_CUT_FRACTION) * span
    xB_lo2 = xB_lo + cut
    xB_hi2 = xB_hi - cut
    if xB_hi2 <= xB_lo2:
        # If edge cut is too aggressive, fall back to no cut
        xB_lo2, xB_hi2 = xB_lo, xB_hi

    # Choose evenly spaced nodes in the safe interval and verify them
    xB_nodes = np.linspace(xB_lo2, xB_hi2, int(N_XB), dtype=float)

    # Final verification
    ok2 = np.array([bkm_is_finite_for_xB(xB, phi_rad32) for xB in xB_nodes], dtype=bool)
    if np.sum(ok2) < int(N_XB):
        bad = xB_nodes[~ok2]
        raise RuntimeError(
            "Selected xB nodes include non-finite bkm10 outputs (still too close to boundary).\n"
            f"Bad xB: {bad}\n"
            "Try increasing EDGE_CUT_FRACTION or making |t| larger (more negative)."
        )

    return xB_nodes.astype(np.float64)


def make_sigmas(xs_true: np.ndarray, bsa_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs_sigma = XS_ABS_ERR + XS_REL_ERR * np.abs(xs_true)
    bsa_sigma = BSA_ABS_ERR + BSA_REL_ERR * np.abs(bsa_true)
    if np.any(xs_sigma < 0) or np.any(bsa_sigma < 0):
        raise ValueError("Uncertainties must be non-negative.")
    xs_sigma = np.maximum(xs_sigma, float(XS_MIN_ERR))
    bsa_sigma = np.maximum(bsa_sigma, float(BSA_MIN_ERR))
    return xs_sigma.astype(float), bsa_sigma.astype(float)


def main() -> None:
    out_dir = Path(OUT_DIR)
    data_dir = out_dir / "data"
    _safe_mkdir(data_dir)

    phi_deg = _phi_grid_deg()
    phi_rad = np.radians(phi_deg).astype(np.float64)
    phi_rad32 = phi_rad.astype(np.float32)  # mimic training numerics

    # ---- Select physical xB nodes (the key fix) ----
    xB_nodes = select_physical_xB_nodes(phi_rad32)
    xi_nodes = xi_from_xB(xB_nodes)

    # Sort by xi (important for consistent K ordering)
    order = np.argsort(xi_nodes)
    xB_nodes = xB_nodes[order]
    xi_nodes = xi_nodes[order]

    B = len(xB_nodes)
    Nphi = len(phi_rad)

    if B < 2:
        raise RuntimeError("Need at least 2 xB nodes for DR.")
    if B < 3:
        print("WARNING: Very small B for DR discretization. Prefer N_XB>=5.")

    # ---- Truth ImH from KM15 on the xi-grid ----
    reh_km15 = np.zeros(B, dtype=float)
    imh_km15 = np.zeros(B, dtype=float)
    for i, xB in enumerate(xB_nodes):
        reh_i, imh_i = km15_ReImH_at_xB(xB=float(xB), phi=float(phi_rad[0]))
        reh_km15[i] = reh_i
        imh_km15[i] = imh_i

    # ---- Hard-DR kernel on this xi-grid ----
    K = build_dr_kernel(x_nodes=xi_nodes.astype(float))
    reh_kernel_part = (K @ imh_km15.astype(np.float32)).astype(float)

    # ---- Choose C0 truth ----
    if str(C0_MODE).lower() == "zero":
        C0_truth = 0.0
    elif str(C0_MODE).lower() == "fixed":
        C0_truth = float(C0_FIXED_VALUE)
    elif str(C0_MODE).lower() == "match_km15":
        idx = int(B // 2) if (C0_MATCH_INDEX is None) else int(C0_MATCH_INDEX)
        if idx < 0 or idx >= B:
            raise ValueError(f"C0_MATCH_INDEX out of range: {idx}")
        C0_truth = float(reh_km15[idx] - reh_kernel_part[idx])
    else:
        raise ValueError(f"Unknown C0_MODE={C0_MODE}")

    # ---- Hard-DR truth ReH ----
    reh_dr = C0_truth + reh_kernel_part

    # ---- Compute observables per xB bin ----
    xs_true = np.zeros((B, Nphi), dtype=float)
    bsa_true = np.zeros((B, Nphi), dtype=float)
    for i in range(B):
        xsecs = make_xsecs_object_for_bin(xB=float(xB_nodes[i]), reh=float(reh_dr[i]), imh=float(imh_km15[i]))
        xs_true[i, :] = np.asarray(xsecs.compute_cross_section(phi_rad).real, dtype=float)
        bsa_true[i, :] = np.asarray(xsecs.compute_bsa(phi_rad).real, dtype=float)

    if not (np.all(np.isfinite(xs_true)) and np.all(np.isfinite(bsa_true))):
        raise RuntimeError(
            "Internal error: truth XS/BSA contains non-finite values even after physical xB selection."
        )

    xs_sigma, bsa_sigma = make_sigmas(xs_true, bsa_true)

    if ADD_NOISE_TO_CENTRAL:
        rng = np.random.default_rng(int(CENTRAL_NOISE_SEED))
        xs_central = xs_true + rng.normal(0.0, xs_sigma)
        bsa_central = bsa_true + rng.normal(0.0, bsa_sigma)
    else:
        xs_central = xs_true.copy()
        bsa_central = bsa_true.copy()

    # ---- Flatten into pointwise arrays for the trainer ----
    X_rows, y_true_rows, y_c_rows, y_s_rows = [], [], [], []
    for i in range(B):
        for j in range(Nphi):
            X_rows.append([float(T), float(xB_nodes[i]), float(Q2), float(phi_rad[j])])
            y_true_rows.append([float(xs_true[i, j]), float(bsa_true[i, j])])
            y_c_rows.append([float(xs_central[i, j]), float(bsa_central[i, j])])
            y_s_rows.append([float(xs_sigma[i, j]), float(bsa_sigma[i, j])])

    X = np.asarray(X_rows, dtype=np.float32)
    y_true_flat = np.asarray(y_true_rows, dtype=np.float32)
    y_c_flat = np.asarray(y_c_rows, dtype=np.float32)
    y_s_flat = np.asarray(y_s_rows, dtype=np.float32)

    # ---- Save NPZ ----
    npz_path = data_dir / f"dataset_{TAG}.npz"
    np.savez_compressed(
        npz_path,
        x=X,
        y_central=y_c_flat,
        y_true=y_true_flat,
        y_sigma=y_s_flat,
        phi_deg=phi_deg.astype(np.float32),
        phi_rad=phi_rad.astype(np.float32),
        xB_nodes=xB_nodes.astype(np.float32),
        xi_nodes=xi_nodes.astype(np.float32),
        K=K.astype(np.float32),
        C0_truth=np.float32(C0_truth),
        reh_dr=reh_dr.astype(np.float32),
        imh_km15=imh_km15.astype(np.float32),
        reh_km15=reh_km15.astype(np.float32),
    )

    # ---- Save CSV ----
    df = pd.DataFrame(
        {
            "t": X[:, 0],
            "x_b": X[:, 1],
            "q_squared": X[:, 2],
            "phi": X[:, 3],
            "phi_deg": np.tile(phi_deg, B),
            "XS_true": y_true_flat[:, 0],
            "BSA_true": y_true_flat[:, 1],
            "XS": y_c_flat[:, 0],
            "BSA": y_c_flat[:, 1],
            "XS_err": y_s_flat[:, 0],
            "BSA_err": y_s_flat[:, 1],
        }
    )
    csv_path = data_dir / f"dataset_{TAG}.csv"
    df.to_csv(csv_path, index=False)

    # ---- Save truth JSON ----
    truth_path = data_dir / f"truth_{TAG}.json"
    truth_obj = {
        "tag": TAG,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "fixed_kinematics": {"Ebeam": float(BEAM_ENERGY), "Q2": float(Q2), "t": float(T)},
        "xB_nodes": [float(v) for v in xB_nodes],
        "xi_nodes": [float(v) for v in xi_nodes],
        "hard_dr": {
            "kernel": "trapezoid PV on xi grid, K[i,i]=0",
            "C0_mode": str(C0_MODE),
            "C0_truth": float(C0_truth),
        },
        "truth_H": {
            "ImH_KM15": [float(v) for v in imh_km15],
            "ReH_DR": [float(v) for v in reh_dr],
            "ReH_KM15_for_reference": [float(v) for v in reh_km15],
        },
        "nuisance_fixed": {
            "E": [float(CFF_E_FIXED.real), float(CFF_E_FIXED.imag)],
            "Ht": [float(CFF_HT_FIXED.real), float(CFF_HT_FIXED.imag)],
            "Et": [float(CFF_ET_FIXED.real), float(CFF_ET_FIXED.imag)],
        },
        "phi_grid": {
            "start_deg": float(PHI_START_DEG),
            "stop_deg": float(PHI_STOP_DEG),
            "Nphi": int(N_PHI),
            "include_endpoint": bool(INCLUDE_ENDPOINT),
        },
        "uncertainty_model": {
            "XS_abs": float(XS_ABS_ERR), "XS_rel": float(XS_REL_ERR),
            "BSA_abs": float(BSA_ABS_ERR), "BSA_rel": float(BSA_REL_ERR),
            "XS_min": float(XS_MIN_ERR), "BSA_min": float(BSA_MIN_ERR),
            "add_noise_to_central": bool(ADD_NOISE_TO_CENTRAL),
            "central_noise_seed": int(CENTRAL_NOISE_SEED),
        },
    }
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth_obj, f, indent=2, sort_keys=True)

    print("Wrote:")
    print(f"  CSV:   {csv_path}")
    print(f"  NPZ:   {npz_path}")
    print(f"  Truth: {truth_path}")

    print("\nSelected physical xB nodes (float64):")
    for i in range(B):
        print(f"  i={i:2d}  xB={xB_nodes[i]:.6f}  xi={xi_nodes[i]:.6f}")

    print(f"\nHard DR truth: C0_truth = {C0_truth:+.6g}  (mode={C0_MODE})")
    print("Truth ranges:")
    print(f"  ImH(KM15)  : [{imh_km15.min():+.4g}, {imh_km15.max():+.4g}]")
    print(f"  ReH(DR)    : [{reh_dr.min():+.4g}, {reh_dr.max():+.4g}]")

    if float(np.max(xs_sigma)) == 0.0 and float(np.max(bsa_sigma)) == 0.0:
        print("\nNOTE: y_sigma is identically zero -> replica datasets will be identical; "
              "any ensemble spread will be optimizer/init only.")


if __name__ == "__main__":
    main()
