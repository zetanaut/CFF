#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py

Generate a closure dataset for a "Gated DR" model with a differentiable PV operator.

Truth construction:
  - ImH_truth(xi) taken from KM15 (Gepard) on a set of physical xB nodes.
  - ReH_DR(xi) computed from ImH_truth via a discretized PV dispersion relation:
        ReH_DR = C0 + K @ ImH_truth
    where K is the PV kernel on the xi grid (trapezoid weights).
  - ReH_free(xi) = KM15 ReH(xi) (used only as a "DR-violating reference").
  - Final truth:
        ReH_truth = g_truth * ReH_DR + (1 - g_truth) * ReH_free
    If g_truth=1 -> exact-DR closure
    If g_truth=0 -> fully DR-violating closure (truth uses KM15 ReH instead)

Observables:
  - XS(phi), BSA(phi) computed with bkm10_lib at each (t, xB, Q2) bin.

Key feature:
  - Automatically selects xB nodes where bkm10_lib returns finite XS/BSA on the full phi grid,
    avoiding the NaN sqrt boundary problem.

Outputs (under <OUT_DIR>/data):
  - dataset_<TAG>.npz  (trainer input)
  - dataset_<TAG>.csv
  - truth_<TAG>.json   (evaluation only)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import gepard as g
from gepard.fits import th_KM15

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs


# =========================
# USER CONFIG
# =========================
OUT_DIR = "output"
TAG = "v_1"

# Fixed kinematics shared across all xB nodes
BEAM_ENERGY = 5.75
Q2 = 1.82
T = -0.17

# Candidate xB range to scan; will be reduced automatically to the physical subrange
XB_MIN = 0.20
XB_MAX = 0.50

# Number of xB nodes to KEEP in final dataset
N_XB = 7

# Dense scan resolution for physical selection
N_XB_CAND = 201

# Edge cut away from physical boundaries (sqrt argument ~ 0)
EDGE_CUT_FRACTION = 0.10

# Phi grid (must be common across all bins)
PHI_START_DEG = 0.0
PHI_STOP_DEG = 360.0
N_PHI = 30
INCLUDE_ENDPOINT = True

# bkm10_lib settings (must match trainer)
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# Fixed nuisance CFFs (must match trainer for closure)
CFF_E_FIXED  = complex(2.217354372014208, 0.0)
CFF_HT_FIXED = complex(1.409393726454478, 1.57736440256014)
CFF_ET_FIXED = complex(144.4101642020152, 0.0)

# --- Hard-DR subtraction constant ---
# Choose C0 so that ReH_DR matches ReH_free at one reference node (recommended).
C0_MODE = "match_free"   # "match_free" | "fixed" | "zero"
C0_FIXED_VALUE = 0.0
C0_MATCH_INDEX = None     # None -> middle node after xi sorting

# --- Truth gate ---
# g_truth=1 => exact-DR closure
# g_truth=0 => truth uses ReH_free (KM15) instead of DR
G_TRUTH = 1.0

# Uncertainty model: sigma = ABS + REL*|y_true| with floors
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


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _phi_grid_deg() -> np.ndarray:
    if INCLUDE_ENDPOINT:
        return np.linspace(PHI_START_DEG, PHI_STOP_DEG, N_PHI, dtype=float)
    return np.linspace(PHI_START_DEG, PHI_STOP_DEG, N_PHI, endpoint=False, dtype=float)


def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    xB = np.asarray(xB, dtype=float)
    return xB / (2.0 - xB)


def build_dr_kernel_trapezoid_pv(x_nodes: np.ndarray) -> np.ndarray:
    """
    PV trapezoid kernel on xi-grid:
      ReH_i = C0 + sum_j K[i,j] ImH_j
      K[i,j]=(1/pi)*w_j*(1/(xi_i-xj)-1/(xi_i+xj)), with PV K[i,i]=0
    """
    x = np.asarray(x_nodes, dtype=float)
    B = len(x)
    if B < 2:
        raise ValueError("Need at least 2 xi nodes.")

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
                K[i, j] = 0.0  # PV prescription
            else:
                K[i, j] = (1.0 / PI) * w[j] * ((1.0 / (xi - xj)) - (1.0 / (xi + xj)))
    return K.astype(np.float32)


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
    Physics/compatibility check to avoid sqrt-domain NaNs:
    verify bkm10 outputs finite XS and BSA on the full phi grid.
    Use float32-rounded kinematics to mimic training numerics.
    """
    xB32 = float(np.float32(xB))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xsecs = make_xsecs_object_for_bin(xB=xB32, reh=0.0, imh=0.0)
            xs = np.asarray(xsecs.compute_cross_section(phi_rad32).real, dtype=np.float64)
            bsa = np.asarray(xsecs.compute_bsa(phi_rad32).real, dtype=np.float64)
    except Exception:
        return False
    return bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(bsa)))


def select_physical_xB_nodes(phi_rad32: np.ndarray) -> np.ndarray:
    xB_cand = np.linspace(float(XB_MIN), float(XB_MAX), int(N_XB_CAND), dtype=float)
    ok = np.array([bkm_is_finite_for_xB(xB, phi_rad32) for xB in xB_cand], dtype=bool)

    if not np.any(ok):
        raise RuntimeError("No physical xB points found. Try more negative t, lower Q2, or narrower xB range.")

    xB_ok = xB_cand[ok]
    xB_lo = float(np.min(xB_ok))
    xB_hi = float(np.max(xB_ok))

    span = xB_hi - xB_lo
    cut = float(EDGE_CUT_FRACTION) * span
    xB_lo2 = xB_lo + cut
    xB_hi2 = xB_hi - cut
    if xB_hi2 <= xB_lo2:
        xB_lo2, xB_hi2 = xB_lo, xB_hi

    xB_nodes = np.linspace(xB_lo2, xB_hi2, int(N_XB), dtype=float)

    # verify final nodes
    ok2 = np.array([bkm_is_finite_for_xB(xB, phi_rad32) for xB in xB_nodes], dtype=bool)
    if np.sum(ok2) < int(N_XB):
        bad = xB_nodes[~ok2]
        raise RuntimeError(
            "Selected xB nodes still include non-finite bkm10 output.\n"
            f"Bad xB: {bad}\n"
            "Increase EDGE_CUT_FRACTION or change (t,Q2,xB range)."
        )

    return xB_nodes.astype(np.float64)


def make_sigmas(xs_true: np.ndarray, bsa_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs_sigma = XS_ABS_ERR + XS_REL_ERR * np.abs(xs_true)
    bsa_sigma = BSA_ABS_ERR + BSA_REL_ERR * np.abs(bsa_true)
    xs_sigma = np.maximum(xs_sigma, float(XS_MIN_ERR))
    bsa_sigma = np.maximum(bsa_sigma, float(BSA_MIN_ERR))
    return xs_sigma.astype(float), bsa_sigma.astype(float)


def main() -> None:
    out_dir = Path(OUT_DIR)
    data_dir = out_dir / "data"
    _safe_mkdir(data_dir)

    phi_deg = _phi_grid_deg()
    phi_rad = np.radians(phi_deg).astype(np.float64)
    phi_rad32 = phi_rad.astype(np.float32)

    # 1) pick physical xB nodes
    xB_nodes = select_physical_xB_nodes(phi_rad32)
    xi_nodes = xi_from_xB(xB_nodes)

    # sort by xi
    order = np.argsort(xi_nodes)
    xB_nodes = xB_nodes[order]
    xi_nodes = xi_nodes[order]

    B = len(xB_nodes)
    Nphi = len(phi_rad)

    # 2) truth from KM15: ImH and ReH_free
    ReH_free = np.zeros(B, dtype=float)
    ImH_truth = np.zeros(B, dtype=float)
    for i, xB in enumerate(xB_nodes):
        reh_i, imh_i = km15_ReImH_at_xB(xB=float(xB), phi=float(phi_rad[0]))
        ReH_free[i] = reh_i
        ImH_truth[i] = imh_i

    # 3) DR kernel + subtraction constant -> ReH_DR
    K = build_dr_kernel_trapezoid_pv(x_nodes=xi_nodes.astype(float))
    ReH_kernel_part = (K @ ImH_truth.astype(np.float32)).astype(float)

    if str(C0_MODE).lower() == "zero":
        C0_truth = 0.0
    elif str(C0_MODE).lower() == "fixed":
        C0_truth = float(C0_FIXED_VALUE)
    elif str(C0_MODE).lower() == "match_free":
        idx = int(B // 2) if (C0_MATCH_INDEX is None) else int(C0_MATCH_INDEX)
        if idx < 0 or idx >= B:
            raise ValueError("C0_MATCH_INDEX out of range.")
        C0_truth = float(ReH_free[idx] - ReH_kernel_part[idx])
    else:
        raise ValueError("C0_MODE must be 'match_free', 'fixed', or 'zero'.")

    ReH_DR = C0_truth + ReH_kernel_part

    # 4) gated truth ReH
    g_truth = float(G_TRUTH)
    g_truth = max(0.0, min(1.0, g_truth))
    ReH_truth = g_truth * ReH_DR + (1.0 - g_truth) * ReH_free
    deltaReH_truth = ReH_free - ReH_DR  # the "correction direction"

    # 5) compute observables per bin
    xs_true = np.zeros((B, Nphi), dtype=float)
    bsa_true = np.zeros((B, Nphi), dtype=float)
    for i in range(B):
        xsecs = make_xsecs_object_for_bin(xB=float(xB_nodes[i]), reh=float(ReH_truth[i]), imh=float(ImH_truth[i]))
        xs_true[i, :] = np.asarray(xsecs.compute_cross_section(phi_rad).real, dtype=float)
        bsa_true[i, :] = np.asarray(xsecs.compute_bsa(phi_rad).real, dtype=float)

    if not (np.all(np.isfinite(xs_true)) and np.all(np.isfinite(bsa_true))):
        raise RuntimeError("Truth XS/BSA is non-finite; adjust kinematics selection or xB range.")

    xs_sigma, bsa_sigma = make_sigmas(xs_true, bsa_true)

    if ADD_NOISE_TO_CENTRAL:
        rng = np.random.default_rng(int(CENTRAL_NOISE_SEED))
        xs_central = xs_true + rng.normal(0.0, xs_sigma)
        bsa_central = bsa_true + rng.normal(0.0, bsa_sigma)
    else:
        xs_central = xs_true.copy()
        bsa_central = bsa_true.copy()

    # 6) flatten to pointwise arrays expected by trainers
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

    # save NPZ
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
        g_truth=np.float32(g_truth),
        ImH_truth=ImH_truth.astype(np.float32),
        ReH_DR=ReH_DR.astype(np.float32),
        ReH_free=ReH_free.astype(np.float32),
        ReH_truth=ReH_truth.astype(np.float32),
        deltaReH_truth=deltaReH_truth.astype(np.float32),
    )

    # save CSV
    csv_path = data_dir / f"dataset_{TAG}.csv"
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
    df.to_csv(csv_path, index=False)

    # save truth JSON
    truth_path = data_dir / f"truth_{TAG}.json"
    truth_obj = {
        "tag": TAG,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "fixed_kinematics": {"Ebeam": float(BEAM_ENERGY), "Q2": float(Q2), "t": float(T)},
        "xB_nodes": [float(v) for v in xB_nodes],
        "xi_nodes": [float(v) for v in xi_nodes],
        "bkm10_settings": {
            "using_ww": bool(USING_WW),
            "target_polarization": float(TARGET_POLARIZATION),
            "lepton_beam_polarization": float(LEPTON_BEAM_POLARIZATION),
        },
        "nuisance_fixed": {
            "E": [float(CFF_E_FIXED.real), float(CFF_E_FIXED.imag)],
            "Ht": [float(CFF_HT_FIXED.real), float(CFF_HT_FIXED.imag)],
            "Et": [float(CFF_ET_FIXED.real), float(CFF_ET_FIXED.imag)],
        },
        "pv_dr": {
            "kernel": "trapezoid PV on xi grid, K[i,i]=0",
            "C0_mode": str(C0_MODE),
            "C0_truth": float(C0_truth),
        },
        "gated_truth": {
            "g_truth": float(g_truth),
            "formula": "ReH_truth = g*ReH_DR + (1-g)*ReH_free",
            "ReH_DR": [float(v) for v in ReH_DR],
            "ReH_free": [float(v) for v in ReH_free],
            "ReH_truth": [float(v) for v in ReH_truth],
            "deltaReH_truth": [float(v) for v in deltaReH_truth],
        },
        "H_truth": {
            "ImH_truth": [float(v) for v in ImH_truth],
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
    print(f"  NPZ:   {npz_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Truth: {truth_path}")
    print("\nSelected xB nodes:")
    for i in range(B):
        print(f"  i={i:02d}  xB={xB_nodes[i]:.6f}  xi={xi_nodes[i]:.6f}")
    print(f"\nTruth gate: g_truth = {g_truth:.3f}")
    print(f"C0_truth = {C0_truth:+.6g}")


if __name__ == "__main__":
    main()
