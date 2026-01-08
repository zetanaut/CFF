#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py  (Python 3.9+)

Generate a multi-xB / multi-xi closure dataset suitable for FiLM-gated non-bias DR training.

Dataset structure:
  - B bins in xB (equivalently xi), each with Nphi points in phi.
  - Fixed (t, Q2, beam energy) across bins by default.

Truth construction:
  1) Choose xi grid -> convert to xB grid
  2) Define a smooth ImH_truth(xi)
  3) Define C0_truth
  4) Compute ReH_DR(xi) = C0_truth + PV[ImH_truth](xi) using a discrete K matrix
  5) Optionally add a DR violation:
        ReH_truth(xi) = ReH_DR(xi) + Delta_viol(xi)

Observables:
  - XS(phi), BSA(phi) computed by bkm10_lib at each (xB_i, phi_j)

Outputs:
  <OUT_DIR>/data/dataset_<TAG>.npz
  <OUT_DIR>/data/dataset_<TAG>.csv
  <OUT_DIR>/data/truth_<TAG>.json

NPZ keys include (at minimum):
  x, y_central, y_sigma
and additionally:
  y_true, phi_rad, phi_deg
  xi_bins, xB_bins, t_bins, Q2_bins
  K
  C0_truth, ImH_truth, ReH_DR_truth, ReH_truth, Delta_viol_truth
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs


# =========================
# USER CONFIG
# =========================

OUT_DIR = "output"
TAG = "v_1"

# Fixed kinematics (shared for all bins)
BEAM_ENERGY = 5.75
Q2 = 1.82
T = -0.17

# Choose xi bins uniformly, then convert to xB:
#   xB = 2 xi / (1 + xi)
B = 7
XI_MIN = 0.122
XI_MAX = 0.218

# Phi grid
PHI_START_DEG = 0.0
PHI_STOP_DEG = 360.0
N_PHI = 30
INCLUDE_ENDPOINT = False  # recommended False (avoid duplicated 0/360)

# bkm10_lib settings
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# Nuisance CFFs (fixed, must match training)
CFF_E  = complex(2.217354372014208, 0.0)
CFF_HT = complex(1.409393726454478, 1.57736440256014)
CFF_ET = complex(144.4101642020152, 0.0)

# Uncertainty model:
#   sigma = ABS + REL * |y_true|
XS_ABS_ERR = 0.0
XS_REL_ERR = 0.0
BSA_ABS_ERR = 0.0
BSA_REL_ERR = 0.0
XS_MIN_ERR = 0.0
BSA_MIN_ERR = 0.0

# Central values:
ADD_NOISE_TO_CENTRAL = True
CENTRAL_NOISE_SEED = 42

# Truth mode:
#   "dr_consistent" -> ReH_truth = ReH_DR
#   "dr_violating"  -> ReH_truth = ReH_DR + Delta_viol(xi)
TRUTH_MODE = "dr_violating"

# Truth knobs (keep within reasonable scales for stability)
C0_TRUTH = -2.6

# Smooth polynomial-like ImH_truth(xi) (safe and smooth on a narrow xi window)
IMH_A0 = 2.2
IMH_A1 = 6.0
IMH_A2 = -12.0

# DR violation shape + amplitude (only used for TRUTH_MODE="dr_violating")
DELTA_VIOL_AMP = 1.0
DELTA_VIOL_POWER = 1.0  # shape sharpening (>=1)

# =========================
# END USER CONFIG
# =========================

PI = float(np.pi)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def xi_to_xB(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    return 2.0 * xi / (1.0 + xi)


def phi_grid_deg() -> np.ndarray:
    if INCLUDE_ENDPOINT:
        return np.linspace(PHI_START_DEG, PHI_STOP_DEG, N_PHI, dtype=float)
    return np.linspace(PHI_START_DEG, PHI_STOP_DEG, N_PHI, endpoint=False, dtype=float)


def build_pv_kernel_trapezoid(xi_nodes: np.ndarray) -> np.ndarray:
    """
    Discrete PV operator matrix K such that:
      ReH_DR = C0 + K @ ImH
    using trapezoid-like weights and kernel:
      (1/pi) * [ 1/(xi-x) - 1/(xi+x) ].

    Diagonal is set to 0 to implement PV on-grid.
    """
    xi = np.asarray(xi_nodes, dtype=float).reshape(-1)
    B = xi.size

    # trapezoid-ish weights
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


def imh_truth_func(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    xm = 0.5 * (XI_MIN + XI_MAX)
    dx = xi - xm
    return IMH_A0 + IMH_A1 * dx + IMH_A2 * dx * dx


def delta_viol_func(xi: np.ndarray) -> np.ndarray:
    """
    Smooth violation that is 0 at the endpoints and peaked in the middle:
      s = (xi - xi_min)/(xi_max-xi_min) in [0,1]
      Delta = amp * [s(1-s)]^power normalized to max=amp
    """
    xi = np.asarray(xi, dtype=float)
    s = (xi - float(XI_MIN)) / (float(XI_MAX) - float(XI_MIN))
    s = np.clip(s, 0.0, 1.0)
    base = (s * (1.0 - s)) ** float(DELTA_VIOL_POWER)
    if np.max(base) > 0:
        base = base / np.max(base)
    return float(DELTA_VIOL_AMP) * base


def make_xsecs_object(xB: float, cff_h: complex) -> DifferentialCrossSection:
    cfg = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k=float(BEAM_ENERGY),
            squared_Q_momentum_transfer=float(Q2),
            x_Bjorken=float(xB),
            squared_hadronic_momentum_transfer_t=float(T),
        ),
        "cff_inputs": CFFInputs(
            compton_form_factor_h=cff_h,
            compton_form_factor_h_tilde=CFF_HT,
            compton_form_factor_e=CFF_E,
            compton_form_factor_e_tilde=CFF_ET,
        ),
        "target_polarization": float(TARGET_POLARIZATION),
        "lepton_beam_polarization": float(LEPTON_BEAM_POLARIZATION),
        "using_ww": bool(USING_WW),
    }
    return DifferentialCrossSection(configuration=cfg, verbose=False, debugging=False)


def make_sigmas(xs_true: np.ndarray, bsa_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs_sigma = float(XS_ABS_ERR) + float(XS_REL_ERR) * np.abs(xs_true)
    bsa_sigma = float(BSA_ABS_ERR) + float(BSA_REL_ERR) * np.abs(bsa_true)
    xs_sigma = np.maximum(xs_sigma, float(XS_MIN_ERR))
    bsa_sigma = np.maximum(bsa_sigma, float(BSA_MIN_ERR))
    return xs_sigma.astype(float), bsa_sigma.astype(float)


def main() -> None:
    out_dir = Path(OUT_DIR)
    data_dir = out_dir / "data"
    _safe_mkdir(data_dir)

    # xi/xB bins
    xi_bins = np.linspace(float(XI_MIN), float(XI_MAX), int(B), dtype=float)
    xB_bins = xi_to_xB(xi_bins)

    # PV kernel
    K = build_pv_kernel_trapezoid(xi_bins)

    # Truth CFFs (binwise)
    ImH_truth = imh_truth_func(xi_bins)
    ReH_DR_truth = float(C0_TRUTH) + (K @ ImH_truth.astype(np.float32)).astype(float)

    if TRUTH_MODE.lower() == "dr_consistent":
        Delta_viol = np.zeros_like(xi_bins, dtype=float)
        ReH_truth = ReH_DR_truth.copy()
    elif TRUTH_MODE.lower() == "dr_violating":
        Delta_viol = delta_viol_func(xi_bins)
        ReH_truth = ReH_DR_truth + Delta_viol
    else:
        raise ValueError("TRUTH_MODE must be 'dr_consistent' or 'dr_violating'")

    # phi grid
    phi_deg = phi_grid_deg()
    phi_rad = np.radians(phi_deg).astype(float)

    # Build full dataset (N = B * Nphi)
    rows = []
    xs_true_all = []
    bsa_true_all = []

    for i in range(int(B)):
        xB = float(xB_bins[i])
        reh = float(ReH_truth[i])
        imh = float(ImH_truth[i])
        cff_h = complex(reh, imh)

        xsecs = make_xsecs_object(xB=xB, cff_h=cff_h)
        xs = np.asarray(xsecs.compute_cross_section(phi_rad).real, dtype=float)
        bsa = np.asarray(xsecs.compute_bsa(phi_rad).real, dtype=float)

        xs_true_all.append(xs)
        bsa_true_all.append(bsa)

        for j in range(len(phi_rad)):
            rows.append([float(T), xB, float(Q2), float(phi_rad[j])])

    xs_true_all = np.asarray(xs_true_all, dtype=float)   # (B,Nphi)
    bsa_true_all = np.asarray(bsa_true_all, dtype=float) # (B,Nphi)

    # pointwise sigmas on the flattened dataset
    xs_sigma_all, bsa_sigma_all = make_sigmas(xs_true_all, bsa_true_all)

    if ADD_NOISE_TO_CENTRAL:
        rng = np.random.default_rng(int(CENTRAL_NOISE_SEED))
        xs_central_all = xs_true_all + rng.normal(0.0, xs_sigma_all)
        bsa_central_all = bsa_true_all + rng.normal(0.0, bsa_sigma_all)
    else:
        xs_central_all = xs_true_all.copy()
        bsa_central_all = bsa_true_all.copy()

    # Flatten bin-major to match x rows
    xs_true_flat = xs_true_all.reshape(-1)
    bsa_true_flat = bsa_true_all.reshape(-1)
    xs_sig_flat = xs_sigma_all.reshape(-1)
    bsa_sig_flat = bsa_sigma_all.reshape(-1)
    xs_cent_flat = xs_central_all.reshape(-1)
    bsa_cent_flat = bsa_central_all.reshape(-1)

    X = np.asarray(rows, dtype=np.float32)
    y_true = np.column_stack([xs_true_flat, bsa_true_flat]).astype(np.float32)
    y_central = np.column_stack([xs_cent_flat, bsa_cent_flat]).astype(np.float32)
    y_sigma = np.column_stack([xs_sig_flat, bsa_sig_flat]).astype(np.float32)

    # Save NPZ
    npz_path = data_dir / f"dataset_{TAG}.npz"
    np.savez_compressed(
        npz_path,
        x=X,
        y_central=y_central,
        y_true=y_true,
        y_sigma=y_sigma,
        phi_deg=phi_deg.astype(np.float32),
        phi_rad=phi_rad.astype(np.float32),
        xi_bins=xi_bins.astype(np.float32),
        xB_bins=xB_bins.astype(np.float32),
        t_bins=np.full(int(B), float(T), dtype=np.float32),
        Q2_bins=np.full(int(B), float(Q2), dtype=np.float32),
        K=K.astype(np.float32),
        C0_truth=np.float32(C0_TRUTH),
        ImH_truth=ImH_truth.astype(np.float32),
        ReH_DR_truth=ReH_DR_truth.astype(np.float32),
        ReH_truth=ReH_truth.astype(np.float32),
        Delta_viol_truth=Delta_viol.astype(np.float32),
        truth_mode=np.array(TRUTH_MODE, dtype=object),
    )

    # CSV (human readable)
    df = pd.DataFrame(
        {
            "t": X[:, 0],
            "xB": X[:, 1],
            "Q2": X[:, 2],
            "phi": X[:, 3],
            "phi_deg": np.repeat(phi_deg, int(B)).reshape(len(phi_deg), int(B)).T.reshape(-1),
            "XS_true": xs_true_flat,
            "BSA_true": bsa_true_flat,
            "XS": xs_cent_flat,
            "BSA": bsa_cent_flat,
            "XS_err": xs_sig_flat,
            "BSA_err": bsa_sig_flat,
        }
    )
    csv_path = data_dir / f"dataset_{TAG}.csv"
    df.to_csv(csv_path, index=False)

    # Truth JSON
    truth_path = data_dir / f"truth_{TAG}.json"
    truth_obj = dict(
        tag=TAG,
        created_at=datetime.utcnow().isoformat() + "Z",
        truth_mode=TRUTH_MODE,
        kinematics=dict(beam_energy=float(BEAM_ENERGY), Q2=float(Q2), t=float(T)),
        bins=dict(B=int(B), xi_min=float(XI_MIN), xi_max=float(XI_MAX)),
        xi_bins=[float(x) for x in xi_bins],
        xB_bins=[float(x) for x in xB_bins],
        C0_truth=float(C0_TRUTH),
        ImH_truth=[float(x) for x in ImH_truth],
        ReH_DR_truth=[float(x) for x in ReH_DR_truth],
        ReH_truth=[float(x) for x in ReH_truth],
        Delta_viol_truth=[float(x) for x in Delta_viol],
        nuisance_CFFs=dict(E=[CFF_E.real, CFF_E.imag], Ht=[CFF_HT.real, CFF_HT.imag], Et=[CFF_ET.real, CFF_ET.imag]),
        phi_grid=dict(start_deg=float(PHI_START_DEG), stop_deg=float(PHI_STOP_DEG), Nphi=int(N_PHI), include_endpoint=bool(INCLUDE_ENDPOINT)),
        uncertainty_model=dict(
            XS_abs=float(XS_ABS_ERR), XS_rel=float(XS_REL_ERR),
            BSA_abs=float(BSA_ABS_ERR), BSA_rel=float(BSA_REL_ERR),
            add_noise_to_central=bool(ADD_NOISE_TO_CENTRAL),
        ),
    )
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth_obj, f, indent=2, sort_keys=True)

    # Print summary
    print("Wrote:")
    print(f"  NPZ:   {npz_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Truth: {truth_path}\n")

    print("Selected xB nodes:")
    for i, (xB, xi) in enumerate(zip(xB_bins, xi_bins)):
        print(f"  i={i:02d}  xB={xB:.6f}  xi={xi:.6f}")

    print(f"\nTruth mode: {TRUTH_MODE}")
    print(f"C0_truth = {float(C0_TRUTH):+.6g}")
    print(f"ImH_truth range = [{ImH_truth.min():.6g}, {ImH_truth.max():.6g}]")
    print(f"ReH_DR_truth range = [{ReH_DR_truth.min():.6g}, {ReH_DR_truth.max():.6g}]")
    print(f"ReH_truth range = [{ReH_truth.min():.6g}, {ReH_truth.max():.6g}]")
    if TRUTH_MODE.lower() == "dr_violating":
        print(f"Delta_viol range = [{Delta_viol.min():.6g}, {Delta_viol.max():.6g}]")

    print("\nUncertainties (median):")
    print("  XS sigma median  =", float(np.median(xs_sig_flat)))
    print("  BSA sigma median =", float(np.median(bsa_sig_flat)))


if __name__ == "__main__":
    main()
