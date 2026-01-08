#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""closure_generate_dataset.py

Generate a closure-test dataset at fixed kinematics.

Truth model:
  - Gepard (KM15) provides CFFs (H, E, H~, E~) at the chosen kinematics.

Observables:
  - bkm10_lib computes:
      * unpolarized cross section XS(phi)
      * beam spin asymmetry BSA(phi)

Uncertainties / pseudo-data:
  - Choose any number of phi points, equally spaced in phi.
  - Choose Gaussian uncertainty sizes (absolute and/or relative).
  - The script produces:
      * y_true(phi):     (XS_true, BSA_true)
      * y_sigma(phi):    (XS_err,  BSA_err)  1sig uncertainties (can be zero)
      * y_central(phi):  either y_true (default) or one noisy pseudo-measurement sample

Replicas are *not* generated here. The training script will generate replica datasets
by sampling within the provided per-point Gaussian uncertainties.

Outputs (default under <OUT_DIR>/data):
  - dataset_<TAG>.csv
  - dataset_<TAG>.npz
  - truth_<TAG>.json   (truth for evaluation only; training script does NOT need this)

Python: 3.9+

Edit the "USER CONFIG" block below, then run:
  python closure_generate_dataset.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

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
# The following names much be changed for what input names you want for training and evaluation
OUT_DIR = "output"
TAG = "v_1"

# Fixed kinematics
BEAM_ENERGY = 5.75
Q2 = 1.82
XB = 0.34
T = -0.17

# Phi grid (equally spaced)
PHI_START_DEG = 0.0
PHI_STOP_DEG = 360.0
N_PHI = 30
INCLUDE_ENDPOINT = True  # True includes both 0 and 360. If False, endpoint=False.

# bkm10_lib settings
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# Uncertainty model:
#   sigma = ABS + REL * |y_true|
# You can set ABS=REL=0 to represent "perfect" data. (y_sigma will be 0.)
# For cross section realistic error is about XS_ABS_ERR = 0.005 and 
# For BSA realistic error is about BSA_ABS_ERR = 0.02
XS_ABS_ERR = 0.00
XS_REL_ERR = 0.0

BSA_ABS_ERR = 0.0
BSA_REL_ERR = 0.0

# Optional floors (useful if you *never* want exactly zero sigmas in the NPZ)
XS_MIN_ERR = 0.0
BSA_MIN_ERR = 0.0

# If True: produce one noisy pseudo-measurement y_central = y_true + N(0,sigma)
# If False: y_central = y_true (recommended for replica studies)
ADD_NOISE_TO_CENTRAL = True
CENTRAL_NOISE_SEED = 42

# =========================
# END USER CONFIG
# =========================


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).item())


def _phi_grid_deg() -> np.ndarray:
    if INCLUDE_ENDPOINT:
        return np.linspace(PHI_START_DEG, PHI_STOP_DEG, N_PHI, dtype=float)
    return np.linspace(PHI_START_DEG, PHI_STOP_DEG, N_PHI, endpoint=False, dtype=float)


def km15_truth_cffs(phi_radians: np.ndarray) -> Dict[str, float]:
    """Compute KM15 CFF components at fixed kinematics.

    We scan phi and take the first value (CFFs should be phi-independent).
    Std over phi is stored for sanity checks.
    """
    dps = [
        g.DataPoint(
            xB=XB,
            t=T,
            Q2=Q2,
            phi=float(phi),
            process="ep2epgamma",
            exptype="fixed target",
            in1energy=BEAM_ENERGY,
            in1charge=-1,
            in1polarization=+1,
            observable="XS",
            fname="Trento",
        )
        for phi in phi_radians
    ]

    reh = np.asarray([th_KM15.ReH(dp) for dp in dps], dtype=float)
    imh = np.asarray([th_KM15.ImH(dp) for dp in dps], dtype=float)

    ree = np.asarray([th_KM15.ReE(dp) for dp in dps], dtype=float)
    ime = np.asarray([th_KM15.ImE(dp) for dp in dps], dtype=float)

    reht = np.asarray([th_KM15.ReHt(dp) for dp in dps], dtype=float)
    imht = np.asarray([th_KM15.ImHt(dp) for dp in dps], dtype=float)

    reet = np.asarray([th_KM15.ReEt(dp) for dp in dps], dtype=float)
    imet = np.asarray([th_KM15.ImEt(dp) for dp in dps], dtype=float)

    return {
        "cff_real_h_km15": _as_float(reh[0]),
        "cff_imag_h_km15": _as_float(imh[0]),
        "cff_real_e_km15": _as_float(ree[0]),
        "cff_imag_e_km15": _as_float(ime[0]),
        "cff_real_ht_km15": _as_float(reht[0]),
        "cff_imag_ht_km15": _as_float(imht[0]),
        "cff_real_et_km15": _as_float(reet[0]),
        "cff_imag_et_km15": _as_float(imet[0]),
        "reh_phi_std": float(np.std(reh, ddof=0)),
        "imh_phi_std": float(np.std(imh, ddof=0)),
    }


def make_xsecs_object(cffs: Dict[str, float]) -> DifferentialCrossSection:
    cff_h = complex(cffs["cff_real_h_km15"], cffs["cff_imag_h_km15"])
    cff_e = complex(cffs["cff_real_e_km15"], cffs["cff_imag_e_km15"])
    cff_ht = complex(cffs["cff_real_ht_km15"], cffs["cff_imag_ht_km15"])
    cff_et = complex(cffs["cff_real_et_km15"], cffs["cff_imag_et_km15"])

    cfg = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k=float(BEAM_ENERGY),
            squared_Q_momentum_transfer=float(Q2),
            x_Bjorken=float(XB),
            squared_hadronic_momentum_transfer_t=float(T),
        ),
        "cff_inputs": CFFInputs(
            compton_form_factor_h=cff_h,
            compton_form_factor_h_tilde=cff_ht,
            compton_form_factor_e=cff_e,
            compton_form_factor_e_tilde=cff_et,
        ),
        "target_polarization": float(TARGET_POLARIZATION),
        "lepton_beam_polarization": float(LEPTON_BEAM_POLARIZATION),
        "using_ww": bool(USING_WW),
    }
    return DifferentialCrossSection(configuration=cfg, verbose=False, debugging=False)


def make_sigmas(xs_true: np.ndarray, bsa_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-point 1σ uncertainties for XS and BSA.

    Allows exact zeros if ABS=REL=0 and MIN_ERR=0.
    """
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
    phi_rad = np.radians(phi_deg).astype(float)

    cffs = km15_truth_cffs(phi_rad)
    xsecs = make_xsecs_object(cffs)

    xs_true = np.asarray(xsecs.compute_cross_section(phi_rad).real, dtype=float)
    bsa_true = np.asarray(xsecs.compute_bsa(phi_rad).real, dtype=float)
    xs_sigma, bsa_sigma = make_sigmas(xs_true, bsa_true)

    if ADD_NOISE_TO_CENTRAL:
        rng = np.random.default_rng(int(CENTRAL_NOISE_SEED))
        xs_central = xs_true + rng.normal(0.0, xs_sigma)
        bsa_central = bsa_true + rng.normal(0.0, bsa_sigma)
    else:
        xs_central = xs_true.copy()
        bsa_central = bsa_true.copy()

    # Features X
    X = np.column_stack(
        [
            np.full_like(phi_rad, float(T), dtype=float),
            np.full_like(phi_rad, float(XB), dtype=float),
            np.full_like(phi_rad, float(Q2), dtype=float),
            phi_rad.astype(float),
        ]
    ).astype(np.float32)

    y_true = np.column_stack([xs_true, bsa_true]).astype(np.float32)
    y_central = np.column_stack([xs_central, bsa_central]).astype(np.float32)
    y_sigma = np.column_stack([xs_sigma, bsa_sigma]).astype(np.float32)

    # Save NPZ for training
    npz_path = data_dir / f"dataset_{TAG}.npz"
    np.savez_compressed(
        npz_path,
        x=X,
        y_central=y_central,
        y_true=y_true,
        y_sigma=y_sigma,
        phi_deg=phi_deg.astype(np.float32),
        phi_rad=phi_rad.astype(np.float32),
    )

    # Save CSV (human readable)
    df = pd.DataFrame(
        {
            "t": X[:, 0],
            "x_b": X[:, 1],
            "q_squared": X[:, 2],
            "phi": X[:, 3],  # radians
            "phi_deg": phi_deg,
            "XS_true": xs_true,
            "BSA_true": bsa_true,
            "XS": xs_central,
            "BSA": bsa_central,
            "XS_err": xs_sigma,
            "BSA_err": bsa_sigma,
        }
    )
    csv_path = data_dir / f"dataset_{TAG}.csv"
    df.to_csv(csv_path, index=False)

    # Save truth CFFs for evaluation (training does NOT need this)
    truth_path = data_dir / f"truth_{TAG}.json"
    truth_obj = {
        "tag": TAG,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "kinematics": {
            "beam_energy": float(BEAM_ENERGY),
            "q_squared": float(Q2),
            "x_b": float(XB),
            "t": float(T),
        },
        "bkm10_settings": {
            "using_ww": bool(USING_WW),
            "target_polarization": float(TARGET_POLARIZATION),
            "lepton_beam_polarization": float(LEPTON_BEAM_POLARIZATION),
        },
        "phi_grid": {
            "phi_start_deg": float(PHI_START_DEG),
            "phi_stop_deg": float(PHI_STOP_DEG),
            "n_phi": int(N_PHI),
            "include_endpoint": bool(INCLUDE_ENDPOINT),
        },
        "uncertainty_model": {
            "XS_abs_err": float(XS_ABS_ERR),
            "XS_rel_err": float(XS_REL_ERR),
            "BSA_abs_err": float(BSA_ABS_ERR),
            "BSA_rel_err": float(BSA_REL_ERR),
            "XS_min_err": float(XS_MIN_ERR),
            "BSA_min_err": float(BSA_MIN_ERR),
            "add_noise_to_central": bool(ADD_NOISE_TO_CENTRAL),
            "central_noise_seed": int(CENTRAL_NOISE_SEED),
        },
        "km15_truth_cffs": cffs,
    }
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth_obj, f, indent=2, sort_keys=True)

    print("Wrote:")
    print(f"  CSV:   {csv_path}")
    print(f"  NPZ:   {npz_path}")
    print(f"  Truth (eval only): {truth_path}")
    print("\nTruth H (KM15): ReH = {:.6g}, ImH = {:.6g}".format(
        float(cffs["cff_real_h_km15"]), float(cffs["cff_imag_h_km15"])
    ))
    print("KM15 phi-constancy check: std(ReH(phi)) = {:.3e}, std(ImH(phi)) = {:.3e}".format(
        float(cffs["reh_phi_std"]), float(cffs["imh_phi_std"])
    ))
    print("\nUncertainties (median):")
    print("  XS sigma median  = {:.6g}".format(float(np.median(xs_sigma))))
    print("  BSA sigma median = {:.6g}".format(float(np.median(bsa_sigma))))
    if float(np.max(xs_sigma)) == 0.0 and float(np.max(bsa_sigma)) == 0.0:
        print("\nNOTE: You set all uncertainties to zero. Replicas will be identical; training will fall back to MSE-like weighting.")

    # Print full truth CFF table + copy/paste helpers (for hardcoding nuisance CFFs)
    print("\nKM15 truth CFFs at this kinematic point:")
    keys = [
        ("H",  "cff_real_h_km15",  "cff_imag_h_km15"),
        ("E",  "cff_real_e_km15",  "cff_imag_e_km15"),
        ("Ht", "cff_real_ht_km15", "cff_imag_ht_km15"),
        ("Et", "cff_real_et_km15", "cff_imag_et_km15"),
    ]
    for label, kr, ki in keys:
        vr = float(cffs[kr])
        vi = float(cffs[ki])
        print(f"  {label:>2s}: Re = {vr:.16g}   Im = {vi:.16g}")

    print("\nCopy/paste (training script nuisance CFFs):")
    print("  # paste into closure_train_replicas_v2.py (USER CONFIG)")
    print("  CFF_E  = complex({:.16g}, {:.16g})".format(float(cffs["cff_real_e_km15"]),  float(cffs["cff_imag_e_km15"])))
    print("  CFF_HT = complex({:.16g}, {:.16g})".format(float(cffs["cff_real_ht_km15"]), float(cffs["cff_imag_ht_km15"])))
    print("  CFF_ET = complex({:.16g}, {:.16g})".format(float(cffs["cff_real_et_km15"]), float(cffs["cff_imag_et_km15"])))

    print("\nCopy/paste (evaluator hardcoded nuisance CFFs):")
    print('  # set NUISANCE_SOURCE = "hardcoded" in closure_evaluate.py then paste these')
    print("  CFF_E_HARDCODED  = complex({:.16g}, {:.16g})".format(float(cffs["cff_real_e_km15"]),  float(cffs["cff_imag_e_km15"])))
    print("  CFF_HT_HARDCODED = complex({:.16g}, {:.16g})".format(float(cffs["cff_real_ht_km15"]), float(cffs["cff_imag_ht_km15"])))
    print("  CFF_ET_HARDCODED = complex({:.16g}, {:.16g})".format(float(cffs["cff_real_et_km15"]), float(cffs["cff_imag_et_km15"])))


if __name__ == "__main__":
    main()
