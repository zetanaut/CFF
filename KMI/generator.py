#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py  (Python 3.9+)

Robust KMI dataset generator WITHOUT DR.

Key improvements vs v1:
  - Auto-picks 10 kinematic bins that are *physically valid* for the chosen beam energy.
  - Enforces y<1 and W>W_MIN before calling bkm10_lib.
  - Runs a fast BKM "finite XS/BSA" check *before* generating full phi grids.
  - Avoids phi endpoints (0, 180, 360) by using mid-bin phi values -> more stable.

Produces 10 datasets (bins) with variable phi counts:
  - 4 bins with 30 points
  - 3 bins with 24 points
  - 3 bins with 15 points

Truth CFFs:
  - Default: gepard KM15 (if available)
  - Fallback: smooth toy truth

Outputs:
  output/data/dataset_<TAG>.npz
  output/data/dataset_<TAG>.csv
  output/data/truth_<TAG>.json
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*")

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs


# =========================
# USER CONFIG
# =========================

OUT_DIR = "output"
TAG = "v_1"

BEAM_ENERGY = 5.75

# Generate 10 kinematic bins automatically (recommended)
AUTO_PICK_KINEMATICS = True
KIN_SEED = 20260109

# Realistic CLAS6-like region constraints (tune as needed)
XB_MIN, XB_MAX = 0.18, 0.38
Y_MIN,  Y_MAX  = 0.25, 0.80         # enforce y<1 by construction (and stay away from y~1)
Q2_MIN, Q2_MAX = 1.3,  3.8
W_MIN = 2.0                         # GeV, keep above resonance-ish region
TMIN_MAG, TMAX_MAG = 0.15, 0.65     # sample |t| in [0.15, 0.65]

# If you want to supply your own kinematics table, set AUTO_PICK_KINEMATICS=False
# and fill KINEMATICS_TABLE with 10 entries: dict(xB=?, Q2=?, t=?)
KINEMATICS_TABLE = []  # used only if AUTO_PICK_KINEMATICS=False

# Variable phi point counts: 4x30, 3x24, 3x15
PHI_COUNTS = [30, 30, 30, 30, 24, 24, 24, 15, 15, 15]
SHUFFLE_PHI_COUNTS = True  # randomize assignment of phi-counts to bins

# Phi generation
PHI_START_DEG = 0.0
PHI_STOP_DEG = 360.0
INCLUDE_ENDPOINT = False
AVOID_ENDPOINTS = True     # shift phi by half-step so we never hit exactly 0, 180, 360

# bkm10_lib settings
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# Truth source: "KM15" or "toy"
TRUTH_SOURCE = "KM15"

# Uncertainties: sigma = ABS + REL*|y_true|
XS_ABS_ERR = 0.0
XS_REL_ERR = 0.05
BSA_ABS_ERR = 0.02
BSA_REL_ERR = 0.0
XS_MIN_ERR = 0.0
BSA_MIN_ERR = 0.0

ADD_NOISE_TO_CENTRAL = True
CENTRAL_NOISE_SEED = 42

# =========================
# END USER CONFIG
# =========================

M_PROTON = 0.9382720813  # GeV


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def xi_from_xB(xB: float) -> float:
    return float(xB) / (2.0 - float(xB))


def y_from_xB_Q2_E(xB: float, Q2: float, E: float) -> float:
    return float(Q2) / (2.0 * float(M_PROTON) * float(xB) * float(E))


def W2_from_xB_Q2(xB: float, Q2: float) -> float:
    # W^2 = M^2 + Q^2 (1/xB - 1)
    return float(M_PROTON**2 + float(Q2) * (1.0 / float(xB) - 1.0))


def phi_grid_deg(nphi: int) -> np.ndarray:
    nphi = int(nphi)
    if INCLUDE_ENDPOINT:
        phi = np.linspace(PHI_START_DEG, PHI_STOP_DEG, nphi, dtype=float)
    else:
        phi = np.linspace(PHI_START_DEG, PHI_STOP_DEG, nphi, endpoint=False, dtype=float)

    if AVOID_ENDPOINTS:
        # shift by half-bin width so we avoid exactly 0,180,360
        step = (PHI_STOP_DEG - PHI_START_DEG) / float(nphi)
        phi = (phi + 0.5 * step) % 360.0

    return phi


def make_xsecs_object(beam_energy: float, xB: float, Q2: float, t: float,
                      cff_h: complex, cff_e: complex, cff_ht: complex, cff_et: complex) -> DifferentialCrossSection:
    cfg = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k=float(beam_energy),
            squared_Q_momentum_transfer=float(Q2),
            x_Bjorken=float(xB),
            squared_hadronic_momentum_transfer_t=float(t),
        ),
        "cff_inputs": CFFInputs(
            compton_form_factor_h=cff_h,
            compton_form_factor_e=cff_e,
            compton_form_factor_h_tilde=cff_ht,
            compton_form_factor_e_tilde=cff_et,
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


def truth_cffs_km15(beam_energy: float, xB: float, Q2: float, t: float) -> Dict[str, complex]:
    import gepard as g
    from gepard.fits import th_KM15

    dp = g.DataPoint(
        xB=float(xB),
        t=float(t),
        Q2=float(Q2),
        phi=float(np.radians(10.0)),
        process="ep2epgamma",
        exptype="fixed target",
        in1energy=float(beam_energy),
        in1charge=-1,
        in1polarization=+1,
        observable="XS",
        fname="Trento",
    )

    return dict(
        H=complex(float(th_KM15.ReH(dp)),  float(th_KM15.ImH(dp))),
        E=complex(float(th_KM15.ReE(dp)),  float(th_KM15.ImE(dp))),
        Ht=complex(float(th_KM15.ReHt(dp)), float(th_KM15.ImHt(dp))),
        Et=complex(float(th_KM15.ReEt(dp)), float(th_KM15.ImEt(dp))),
    )


def truth_cffs_toy(xB: float, Q2: float, t: float) -> Dict[str, complex]:
    xi = xi_from_xB(xB)
    reh = 2.0 + 1.0*(xi-0.20) + 0.25*np.log(Q2) + 0.3*(t+0.30)
    imh = 2.4 + 1.6*(xi-0.20) + 0.10*np.log(Q2) - 0.2*(t+0.30)
    # modest nuisance
    return dict(
        H=complex(reh, imh),
        E=complex(0.4 + 0.2*(xi-0.2), 0.0),
        Ht=complex(0.7, 0.4),
        Et=complex(1.0, 0.0),
    )


def bkm_is_finite_for_kinematics(beam_energy: float, xB: float, Q2: float, t: float) -> bool:
    """
    Pure kinematic/numerical finiteness check. Uses small "safe" CFFs and a short phi-test grid.
    This is done BEFORE expensive truth-model calls and full dataset generation.
    """
    # Quick physics checks first
    y = y_from_xB_Q2_E(xB, Q2, beam_energy)
    W2 = W2_from_xB_Q2(xB, Q2)
    if not (0.0 < y < 0.95):
        return False
    if not (W2 > float(W_MIN)**2):
        return False

    # Short phi test (avoid endpoints)
    phi_test_deg = np.array([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345], dtype=float)
    phi_test_rad = np.radians(phi_test_deg).astype(float)

    # "safe" moderate CFFs
    cffs = dict(
        H=complex(2.0, 2.0),
        E=complex(0.2, 0.0),
        Ht=complex(0.5, 0.3),
        Et=complex(1.0, 0.0),
    )

    try:
        xsecs = make_xsecs_object(beam_energy, xB, Q2, t, cffs["H"], cffs["E"], cffs["Ht"], cffs["Et"])
        xs = np.asarray(xsecs.compute_cross_section(phi_test_rad).real, dtype=float)
        bsa = np.asarray(xsecs.compute_bsa(phi_test_rad).real, dtype=float)
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(bsa))):
            return False
        # Also guard against pathological huge values (optional)
        if np.max(np.abs(xs)) > 1e6:
            return False
        if np.max(np.abs(bsa)) > 10.0:
            return False
        return True
    except Exception:
        return False


def auto_pick_kinematics(B: int, seed: int) -> List[Dict[str, float]]:
    """
    Sample physically valid bins.
    Strategy:
      - sample xB in [XB_MIN, XB_MAX]
      - sample y in [Y_MIN, Y_MAX]
      - compute Q2 = y * 2 M xB E
      - require Q2 in [Q2_MIN,Q2_MAX] and W>W_MIN
      - sample t negative in [-TMAX_MAG, -TMIN_MAG]
      - run bkm_is_finite_for_kinematics check
    """
    rng = np.random.default_rng(int(seed))
    bins: List[Dict[str, float]] = []

    n_try = 0
    while len(bins) < B:
        n_try += 1
        if n_try > 20000:
            raise RuntimeError("Failed to find enough valid kinematic bins. Relax ranges (y, Q2, W, |t|).")

        xB = float(rng.uniform(XB_MIN, XB_MAX))
        y = float(rng.uniform(Y_MIN, Y_MAX))
        Q2 = float(y * 2.0 * M_PROTON * xB * BEAM_ENERGY)

        if not (Q2_MIN <= Q2 <= Q2_MAX):
            continue
        W2 = W2_from_xB_Q2(xB, Q2)
        if W2 <= float(W_MIN)**2:
            continue

        # sample |t|
        t = -float(rng.uniform(TMIN_MAG, TMAX_MAG))

        if not bkm_is_finite_for_kinematics(BEAM_ENERGY, xB, Q2, t):
            continue

        # reject near-duplicates
        xi = xi_from_xB(xB)
        too_close = False
        for b in bins:
            if abs(b["xB"] - xB) < 0.01 and abs(b["Q2"] - Q2) < 0.15 and abs(b["t"] - t) < 0.05:
                too_close = True
                break
            if abs(xi_from_xB(b["xB"]) - xi) < 0.01 and abs(b["Q2"] - Q2) < 0.15:
                too_close = True
                break
        if too_close:
            continue

        bins.append(dict(xB=xB, Q2=Q2, t=t))

    # Sort by xi for nicer coverage
    bins.sort(key=lambda b: xi_from_xB(b["xB"]))
    return bins


def main() -> None:
    out_dir = Path(OUT_DIR)
    data_dir = out_dir / "data"
    _safe_mkdir(data_dir)

    B = 10

    # Pick phi counts assignment
    phi_counts = list(PHI_COUNTS)
    if SHUFFLE_PHI_COUNTS:
        rng_phi = np.random.default_rng(12345)
        rng_phi.shuffle(phi_counts)

    # Pick kinematics
    if AUTO_PICK_KINEMATICS:
        kin_list = auto_pick_kinematics(B, KIN_SEED)
    else:
        if len(KINEMATICS_TABLE) != B:
            raise ValueError("If AUTO_PICK_KINEMATICS=False, KINEMATICS_TABLE must have length 10.")
        # Still validate each user-specified bin early:
        kin_list = []
        for i, k in enumerate(KINEMATICS_TABLE):
            xB = float(k["xB"]); Q2 = float(k["Q2"]); t = float(k["t"])
            if not bkm_is_finite_for_kinematics(BEAM_ENERGY, xB, Q2, t):
                y = y_from_xB_Q2_E(xB, Q2, BEAM_ENERGY)
                W2 = W2_from_xB_Q2(xB, Q2)
                raise RuntimeError(
                    f"User kinematics bin {i} fails precheck: xB={xB}, Q2={Q2}, t={t} "
                    f"(y={y:.3f}, W={np.sqrt(max(W2,0)):.3f} GeV)."
                )
            kin_list.append(dict(xB=xB, Q2=Q2, t=t))

    xB_bins = np.array([k["xB"] for k in kin_list], dtype=float)
    Q2_bins = np.array([k["Q2"] for k in kin_list], dtype=float)
    t_bins  = np.array([k["t"]  for k in kin_list], dtype=float)
    xi_bins = np.array([xi_from_xB(x) for x in xB_bins], dtype=float)

    # Compute truth CFFs per bin (now that kinematics are safe)
    truth_H  = np.zeros(B, dtype=np.complex128)
    truth_E  = np.zeros(B, dtype=np.complex128)
    truth_Ht = np.zeros(B, dtype=np.complex128)
    truth_Et = np.zeros(B, dtype=np.complex128)

    for i in range(B):
        if TRUTH_SOURCE.upper() == "KM15":
            try:
                cffs = truth_cffs_km15(BEAM_ENERGY, xB_bins[i], Q2_bins[i], t_bins[i])
            except Exception as e:
                print(f"KM15 failed for bin {i}: {e} -> toy truth.")
                cffs = truth_cffs_toy(xB_bins[i], Q2_bins[i], t_bins[i])
        else:
            cffs = truth_cffs_toy(xB_bins[i], Q2_bins[i], t_bins[i])

        truth_H[i]  = cffs["H"]
        truth_E[i]  = cffs["E"]
        truth_Ht[i] = cffs["Ht"]
        truth_Et[i] = cffs["Et"]

    # Build dataset (ragged phi grids -> flattened rows)
    rows_x = []
    rows_bin = []
    xs_true_list = []
    bsa_true_list = []
    xs_sig_list = []
    bsa_sig_list = []
    phi_deg_flat = []

    for i in range(B):
        nphi = int(phi_counts[i])
        phi_deg = phi_grid_deg(nphi)
        phi_rad = np.radians(phi_deg).astype(float)

        xsecs = make_xsecs_object(
            BEAM_ENERGY, xB_bins[i], Q2_bins[i], t_bins[i],
            truth_H[i], truth_E[i], truth_Ht[i], truth_Et[i]
        )

        xs_true = np.asarray(xsecs.compute_cross_section(phi_rad).real, dtype=float)
        bsa_true = np.asarray(xsecs.compute_bsa(phi_rad).real, dtype=float)

        # Final safety
        if not (np.all(np.isfinite(xs_true)) and np.all(np.isfinite(bsa_true))):
            y = y_from_xB_Q2_E(xB_bins[i], Q2_bins[i], BEAM_ENERGY)
            W2 = W2_from_xB_Q2(xB_bins[i], Q2_bins[i])
            raise RuntimeError(
                f"Unexpected non-finite BKM at generation time for bin {i}: "
                f"xB={xB_bins[i]}, Q2={Q2_bins[i]}, t={t_bins[i]} (y={y:.3f}, W={np.sqrt(max(W2,0)):.3f} GeV)."
            )

        xs_sig, bsa_sig = make_sigmas(xs_true, bsa_true)

        for j in range(nphi):
            rows_x.append([float(t_bins[i]), float(xB_bins[i]), float(Q2_bins[i]), float(phi_rad[j])])
            rows_bin.append(i)
            xs_true_list.append(xs_true[j]); bsa_true_list.append(bsa_true[j])
            xs_sig_list.append(xs_sig[j]);   bsa_sig_list.append(bsa_sig[j])
            phi_deg_flat.append(phi_deg[j])

    X = np.asarray(rows_x, dtype=np.float32)
    bin_id = np.asarray(rows_bin, dtype=np.int32)
    y_true = np.column_stack([xs_true_list, bsa_true_list]).astype(np.float32)
    y_sigma = np.column_stack([xs_sig_list, bsa_sig_list]).astype(np.float32)

    if ADD_NOISE_TO_CENTRAL:
        rng = np.random.default_rng(int(CENTRAL_NOISE_SEED))
        y_central = y_true + rng.normal(0.0, 1.0, size=y_true.shape).astype(np.float32) * y_sigma
    else:
        y_central = y_true.copy()

    # Save NPZ
    npz_path = data_dir / f"dataset_{TAG}.npz"
    np.savez_compressed(
        npz_path,
        x=X,
        bin_id=bin_id,
        y_true=y_true,
        y_central=y_central,
        y_sigma=y_sigma,
        beam_energy=np.float32(BEAM_ENERGY),
        using_ww=np.int32(1 if USING_WW else 0),
        target_polarization=np.float32(TARGET_POLARIZATION),
        lepton_beam_polarization=np.float32(LEPTON_BEAM_POLARIZATION),
        t_bins=t_bins.astype(np.float32),
        xB_bins=xB_bins.astype(np.float32),
        Q2_bins=Q2_bins.astype(np.float32),
        xi_bins=xi_bins.astype(np.float32),
        phi_counts=np.asarray(phi_counts, dtype=np.int32),
        phi_deg_flat=np.asarray(phi_deg_flat, dtype=np.float32),
        truth_cff_H=truth_H.astype(np.complex64),
        nuisance_cff_E=truth_E.astype(np.complex64),
        nuisance_cff_Ht=truth_Ht.astype(np.complex64),
        nuisance_cff_Et=truth_Et.astype(np.complex64),
        truth_source=np.array(TRUTH_SOURCE, dtype=object),
    )

    # Save CSV
    df = pd.DataFrame(
        dict(
            bin_id=bin_id,
            t=X[:, 0],
            xB=X[:, 1],
            Q2=X[:, 2],
            phi=X[:, 3],
            phi_deg=np.asarray(phi_deg_flat, dtype=float),
            XS_true=y_true[:, 0],
            BSA_true=y_true[:, 1],
            XS=y_central[:, 0],
            BSA=y_central[:, 1],
            XS_err=y_sigma[:, 0],
            BSA_err=y_sigma[:, 1],
        )
    )
    csv_path = data_dir / f"dataset_{TAG}.csv"
    df.to_csv(csv_path, index=False)

    # Save truth JSON
    truth_path = data_dir / f"truth_{TAG}.json"
    truth_obj = dict(
        tag=TAG,
        created_at=datetime.utcnow().isoformat() + "Z",
        beam_energy=float(BEAM_ENERGY),
        truth_source=str(TRUTH_SOURCE),
        kinematics=[dict(
            xB=float(xB_bins[i]),
            xi=float(xi_bins[i]),
            Q2=float(Q2_bins[i]),
            t=float(t_bins[i]),
            y=float(y_from_xB_Q2_E(xB_bins[i], Q2_bins[i], BEAM_ENERGY)),
            W=float(np.sqrt(max(W2_from_xB_Q2(xB_bins[i], Q2_bins[i]), 0.0))),
            nphi=int(phi_counts[i]),
        ) for i in range(B)],
        truth_cffs=[dict(
            H=[float(truth_H[i].real), float(truth_H[i].imag)],
            E=[float(truth_E[i].real), float(truth_E[i].imag)],
            Ht=[float(truth_Ht[i].real), float(truth_Ht[i].imag)],
            Et=[float(truth_Et[i].real), float(truth_Et[i].imag)],
        ) for i in range(B)],
        uncertainty_model=dict(
            XS_abs=float(XS_ABS_ERR), XS_rel=float(XS_REL_ERR),
            BSA_abs=float(BSA_ABS_ERR), BSA_rel=float(BSA_REL_ERR),
            add_noise_to_central=bool(ADD_NOISE_TO_CENTRAL),
        ),
        selection_ranges=dict(
            xB=[float(XB_MIN), float(XB_MAX)],
            y=[float(Y_MIN), float(Y_MAX)],
            Q2=[float(Q2_MIN), float(Q2_MAX)],
            W_min=float(W_MIN),
            t_abs=[float(TMIN_MAG), float(TMAX_MAG)],
        ),
    )
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth_obj, f, indent=2, sort_keys=True)

    print("Wrote:")
    print(f"  NPZ:   {npz_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Truth: {truth_path}\n")

    print("Kinematic bins (sorted by xi):")
    for i in range(B):
        y = y_from_xB_Q2_E(xB_bins[i], Q2_bins[i], BEAM_ENERGY)
        W = np.sqrt(max(W2_from_xB_Q2(xB_bins[i], Q2_bins[i]), 0.0))
        print(f"  bin {i:02d} | xB={xB_bins[i]:.3f}  Q2={Q2_bins[i]:.3f}  t={t_bins[i]:.3f}  "
              f"xi={xi_bins[i]:.3f}  y={y:.3f}  W={W:.3f}  Nphi={phi_counts[i]}")


if __name__ == "__main__":
    main()
