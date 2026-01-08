#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HardDR_train.py  (Python 3.9+)

Goal
----
Demonstrate a *Hard Dispersion Relation (Hard DR)* closure test for DVCS CFF extraction
using only:
  - unpolarized cross section   XS(phi)
  - beam spin asymmetry         BSA(phi)

at fixed (t, Q2, Ebeam) but for *multiple* xB (equivalently multiple xi).

Hard DR means:
--------------
We do NOT fit ReH as an independent parameter/function.
Instead we:
  1) parameterize ImH(xi) with a neural network,
  2) compute ReH(xi) from ImH(x) via a (discretized) fixed-t dispersion relation
     plus a subtraction constant C(t),
  3) feed the resulting complex H(xi)=ReH+i ImH into the BKM forward model,
  4) train only on data (XS,BSA), enforcing DR by construction.

Why multiple xB/xi points are required
-------------------------------------
A dispersion relation connects ReH at a given xi to an *integral over ImH(x)*.
Therefore a Hard-DR extraction is inherently *non-local in xi*. You need ImH(x)
over a range of x to determine ReH(xi). In this pedagogical closure script we
discretize the integral on a finite xi-grid and require the dataset to contain
multiple xi nodes. Lets of variations of this are possible.

Inputs (NPZ from your pipeline)
-------------------------------
<VERSION_DIR>/data/dataset_<TAG>.npz containing:
  x:         (N,4)  [t, xB, Q2, phi]  phi in radians
  y_central: (N,2)  [XS, BSA]
  y_sigma:   (N,2)  [XS_err, BSA_err] (used for replica noise)

IMPORTANT: This script expects the dataset to contain MULTIPLE xB values
(i.e., multiple unique (t,xB,Q2) bins) and multiple phi points per bin
with a common phi grid across bins.

For a true closure test, the dataset generator should also:
  - compute ReH(xi) from the SAME discretized DR used here,
  - or at least generate pseudo-data consistent with the DR model class.

Outputs
-------
<VERSION_DIR>/replicas_hard_dr/replica_XXX_<TAG>.keras   (optional; can save weights-only)
<VERSION_DIR>/histories_hard_dr/history_replica_XXX_<TAG>.json
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*")

# ---- Your physics forward model (external) ----
from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

# =========================
# USER CONFIG (edit here)
# =========================

VERSION_DIR = "output"
TAG = "v_1"
DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")

MODELS_DIR = os.path.join(VERSION_DIR, "replicas_hard_dr")
HIST_DIR = os.path.join(VERSION_DIR, "histories_hard_dr")

# If saving a subclassed model to .keras fails in your TF/Keras version,
# set this True to save weights + metadata instead.
SAVE_WEIGHTS_ONLY = True

# Replica settings
N_REPLICAS = 50
REPLICA_SEED = 20250101  # controls replica noise draws

# Train/val split (random over phi points, shared across all xi nodes).
# IMPORTANT: We keep ALL xi nodes in both train and validation because DR couples nodes.
TRAIN_FRACTION = 0.8
SPLIT_SEED = 1234

# Training hyperparameters
EPOCHS = 2500
PATIENCE = 250
LEARNING_RATE = 3e-3

# Finite-difference step for custom gradient (BKM w.r.t ReH,ImH)
FD_EPS = 5e-3

# Gradient clipping (recommended for stability)
ADAM_CLIPNORM = 5.0  # set 0 to disable

# Residual clipping in units of sigma_soft (post-division)
RATIO_CLIP = 1e4  # set None/0 to disable

# ---- Soft-chi2 normalization knobs ----
USE_POINTWISE_SIGMAS = True

SOFT_XS_REL = 0.02   # default ~2% XS relative error when sigma_data=0
SOFT_XS_ABS = 0.0    # absolute XS floor

SOFT_BSA_REL = 0.0
SOFT_BSA_ABS = 0.01  # absolute asymmetry floor

# Diagnostics
PRINT_POSTFIT_RESIDUALS = True

# Fixed beam energy (must match generator)
BEAM_ENERGY = 5.75

# bkm10_lib settings (must match generator)
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# Fixed nuisance CFFs (must match generator for closure)
CFF_E  = complex(2.217354372014208, 0)
CFF_HT = complex(1.409393726454478, 1.57736440256014)
CFF_ET = complex(144.4101642020152, 0)

# =========================
# END USER CONFIG
# =========================

_FLOATX = tf.float32
PI = float(np.pi)


# -------------------------
# Utility I/O
# -------------------------
def _safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    required = {"x", "y_central", "y_sigma"}
    if not required.issubset(set(d.files)):
        raise KeyError(f"NPZ missing required keys {required}; found {d.files}")
    return d


# -------------------------
# Kinematics helpers
# -------------------------
def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    """DVCS skewness approximation xi = xB/(2-xB)."""
    xB = np.asarray(xB, dtype=float)
    return xB / (2.0 - xB)


def group_by_kinematics(
    X: np.ndarray, y: np.ndarray, sigma_soft: np.ndarray
) -> Dict[Tuple[float, float, float], Dict[str, np.ndarray]]:
    """
    Group pointwise arrays into bins of unique (t, xB, Q2), each containing vectors over phi.

    Returns dict keyed by (t, xB, Q2) with:
      phi: (Nphi,)
      xs:  (Nphi,)
      bsa: (Nphi,)
      xs_sig: (Nphi,)
      bsa_sig:(Nphi,)
    """
    t_arr = X[:, 0]
    xB_arr = X[:, 1]
    Q2_arr = X[:, 2]
    phi_arr = X[:, 3]

    # Build keys (rounding helps avoid float uniqueness issues)
    key_arr = np.stack([t_arr, xB_arr, Q2_arr], axis=1)
    key_arr_rounded = np.round(key_arr, decimals=10)

    bins: Dict[Tuple[float, float, float], Dict[str, np.ndarray]] = {}
    for (t_r, xB_r, Q2_r) in np.unique(key_arr_rounded, axis=0):
        mask = np.all(key_arr_rounded == np.array([t_r, xB_r, Q2_r]), axis=1)
        phi = phi_arr[mask]
        order = np.argsort(phi)
        bins[(float(t_r), float(xB_r), float(Q2_r))] = {
            "phi": phi[order].astype(np.float32),
            "xs": y[mask, 0][order].astype(np.float32),
            "bsa": y[mask, 1][order].astype(np.float32),
            "xs_sig": sigma_soft[mask, 0][order].astype(np.float32),
            "bsa_sig": sigma_soft[mask, 1][order].astype(np.float32),
        }

    return bins


def assert_common_phi_grid(
    bins: Dict[Tuple[float, float, float], Dict[str, np.ndarray]], atol: float = 1e-6
) -> np.ndarray:
    """
    Ensure all bins share the same phi grid.
    Returns phi_grid (Nphi,). Raises if inconsistent.
    """
    keys = list(bins.keys())
    phi0 = bins[keys[0]]["phi"]
    for k in keys[1:]:
        phik = bins[k]["phi"]
        if len(phik) != len(phi0) or np.max(np.abs(phik - phi0)) > atol:
            raise ValueError(
                "Hard-DR script expects a common phi grid across xB bins.\n"
                f"Bin {keys[0]} has Nphi={len(phi0)}; bin {k} has Nphi={len(phik)}.\n"
                "If your dataset has varying phi coverage, you must pad/mask per bin."
            )
    return phi0


# -------------------------
# Soft sigma construction (your idea, generalized)
# -------------------------
def soft_sigmas(y_central: np.ndarray, y_sigma: np.ndarray) -> np.ndarray:
    """
    Construct sigma_soft per point for weighting.

    sigma_soft = sqrt( sigma_data^2 + (rel_floor*scale)^2 + abs_floor^2 )

    If USE_POINTWISE_SIGMAS=False, sigma_data is treated as 0 everywhere.
    """
    xs = y_central[:, 0].astype(float)
    bsa = y_central[:, 1].astype(float)
    xs_sig = y_sigma[:, 0].astype(float)
    bsa_sig = y_sigma[:, 1].astype(float)

    # Robust scales (avoid scale->0)
    xs_scale = float(np.median(np.abs(xs))) if np.isfinite(xs).any() else 1.0
    xs_scale = xs_scale if xs_scale > 0 else 1.0

    abs_bsa = np.abs(bsa[np.isfinite(bsa)])
    if abs_bsa.size == 0:
        bsa_scale = 1.0
    else:
        bsa_scale = float(np.median(abs_bsa))
        if bsa_scale <= 1e-6:
            bsa_scale = float(np.percentile(abs_bsa, 90))
        bsa_scale = bsa_scale if bsa_scale > 0 else 1.0

    if not bool(USE_POINTWISE_SIGMAS):
        xs_sig = np.zeros_like(xs_sig)
        bsa_sig = np.zeros_like(bsa_sig)

    xs_floor = float(SOFT_XS_REL) * xs_scale
    bsa_floor = float(SOFT_BSA_REL) * bsa_scale

    xs_soft = np.sqrt(xs_sig**2 + xs_floor**2 + float(SOFT_XS_ABS)**2)
    bsa_soft = np.sqrt(bsa_sig**2 + bsa_floor**2 + float(SOFT_BSA_ABS)**2)

    xs_soft = np.where(xs_soft > 0, xs_soft, 1.0)
    bsa_soft = np.where(bsa_soft > 0, bsa_soft, 1.0)

    return np.column_stack([xs_soft, bsa_soft]).astype(np.float32)


# -------------------------
# Discretized fixed-t dispersion relation (Hard DR)
# -------------------------
def build_dr_kernel(x_nodes: np.ndarray) -> np.ndarray:
    """
    Build a trapezoidal-quadrature DR kernel matrix K such that:

        ReH_i = C0 + sum_j K[i,j] * ImH_j

    with
        K[i,j] ~ (1/pi) * w_j * [ 1/(xi_i - x_j) - 1/(xi_i + x_j) ]

    The principal value at x_j=xi_i is approximated by setting K[i,i]=0.

    Notes:
    - This is a pedagogical discretization for a closure example.
    - In a production analysis you may want a dedicated quadrature grid on x in [0,1]
      and more careful PV treatment.
    """
    x = np.asarray(x_nodes, dtype=float)
    B = len(x)
    if B < 2:
        raise ValueError("Need at least 2 xi nodes for DR discretization.")

    # trapezoidal weights on the x-grid
    w = np.zeros(B, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if B > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])

    # kernel matrix
    K = np.zeros((B, B), dtype=float)
    for i in range(B):
        xi = x[i]
        for j in range(B):
            xj = x[j]
            if i == j:
                K[i, j] = 0.0  # PV prescription (skip singular)
            else:
                K[i, j] = (1.0 / PI) * w[j] * ((1.0 / (xi - xj)) - (1.0 / (xi + xj)))

    return K.astype(np.float32)


# -------------------------
# BKM forward op with custom gradient (finite differences)
# -------------------------
def _forward_np_single_bin(
    reh: float, imh: float, t: float, xB: float, Q2: float, phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute XS(phi), BSA(phi) using bkm10_lib for one (t,xB,Q2) bin."""
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
    xs = np.asarray(xsecs.compute_cross_section(phi).real, dtype=np.float32)
    bsa = np.asarray(xsecs.compute_bsa(phi).real, dtype=np.float32)
    return xs, bsa


def make_bkm_bin_op(fd_eps: float):
    """
    Create a TF op:
      (reh, imh, t, xB, Q2, phi_vec) -> (xs_vec, bsa_vec)
    with a custom gradient w.r.t reh and imh using central finite differences.

    We treat (t,xB,Q2,phi) as constants for gradient purposes in this example.
    """
    def _np(reh_np, imh_np, t_np, xB_np, Q2_np, phi_np):
        reh = float(np.asarray(reh_np).item())
        imh = float(np.asarray(imh_np).item())
        t = float(np.asarray(t_np).item())
        xB = float(np.asarray(xB_np).item())
        Q2 = float(np.asarray(Q2_np).item())
        phi = np.asarray(phi_np, dtype=float)
        xs, bsa = _forward_np_single_bin(reh, imh, t, xB, Q2, phi)
        return xs, bsa

    @tf.custom_gradient
    def op(reh_tf, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf):
        xs_tf, bsa_tf = tf.numpy_function(
            func=_np,
            inp=[reh_tf, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf],
            Tout=[_FLOATX, _FLOATX],
        )
        xs_tf.set_shape(phi_tf.shape)
        bsa_tf.set_shape(phi_tf.shape)

        def grad(dxs, dbsa):
            eps = tf.constant(float(fd_eps), dtype=_FLOATX)

            # d/dReH
            xs_p, bsa_p = tf.numpy_function(_np, [reh_tf + eps, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf], [_FLOATX, _FLOATX])
            xs_m, bsa_m = tf.numpy_function(_np, [reh_tf - eps, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf], [_FLOATX, _FLOATX])
            xs_p.set_shape(phi_tf.shape); xs_m.set_shape(phi_tf.shape)
            bsa_p.set_shape(phi_tf.shape); bsa_m.set_shape(phi_tf.shape)
            d_xs_d_reh = (xs_p - xs_m) / (2.0 * eps)
            d_bsa_d_reh = (bsa_p - bsa_m) / (2.0 * eps)

            # d/dImH
            xs_p2, bsa_p2 = tf.numpy_function(_np, [reh_tf, imh_tf + eps, t_tf, xB_tf, Q2_tf, phi_tf], [_FLOATX, _FLOATX])
            xs_m2, bsa_m2 = tf.numpy_function(_np, [reh_tf, imh_tf - eps, t_tf, xB_tf, Q2_tf, phi_tf], [_FLOATX, _FLOATX])
            xs_p2.set_shape(phi_tf.shape); xs_m2.set_shape(phi_tf.shape)
            bsa_p2.set_shape(phi_tf.shape); bsa_m2.set_shape(phi_tf.shape)
            d_xs_d_imh = (xs_p2 - xs_m2) / (2.0 * eps)
            d_bsa_d_imh = (bsa_p2 - bsa_m2) / (2.0 * eps)

            # Chain rule: sum over phi components
            g_reh = tf.reduce_sum(dxs * d_xs_d_reh + dbsa * d_bsa_d_reh)
            g_imh = tf.reduce_sum(dxs * d_xs_d_imh + dbsa * d_bsa_d_imh)

            # No gradients for kinematics or phi in this example
            return g_reh, g_imh, None, None, None, None

        return (xs_tf, bsa_tf), grad

    return op


# -------------------------
# Model: ImH(xi) network + subtraction constant, with Hard DR ReH = C0 + K*ImH
# -------------------------
class TrainableScalar(tf.keras.layers.Layer):
    """A single trainable scalar, e.g. the DR subtraction constant C0(t)."""
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
        # Broadcast scalar to batch as shape (B,)
        b = tf.shape(x)[0]
        return tf.ones((b,), dtype=_FLOATX) * self.scalar


def make_imh_network(seed: Optional[int] = None) -> tf.keras.Model:
    """
    Small MLP mapping kinematics -> ImH.
    For this closure script, the informative input is xi (via xB), but we keep (t,xB,Q2)
    so this can be extended to broader kinematics without rewriting the model.
    """
    init = tf.keras.initializers.RandomUniform(minval=-0.3, maxval=0.3, seed=seed)
    kin_in = tf.keras.Input(shape=(3,), name="kin")  # (t, xB, Q2)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init)(kin_in)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init)(x)
    imh = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init, name="ImH")(x)
    return tf.keras.Model(inputs=kin_in, outputs=imh, name="ImHNet")


class HardDRHModel(tf.keras.Model):
    """
    End-to-end model for Hard-DR extraction of H:
      - learns ImH(kin) via MLP
      - learns subtraction constant C0 via TrainableScalar
      - computes ReH on the xi-grid using a fixed kernel K (PV trapezoid)
    """
    def __init__(self, K: np.ndarray, seed: Optional[int] = None):
        super().__init__(name="HardDRHModel")
        self.imh_net = make_imh_network(seed=seed)
        self.c0_layer = TrainableScalar(init_value=0.0, name="C0")
        self.K = tf.constant(K, dtype=_FLOATX)  # shape (B,B)

    def call(self, kin: tf.Tensor):
        """
        kin: (B,3) tensor of (t, xB, Q2) for all xi nodes in the DR grid.
        Returns:
          reh: (B,) ReH at each node (Hard DR)
          imh: (B,) ImH at each node (learned)
          c0:  ()   scalar subtraction constant (same for all nodes)
        """
        imh = tf.squeeze(self.imh_net(kin, training=True), axis=1)  # (B,)
        c0_vec = self.c0_layer(kin)  # (B,) broadcast
        c0 = c0_vec[0]
        reh = c0 + tf.linalg.matvec(self.K, imh)  # (B,)
        return reh, imh, c0


# -------------------------
# Loss computation
# -------------------------
def compute_loss(
    bkm_op,
    reh: tf.Tensor,
    imh: tf.Tensor,
    kin_bins: tf.Tensor,
    phi_grid: tf.Tensor,
    xs_obs: tf.Tensor,
    bsa_obs: tf.Tensor,
    xs_sig: tf.Tensor,
    bsa_sig: tf.Tensor,
    ratio_clip: Optional[float] = None,
) -> tf.Tensor:
    """
    Compute soft-chi2 loss over all xi nodes and selected phi points.

    Shapes:
      reh, imh:   (B,)
      kin_bins:   (B,3) -> (t,xB,Q2)
      phi_grid:   (Nphi,)  common phi grid
      xs_obs:     (B,Nphi)
      bsa_obs:    (B,Nphi)
      xs_sig:     (B,Nphi)
      bsa_sig:    (B,Nphi)
    """

    # Map over bins because bkm10_lib forward builds a BKM10Inputs per bin.
    def _one_bin(args):
        reh_b, imh_b, kin_b = args
        t_b = kin_b[0]
        xB_b = kin_b[1]
        Q2_b = kin_b[2]
        xs_pred_b, bsa_pred_b = bkm_op(reh_b, imh_b, t_b, xB_b, Q2_b, phi_grid)
        return xs_pred_b, bsa_pred_b

    xs_pred, bsa_pred = tf.map_fn(
        _one_bin,
        (reh, imh, kin_bins),
        fn_output_signature=(tf.TensorSpec(shape=phi_grid.shape, dtype=_FLOATX),
                             tf.TensorSpec(shape=phi_grid.shape, dtype=_FLOATX)),
    )  # outputs (B,Nphi)

    # safety for non-finite
    big = tf.constant(1e30, dtype=_FLOATX)
    xs_pred = tf.where(tf.math.is_finite(xs_pred), xs_pred, big * tf.ones_like(xs_pred))
    bsa_pred = tf.where(tf.math.is_finite(bsa_pred), bsa_pred, big * tf.ones_like(bsa_pred))

    rx = (xs_obs - xs_pred) / xs_sig
    rb = (bsa_obs - bsa_pred) / bsa_sig

    if ratio_clip is not None and ratio_clip > 0:
        c = tf.constant(float(ratio_clip), dtype=_FLOATX)
        rx = tf.clip_by_value(rx, -c, c)
        rb = tf.clip_by_value(rb, -c, c)

    return 0.5 * (tf.reduce_mean(tf.square(rx)) + tf.reduce_mean(tf.square(rb)))


# -------------------------
# Training loop (replicas)
# -------------------------
def main() -> None:
    _safe_mkdir(MODELS_DIR)
    _safe_mkdir(HIST_DIR)

    d = _load_npz(DATA_NPZ)
    X = d["x"].astype(np.float32)                 # (N,4) [t,xB,Q2,phi]
    y_central = d["y_central"].astype(np.float32) # (N,2) [XS,BSA]
    y_sigma = d["y_sigma"].astype(np.float32)     # (N,2) sampling sigmas

    N = X.shape[0]
    if X.shape[1] != 4 or y_central.shape != (N, 2) or y_sigma.shape != (N, 2):
        raise ValueError(f"Unexpected shapes: X={X.shape}, y_central={y_central.shape}, y_sigma={y_sigma.shape}")

    # Build sigma_soft (for loss weighting)
    sigma_soft = soft_sigmas(y_central, y_sigma)

    print("Loss sigma_soft medians (weighting only):")
    print("  XS sigma_soft median  = {:.6g}".format(float(np.median(sigma_soft[:, 0]))))
    print("  BSA sigma_soft median = {:.6g}".format(float(np.median(sigma_soft[:, 1]))))
    print("  USE_POINTWISE_SIGMAS  =", bool(USE_POINTWISE_SIGMAS))
    if float(np.max(y_sigma[:, 0])) == 0.0 and float(np.max(y_sigma[:, 1])) == 0.0:
        print("NOTE: dataset y_sigma is identically zero -> replicas are identical; any spread is optimizer/init only.")
        print("      This script still uses nonzero sigma_soft, so optimization behaves like a moderate-error fit.")

    # Group into (t,xB,Q2) bins
    bins_central = group_by_kinematics(X, y_central, sigma_soft)
    if len(bins_central) < 3:
        raise ValueError(
            f"Hard DR needs multiple xi nodes; found only {len(bins_central)} unique (t,xB,Q2) bins.\n"
            "Regenerate closure dataset with multiple xB values at fixed (t,Q2) and same phi grid."
        )

    phi_grid = assert_common_phi_grid(bins_central)
    Nphi = len(phi_grid)

    # Sort bins by increasing xi (stable DR kernel ordering)
    keys_sorted = sorted(bins_central.keys(), key=lambda k: xi_from_xB(k[1]))
    t_bins = np.array([k[0] for k in keys_sorted], dtype=np.float32)
    xB_bins = np.array([k[1] for k in keys_sorted], dtype=np.float32)
    Q2_bins = np.array([k[2] for k in keys_sorted], dtype=np.float32)

    xi_bins = xi_from_xB(xB_bins).astype(np.float32)
    print(f"Number of xi nodes (bins) = {len(xi_bins)}")
    print(f"xi range: [{float(np.min(xi_bins)):.4g}, {float(np.max(xi_bins)):.4g}]")

    # Observed arrays in sorted bin order
    xs_c = np.stack([bins_central[k]["xs"] for k in keys_sorted], axis=0)         # (B,Nphi)
    bsa_c = np.stack([bins_central[k]["bsa"] for k in keys_sorted], axis=0)       # (B,Nphi)
    xs_sig = np.stack([bins_central[k]["xs_sig"] for k in keys_sorted], axis=0)   # (B,Nphi)
    bsa_sig = np.stack([bins_central[k]["bsa_sig"] for k in keys_sorted], axis=0) # (B,Nphi)

    # Train/val split over phi points (shared across bins)
    rng_split = np.random.default_rng(int(SPLIT_SEED))
    phi_idx = np.arange(Nphi)
    rng_split.shuffle(phi_idx)
    n_train_phi = int(np.floor(float(TRAIN_FRACTION) * Nphi))
    n_train_phi = max(1, min(Nphi, n_train_phi))
    train_phi_idx = np.sort(phi_idx[:n_train_phi])
    val_phi_idx = np.sort(phi_idx[n_train_phi:]) if n_train_phi < Nphi else np.array([], dtype=int)

    print(f"Phi split: Nphi={Nphi}, train={len(train_phi_idx)}, val={len(val_phi_idx)}")

    # Tensors
    kin_bins = np.stack([t_bins, xB_bins, Q2_bins], axis=1).astype(np.float32)  # (B,3)

    phi_tf = tf.convert_to_tensor(phi_grid, dtype=_FLOATX)
    kin_tf = tf.convert_to_tensor(kin_bins, dtype=_FLOATX)

    # DR kernel on the xi grid
    # NOTE: For a pedagogical closure demo we discretize the DR integral on the same xi-grid.
    K = build_dr_kernel(x_nodes=xi_bins.astype(float))

    # Build BKM op
    bkm_op = make_bkm_bin_op(fd_eps=float(FD_EPS))

    rng_rep = np.random.default_rng(int(REPLICA_SEED))

    for r in range(int(N_REPLICAS)):
        seed = int(REPLICA_SEED) + 1000 * (r + 1)

        # Replica sampling: draw noisy pseudo-data for each point in original NPZ,
        # then regroup into bins with the same ordering.
        noise = rng_rep.normal(0.0, 1.0, size=y_central.shape).astype(np.float32)
        y_rep_pointwise = y_central + noise * y_sigma

        bins_rep = group_by_kinematics(X, y_rep_pointwise, sigma_soft)
        xs_obs = np.stack([bins_rep[k]["xs"] for k in keys_sorted], axis=0).astype(np.float32)
        bsa_obs = np.stack([bins_rep[k]["bsa"] for k in keys_sorted], axis=0).astype(np.float32)

        xs_obs_tf = tf.convert_to_tensor(xs_obs, dtype=_FLOATX)
        bsa_obs_tf = tf.convert_to_tensor(bsa_obs, dtype=_FLOATX)
        xs_sig_tf = tf.convert_to_tensor(xs_sig, dtype=_FLOATX)
        bsa_sig_tf = tf.convert_to_tensor(bsa_sig, dtype=_FLOATX)

        # Model init
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = HardDRHModel(K=K, seed=seed)

        if ADAM_CLIPNORM and float(ADAM_CLIPNORM) > 0:
            opt = tf.keras.optimizers.Adam(learning_rate=float(LEARNING_RATE), clipnorm=float(ADAM_CLIPNORM))
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=float(LEARNING_RATE))

        history = {"loss": [], "val_loss": []}
        best_val = np.inf
        best_weights = None
        bad_epochs = 0

        # Helper: compute loss on selected phi indices
        def loss_on_phi(phi_sel: np.ndarray) -> tf.Tensor:
            reh, imh, c0 = model(kin_tf)

            phi_sub = tf.gather(phi_tf, phi_sel, axis=0)
            xs_obs_sub = tf.gather(xs_obs_tf, phi_sel, axis=1)
            bsa_obs_sub = tf.gather(bsa_obs_tf, phi_sel, axis=1)
            xs_sig_sub = tf.gather(xs_sig_tf, phi_sel, axis=1)
            bsa_sig_sub = tf.gather(bsa_sig_tf, phi_sel, axis=1)

            return compute_loss(
                bkm_op=bkm_op,
                reh=reh,
                imh=imh,
                kin_bins=kin_tf,
                phi_grid=phi_sub,
                xs_obs=xs_obs_sub,
                bsa_obs=bsa_obs_sub,
                xs_sig=xs_sig_sub,
                bsa_sig=bsa_sig_sub,
                ratio_clip=float(RATIO_CLIP) if RATIO_CLIP else None,
            )

        # Training loop
        for epoch in range(int(EPOCHS)):
            with tf.GradientTape() as tape:
                loss = loss_on_phi(train_phi_idx)

            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            loss_val = loss
            if len(val_phi_idx) > 0:
                loss_val = loss_on_phi(val_phi_idx)

            l = float(loss.numpy())
            lv = float(loss_val.numpy())
            history["loss"].append(l)
            history["val_loss"].append(lv)

            metric = lv if len(val_phi_idx) > 0 else l
            if metric < best_val - 1e-12:
                best_val = metric
                best_weights = model.get_weights()
                bad_epochs = 0
            else:
                bad_epochs += 1

            if (epoch + 1) % 50 == 0 or epoch == 0:
                reh_hat, imh_hat, c0_hat = model(kin_tf)
                msg = (
                    f"Replica {r+1:03d} | epoch {epoch+1:4d} "
                    f"| loss={l:.6g} val={lv:.6g} "
                    f"| C0={float(c0_hat.numpy()):+.4g} "
                    f"| ImH(xi0)={float(imh_hat.numpy()[0]):+.4g}"
                )
                print(msg)

            if bad_epochs >= int(PATIENCE):
                break

        # Restore best weights
        if best_weights is not None:
            model.set_weights(best_weights)

        # Save model/weights + history
        base = os.path.join(MODELS_DIR, f"replica_{r+1:03d}_{TAG}")
        if SAVE_WEIGHTS_ONLY:
            model.save_weights(base + ".weights.h5")
            # Save DR grid + kernel so the replica is fully reconstructable
            np.savez(
                base + "_meta.npz",
                xi_bins=xi_bins.astype(np.float32),
                t_bins=t_bins.astype(np.float32),
                xB_bins=xB_bins.astype(np.float32),
                Q2_bins=Q2_bins.astype(np.float32),
                K=K.astype(np.float32),
            )
        else:
            # Try saving full model; if your environment can't serialize subclassed models,
            # set SAVE_WEIGHTS_ONLY=True.
            model.save(base + ".keras")

        hist_path = os.path.join(HIST_DIR, f"history_replica_{r+1:03d}_{TAG}.json")
        _save_json(hist_path, history)

        # Post-fit summary
        reh_hat, imh_hat, c0_hat = model(kin_tf)
        reh_hat_np = reh_hat.numpy().astype(float)
        imh_hat_np = imh_hat.numpy().astype(float)
        c0_hat_np = float(c0_hat.numpy())

        msg = (
            f"Replica {r+1:03d}: C0={c0_hat_np:+.6g} | "
            f"ImH range=[{imh_hat_np.min():+.4g},{imh_hat_np.max():+.4g}] | "
            f"ReH range=[{reh_hat_np.min():+.4g},{reh_hat_np.max():+.4g}]"
        )

        if PRINT_POSTFIT_RESIDUALS:
            xs_pred_all = []
            bsa_pred_all = []
            for b in range(len(keys_sorted)):
                xs_p, bsa_p = _forward_np_single_bin(
                    reh=float(reh_hat_np[b]),
                    imh=float(imh_hat_np[b]),
                    t=float(t_bins[b]),
                    xB=float(xB_bins[b]),
                    Q2=float(Q2_bins[b]),
                    phi=phi_grid.astype(float),
                )
                xs_pred_all.append(xs_p)
                bsa_pred_all.append(bsa_p)
            xs_pred_all = np.stack(xs_pred_all, axis=0)
            bsa_pred_all = np.stack(bsa_pred_all, axis=0)

            max_dx = float(np.max(np.abs(xs_obs - xs_pred_all)))
            max_db = float(np.max(np.abs(bsa_obs - bsa_pred_all)))
            msg += f" | max|ΔXS|={max_dx:.3e}, max|ΔBSA|={max_db:.3e}"

        print(msg)

    print("\nSaved models/weights to:", MODELS_DIR)
    print("Saved histories to:", HIST_DIR)


if __name__ == "__main__":
    main()
