#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training script for gated dispersion example (Python 3.9+)

This code is meant to be a basic exmaple for optimized replica training for a Gated-DR + differentiable PV (principal value) layer
closure test.

The main idea of this example:
-----------------------------
1) Hard-DR should converge BEFORE any correction is allowed.
2) Release correction safely: correction=0 at release, gate~1 at release.
3) Penalize the *effective correction* corr = alpha*(1-g)*delta (the actual DR violation).
4) Ramp correction on gradually after release (alpha ramp), to avoid basin jumps.
5) If the dataset NPZ provides a kernel matrix K, use it (guarantees closure consistency).

Outputs
-------
<version>/replicas_gated_dr_pv_opt/replica_XXX_<TAG>.weights.h5
<version>/replicas_gated_dr_pv_opt/replica_XXX_<TAG>_meta.npz
<version>/histories_gated_dr_pv_opt/history_replica_XXX_<TAG>.json
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
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
TAG = "v_1"  # <-- whatever tag you want, needed to read file (NPZ should be dataset_<TAG>.npz)
DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")

MODELS_DIR = os.path.join(VERSION_DIR, "replicas_gated_dr_pv_opt")
HIST_DIR = os.path.join(VERSION_DIR, "histories_gated_dr_pv_opt")

N_REPLICAS = 50
REPLICA_SEED = 20250101

# Train/val split over phi points (shared across bins)
TRAIN_FRACTION = 0.8
SPLIT_SEED = 1234

# ========= Stage 1: Hard-DR optimization =========
STAGE1_LR = 3e-3
STAGE1_MAX_EPOCHS = 5000
STAGE1_MIN_EPOCHS = 300
STAGE1_PATIENCE = 500

# Adaptive release: only allow correction if Hard-DR data-loss is already "good enough"
# For your sigma_soft floors, values ~0.1-0.5 are reasonable targets.
RELEASE_TARGET_VAL_DATA = 0.30
RELEASE_FORCE_EPOCH = 2500   # safety: release anyway if you want (or set None to never force)

# ========= Stage 2: Gated soft-DR fine-tune =========
STAGE2_LR = 8e-4
STAGE2_MAX_EPOCHS = 2500
STAGE2_MIN_EPOCHS = 200
STAGE2_PATIENCE = 600

# Ramp correction on over this many epochs after release (alpha goes 0->1)
RAMP_EPOCHS = 400

# FD epsilon for bkm gradients
FD_EPS = 5e-3

# Gradient clipping
ADAM_CLIPNORM = 5.0

# Residual clipping after division by sigma_soft
RATIO_CLIP = 1e4

# ===== Soft-chi2 weighting (same idea as your original script) =====
USE_POINTWISE_SIGMAS = True
SOFT_XS_REL = 0.02
SOFT_XS_ABS = 0.0
SOFT_BSA_REL = 0.0
SOFT_BSA_ABS = 0.01

# ===== Output scaling (tanh parameterization) =====
IMH_SCALE = 8.0
DELTA_REH_SCALE = 2.0  # smaller is safer if you want DR preferred

# ===== Gate + delta initialization (critical!) =====
# Gate starts near 1: sigmoid(6)=0.9975
GATE_BIAS_INIT = 6.0

# Delta starts exactly 0 (kernel/bias zeros) => no random jump at release
DELTA_INIT_ZERO = True

# ===== Regularization (principled) =====
# Always (Stage 1 & 2): encourage smooth ImH(xi)
LAMBDA_SMOOTH_IMH = 1e-4

# Stage 2 only: DR preference penalties
# These should be LARGE enough that corr_rms ~ O(1) is expensive.
LAMBDA_CORR_L2 = 3.0          # penalize corr = alpha*(1-g)*delta
LAMBDA_GATE_PRIOR = 1.0       # encourage g -> 1
LAMBDA_DELTA_L2 = 0.05        # keep delta modest (secondary)
LAMBDA_SMOOTH_DELTA = 1e-4    # keep delta smooth (optional)

# ===== Diagnostics =====
PRINT_EVERY = 50
PRINT_POSTFIT_RESIDUALS = True

# Fixed beam energy & bkm settings (must match generator)
BEAM_ENERGY = 5.75
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# Fixed nuisance CFFs (must match generator for closure)
CFF_E  = complex(2.217354372014208, 0.0)
CFF_HT = complex(1.409393726454478, 1.57736440256014)
CFF_ET = complex(144.4101642020152, 0.0)

# =========================
# END USER CONFIG
# =========================

_FLOATX = tf.float32
PI = float(np.pi)


# -------------------------
# Utilities
# -------------------------
def _safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    for k in ("x", "y_central", "y_sigma"):
        if k not in d.files:
            raise KeyError(f"NPZ missing key '{k}', found {d.files}")
    return d

def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    xB = np.asarray(xB, dtype=float)
    return xB / (2.0 - xB)

def _soft_sigmas(y_central: np.ndarray, y_sigma: np.ndarray) -> np.ndarray:
    xs = y_central[:, 0].astype(float)
    bsa = y_central[:, 1].astype(float)
    xs_sig = y_sigma[:, 0].astype(float)
    bsa_sig = y_sigma[:, 1].astype(float)

    xs_scale = float(np.median(np.abs(xs))) if np.isfinite(xs).any() else 1.0
    xs_scale = xs_scale if xs_scale > 0 else 1.0

    abs_bsa = np.abs(bsa[np.isfinite(bsa)])
    bsa_scale = float(np.median(abs_bsa)) if abs_bsa.size else 1.0
    if bsa_scale <= 1e-6 and abs_bsa.size:
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

def group_by_kinematics(
    X: np.ndarray,
    y: np.ndarray,
    sigma_soft: np.ndarray,
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
            "xs_sig": sigma_soft[mask, 0][order].astype(np.float32),
            "bsa_sig": sigma_soft[mask, 1][order].astype(np.float32),
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


# -------------------------
# bkm10 forward (vector op) with FD custom gradient
# -------------------------
def _forward_np_single_bin(reh: float, imh: float, t: float, xB: float, Q2: float, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def make_bkm_vector_op(B: int, fd_eps: float):
    """
    (reh[B], imh[B], t[B], xB[B], Q2[B], phi[N]) -> (xs[B,N], bsa[B,N])
    with custom gradient returning grads for reh and imh (vectors).
    """
    def _forward_np(reh_np, imh_np, t_np, xB_np, Q2_np, phi_np):
        reh = np.asarray(reh_np, dtype=float).reshape(-1)
        imh = np.asarray(imh_np, dtype=float).reshape(-1)
        t = np.asarray(t_np, dtype=float).reshape(-1)
        xB = np.asarray(xB_np, dtype=float).reshape(-1)
        Q2 = np.asarray(Q2_np, dtype=float).reshape(-1)
        phi = np.asarray(phi_np, dtype=float).reshape(-1)

        xs = np.empty((reh.shape[0], phi.shape[0]), dtype=np.float32)
        bsa = np.empty((reh.shape[0], phi.shape[0]), dtype=np.float32)

        for b in range(reh.shape[0]):
            xs_b, bsa_b = _forward_np_single_bin(reh[b], imh[b], t[b], xB[b], Q2[b], phi)
            xs[b, :] = xs_b
            bsa[b, :] = bsa_b
        return xs, bsa

    def _grads_np(reh_np, imh_np, t_np, xB_np, Q2_np, phi_np, dxs_np, dbsa_np):
        reh = np.asarray(reh_np, dtype=float).reshape(-1)
        imh = np.asarray(imh_np, dtype=float).reshape(-1)
        t = np.asarray(t_np, dtype=float).reshape(-1)
        xB = np.asarray(xB_np, dtype=float).reshape(-1)
        Q2 = np.asarray(Q2_np, dtype=float).reshape(-1)
        phi = np.asarray(phi_np, dtype=float).reshape(-1)

        dxs = np.asarray(dxs_np, dtype=float)
        dbsa = np.asarray(dbsa_np, dtype=float)

        eps = float(fd_eps)
        g_reh = np.zeros_like(reh, dtype=np.float32)
        g_imh = np.zeros_like(imh, dtype=np.float32)

        for b in range(reh.shape[0]):
            xs_p, bsa_p = _forward_np_single_bin(reh[b] + eps, imh[b], t[b], xB[b], Q2[b], phi)
            xs_m, bsa_m = _forward_np_single_bin(reh[b] - eps, imh[b], t[b], xB[b], Q2[b], phi)
            d_xs_d_reh = (xs_p - xs_m) / (2.0 * eps)
            d_bsa_d_reh = (bsa_p - bsa_m) / (2.0 * eps)

            xs_p2, bsa_p2 = _forward_np_single_bin(reh[b], imh[b] + eps, t[b], xB[b], Q2[b], phi)
            xs_m2, bsa_m2 = _forward_np_single_bin(reh[b], imh[b] - eps, t[b], xB[b], Q2[b], phi)
            d_xs_d_imh = (xs_p2 - xs_m2) / (2.0 * eps)
            d_bsa_d_imh = (bsa_p2 - bsa_m2) / (2.0 * eps)

            g_reh[b] = np.sum(dxs[b, :] * d_xs_d_reh + dbsa[b, :] * d_bsa_d_reh).astype(np.float32)
            g_imh[b] = np.sum(dxs[b, :] * d_xs_d_imh + dbsa[b, :] * d_bsa_d_imh).astype(np.float32)

        return g_reh, g_imh

    @tf.custom_gradient
    def op(reh_tf, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf):
        xs_tf, bsa_tf = tf.numpy_function(
            _forward_np,
            [reh_tf, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf],
            [_FLOATX, _FLOATX],
        )
        xs_tf.set_shape((B, None))
        bsa_tf.set_shape((B, None))

        def grad(dxs, dbsa):
            g_reh, g_imh = tf.numpy_function(
                _grads_np,
                [reh_tf, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf, dxs, dbsa],
                [_FLOATX, _FLOATX],
            )
            g_reh.set_shape(reh_tf.shape)
            g_imh.set_shape(imh_tf.shape)
            return g_reh, g_imh, None, None, None, None

        return (xs_tf, bsa_tf), grad

    return op


# -------------------------
# PV dispersion layer
# -------------------------
def build_pv_kernel_trapezoid(xi_nodes: np.ndarray) -> np.ndarray:
    """
    Build K[i,j] = (1/pi) * w_j * ( 1/(xi_i - xi_j) - 1/(xi_i + xi_j) ), with PV diag=0.
    Trapezoid weights on the xi grid.
    """
    xi = np.asarray(xi_nodes, dtype=float).reshape(-1)
    B = xi.size
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

class PVDispersionLayer(tf.keras.layers.Layer):
    def __init__(self, xi_nodes: np.ndarray, K_matrix: Optional[np.ndarray] = None, name: str = "pv_dr"):
        super().__init__(name=name)
        xi = np.asarray(xi_nodes, dtype=np.float32).reshape(-1)
        self.xi_nodes = tf.constant(xi, dtype=_FLOATX)
        self.B = int(xi.size)

        if K_matrix is None:
            K_matrix = build_pv_kernel_trapezoid(xi)
        K_matrix = np.asarray(K_matrix, dtype=np.float32)

        if K_matrix.shape != (self.B, self.B):
            raise ValueError(f"K_matrix must have shape ({self.B},{self.B}), got {K_matrix.shape}")

        self.K = tf.Variable(K_matrix, trainable=False, dtype=_FLOATX, name="K")

    def call(self, imh: tf.Tensor, c0: tf.Tensor) -> tf.Tensor:
        # reh_dr = K @ imh + c0
        return tf.linalg.matvec(self.K, imh) + c0


# -------------------------
# Model (DNN trunk + heads)
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

class GatedDRModel(tf.keras.Model):
    """
    Predict:
      ImH(xi_i), DeltaReH(xi_i), gate(xi_i), and a global C0
    Construct:
      ReH_DR = C0 + K @ ImH
      ReH = ReH_DR + (1-gate)*DeltaReH   (correction is applied outside w/ ramp alpha)
    """
    def __init__(self, xi_nodes: np.ndarray, K_matrix: Optional[np.ndarray] = None, seed: Optional[int] = None):
        super().__init__(name="GatedDRModel_OPT")
        init = tf.keras.initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=seed)

        self.pv = PVDispersionLayer(xi_nodes=xi_nodes, K_matrix=K_matrix)
        self.c0_layer = TrainableScalar(init_value=0.0, name="C0")

        self.d1 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init)
        self.d2 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init)

        # ImH head: normal init
        self.h_imh = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init, name="imh_raw")

        # Delta head: start at exactly 0 to avoid a jump at release
        if DELTA_INIT_ZERO:
            delta_kernel_init = tf.keras.initializers.Zeros()
            delta_bias_init = tf.keras.initializers.Zeros()
        else:
            delta_kernel_init = init
            delta_bias_init = tf.keras.initializers.Zeros()

        self.h_delta = tf.keras.layers.Dense(
            1, activation="linear",
            kernel_initializer=delta_kernel_init,
            bias_initializer=delta_bias_init,
            name="delta_raw"
        )

        # Gate head: start near 1, input-independent initially
        self.h_gate = tf.keras.layers.Dense(
            1, activation="linear",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Constant(float(GATE_BIAS_INIT)),
            name="gate_logit"
        )

    def call(self, kin: tf.Tensor, training: bool = False):
        x = self.d1(kin, training=training)
        x = self.d2(x, training=training)

        imh_raw = tf.squeeze(self.h_imh(x, training=training), axis=1)
        delta_raw = tf.squeeze(self.h_delta(x, training=training), axis=1)
        gate_logit = tf.squeeze(self.h_gate(x, training=training), axis=1)

        imh = tf.constant(float(IMH_SCALE), dtype=_FLOATX) * tf.tanh(imh_raw)
        delta = tf.constant(float(DELTA_REH_SCALE), dtype=_FLOATX) * tf.tanh(delta_raw)
        gate = tf.sigmoid(gate_logit)

        c0_vec = self.c0_layer(kin)
        c0 = c0_vec[0]

        reh_dr = self.pv(imh, c0)
        # reh_total is assembled outside, since we use a ramp alpha there
        return reh_dr, imh, delta, gate, c0


# -------------------------
# Losses / penalties
# -------------------------
def smoothness_penalty(v: tf.Tensor) -> tf.Tensor:
    if v.shape[0] is not None and v.shape[0] < 3:
        return tf.constant(0.0, dtype=_FLOATX)
    d2 = v[2:] - 2.0 * v[1:-1] + v[:-2]
    return tf.reduce_mean(tf.square(d2))

def compute_softchi2_loss(
    bkm_op,
    reh: tf.Tensor,              # (B,)
    imh: tf.Tensor,              # (B,)
    t_bins: tf.Tensor,           # (B,)
    xB_bins: tf.Tensor,          # (B,)
    Q2_bins: tf.Tensor,          # (B,)
    phi: tf.Tensor,              # (Nphi_sel,)
    xs_obs: tf.Tensor,           # (B,Nphi_sel)
    bsa_obs: tf.Tensor,          # (B,Nphi_sel)
    xs_sig: tf.Tensor,           # (B,Nphi_sel)
    bsa_sig: tf.Tensor,          # (B,Nphi_sel)
) -> tf.Tensor:
    xs_pred, bsa_pred = bkm_op(reh, imh, t_bins, xB_bins, Q2_bins, phi)

    big = tf.constant(1e30, dtype=_FLOATX)
    xs_pred = tf.where(tf.math.is_finite(xs_pred), xs_pred, big * tf.ones_like(xs_pred))
    bsa_pred = tf.where(tf.math.is_finite(bsa_pred), bsa_pred, big * tf.ones_like(bsa_pred))

    rx = (xs_obs - xs_pred) / xs_sig
    rb = (bsa_obs - bsa_pred) / bsa_sig

    if RATIO_CLIP and float(RATIO_CLIP) > 0:
        c = tf.constant(float(RATIO_CLIP), dtype=_FLOATX)
        rx = tf.clip_by_value(rx, -c, c)
        rb = tf.clip_by_value(rb, -c, c)

    return 0.5 * (tf.reduce_mean(tf.square(rx)) + tf.reduce_mean(tf.square(rb)))


# -------------------------
# Training helpers
# -------------------------
def pick_phi_split(Nphi: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(SPLIT_SEED))
    idx = np.arange(Nphi)
    rng.shuffle(idx)
    n_train = max(1, int(np.floor(float(TRAIN_FRACTION) * Nphi)))
    train_idx = np.sort(idx[:n_train])
    val_idx = np.sort(idx[n_train:]) if n_train < Nphi else np.array([], dtype=int)
    return train_idx, val_idx

def ramp_alpha(epoch_since_release: int) -> float:
    if RAMP_EPOCHS <= 0:
        return 1.0
    a = float(epoch_since_release) / float(RAMP_EPOCHS)
    return 0.0 if a < 0 else (1.0 if a > 1.0 else a)


def main() -> None:
    _safe_mkdir(MODELS_DIR)
    _safe_mkdir(HIST_DIR)

    d = _load_npz(DATA_NPZ)
    X = d["x"].astype(np.float32)
    y_central = d["y_central"].astype(np.float32)
    y_sigma = d["y_sigma"].astype(np.float32)

    # If kernel is saved in NPZ, use it (best for closure consistency)
    K_from_file = None
    for key in ("K", "K_matrix", "pv_K", "K_pv"):
        if key in d.files:
            K_from_file = d[key].astype(np.float32)
            print(f"Loaded PV kernel from NPZ key '{key}' with shape {K_from_file.shape}")
            break

    sigma_soft = _soft_sigmas(y_central, y_sigma)

    bins_central = group_by_kinematics(X, y_central, sigma_soft)
    phi_grid = assert_common_phi_grid(bins_central)
    Nphi = len(phi_grid)

    # Sort bins by xi increasing
    keys_sorted = sorted(bins_central.keys(), key=lambda k: xi_from_xB(k[1]))
    B = len(keys_sorted)

    t_bins = np.array([k[0] for k in keys_sorted], dtype=np.float32)
    xB_bins = np.array([k[1] for k in keys_sorted], dtype=np.float32)
    Q2_bins = np.array([k[2] for k in keys_sorted], dtype=np.float32)
    xi_bins = xi_from_xB(xB_bins).astype(np.float32)

    print(f"Bins: B={B}, xi range=[{float(np.min(xi_bins)):.6g},{float(np.max(xi_bins)):.6g}], Nphi={Nphi}")

    # Pack sigmas (B,Nphi)
    xs_sig = np.stack([bins_central[k]["xs_sig"] for k in keys_sorted], axis=0).astype(np.float32)
    bsa_sig = np.stack([bins_central[k]["bsa_sig"] for k in keys_sorted], axis=0).astype(np.float32)

    # Phi split indices
    train_phi_idx, val_phi_idx = pick_phi_split(Nphi)
    print(f"Phi split: train={len(train_phi_idx)}, val={len(val_phi_idx)}")

    # TF constants
    phi_tf = tf.constant(phi_grid, dtype=_FLOATX)
    t_tf = tf.constant(t_bins, dtype=_FLOATX)
    xB_tf = tf.constant(xB_bins, dtype=_FLOATX)
    Q2_tf = tf.constant(Q2_bins, dtype=_FLOATX)

    xs_sig_tf = tf.constant(xs_sig, dtype=_FLOATX)
    bsa_sig_tf = tf.constant(bsa_sig, dtype=_FLOATX)

    # Kinematics features to the model (B,3)
    kin_bins = np.stack([t_bins, xB_bins, Q2_bins], axis=1).astype(np.float32)
    kin_tf = tf.constant(kin_bins, dtype=_FLOATX)

    # bkm op
    bkm_op = make_bkm_vector_op(B=B, fd_eps=float(FD_EPS))

    # Replica noise generator
    rng_rep = np.random.default_rng(int(REPLICA_SEED))

    # Legacy Adam on Apple Silicon is typically faster
    Adam = getattr(tf.keras.optimizers, "legacy", tf.keras.optimizers).Adam

    for r in range(int(N_REPLICAS)):
        seed = int(REPLICA_SEED) + 1000 * (r + 1)
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Replica sample (pointwise)
        noise = rng_rep.normal(0.0, 1.0, size=y_central.shape).astype(np.float32)
        y_rep_pointwise = y_central + noise * y_sigma
        bins_rep = group_by_kinematics(X, y_rep_pointwise, sigma_soft)

        xs_obs = np.stack([bins_rep[k]["xs"] for k in keys_sorted], axis=0).astype(np.float32)
        bsa_obs = np.stack([bins_rep[k]["bsa"] for k in keys_sorted], axis=0).astype(np.float32)

        xs_obs_tf = tf.constant(xs_obs, dtype=_FLOATX)
        bsa_obs_tf = tf.constant(bsa_obs, dtype=_FLOATX)

        # Model
        model = GatedDRModel(xi_nodes=xi_bins, K_matrix=K_from_file, seed=seed)
        _ = model(kin_tf, training=False)  # build

        # Two optimizers (stage-specific LR)
        opt1 = Adam(learning_rate=float(STAGE1_LR), clipnorm=float(ADAM_CLIPNORM) if ADAM_CLIPNORM else None)
        opt2 = Adam(learning_rate=float(STAGE2_LR), clipnorm=float(ADAM_CLIPNORM) if ADAM_CLIPNORM else None)

        history = {
            "stage": [],
            "epoch": [],
            "loss": [],
            "val_loss": [],
            "loss_data": [],
            "val_loss_data": [],
            "loss_reg": [],
            "corr_rms": [],
            "gate_mean": [],
            "C0": [],
        }

        # -------------------------
        # Stage 1: Hard-DR only
        # -------------------------
        best_val = np.inf
        best_weights = None
        bad = 0
        released = False
        release_epoch_global = None  # epoch index in stage1 when released (if we do in stage1)
        # Note: in this optimized script, we *finish* stage1, then stage2 begins after release.
        # But we still use adaptive criterion to decide whether to even do stage2.

        for ep in range(int(STAGE1_MAX_EPOCHS)):
            phi_train = tf.gather(phi_tf, train_phi_idx, axis=0)
            phi_val = tf.gather(phi_tf, val_phi_idx, axis=0) if len(val_phi_idx) else None

            # ----- TRAIN STEP -----
            with tf.GradientTape() as tape:
                reh_dr, imh, delta, gate, c0 = model(kin_tf, training=True)

                # Hard-DR: no correction
                reh_eff = reh_dr

                xs_train = tf.gather(xs_obs_tf, train_phi_idx, axis=1)
                bsa_train = tf.gather(bsa_obs_tf, train_phi_idx, axis=1)
                xs_sig_train = tf.gather(xs_sig_tf, train_phi_idx, axis=1)
                bsa_sig_train = tf.gather(bsa_sig_tf, train_phi_idx, axis=1)

                loss_data = compute_softchi2_loss(
                    bkm_op=bkm_op,
                    reh=reh_eff,
                    imh=imh,
                    t_bins=t_tf, xB_bins=xB_tf, Q2_bins=Q2_tf,
                    phi=phi_train,
                    xs_obs=xs_train, bsa_obs=bsa_train,
                    xs_sig=xs_sig_train, bsa_sig=bsa_sig_train,
                )

                loss_reg = tf.constant(0.0, dtype=_FLOATX)
                if LAMBDA_SMOOTH_IMH and float(LAMBDA_SMOOTH_IMH) > 0:
                    loss_reg += tf.constant(float(LAMBDA_SMOOTH_IMH), dtype=_FLOATX) * smoothness_penalty(imh)

                loss = loss_data + loss_reg

            grads = tape.gradient(loss, model.trainable_variables)
            gv = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            opt1.apply_gradients(gv)

            # ----- VAL -----
            if len(val_phi_idx):
                reh_dr_v, imh_v, _, _, _ = model(kin_tf, training=False)
                reh_eff_v = reh_dr_v

                xs_val = tf.gather(xs_obs_tf, val_phi_idx, axis=1)
                bsa_val = tf.gather(bsa_obs_tf, val_phi_idx, axis=1)
                xs_sig_val = tf.gather(xs_sig_tf, val_phi_idx, axis=1)
                bsa_sig_val = tf.gather(bsa_sig_tf, val_phi_idx, axis=1)

                val_data = compute_softchi2_loss(
                    bkm_op=bkm_op,
                    reh=reh_eff_v,
                    imh=imh_v,
                    t_bins=t_tf, xB_bins=xB_tf, Q2_bins=Q2_tf,
                    phi=phi_val,
                    xs_obs=xs_val, bsa_obs=bsa_val,
                    xs_sig=xs_sig_val, bsa_sig=bsa_sig_val,
                )
                val_loss = val_data + loss_reg  # same reg (close enough)
            else:
                val_data = loss_data
                val_loss = loss

            l = float(loss.numpy())
            ld = float(loss_data.numpy())
            lv = float(val_loss.numpy())
            lvd = float(val_data.numpy())

            # diag
            gate_mean = float(tf.reduce_mean(gate).numpy())
            corr_rms = 0.0  # by definition in hard stage
            c0v = float(c0.numpy())

            history["stage"].append("hard")
            history["epoch"].append(ep + 1)
            history["loss"].append(l)
            history["val_loss"].append(lv)
            history["loss_data"].append(ld)
            history["val_loss_data"].append(lvd)
            history["loss_reg"].append(float(loss_reg.numpy()))
            history["corr_rms"].append(corr_rms)
            history["gate_mean"].append(gate_mean)
            history["C0"].append(c0v)

            metric = lvd if len(val_phi_idx) else ld
            if np.isfinite(metric) and metric < best_val - 1e-12:
                best_val = metric
                best_weights = model.get_weights()
                bad = 0
            else:
                if (ep + 1) >= int(STAGE1_MIN_EPOCHS):
                    bad += 1

            if (ep == 0) or ((ep + 1) % int(PRINT_EVERY) == 0):
                print(
                    f"Replica {r+1:03d} | Stage1(HardDR) ep {ep+1:4d} "
                    f"| loss={l:.6g} val={lv:.6g} | val_data={lvd:.6g} "
                    f"| C0={c0v:+.4g}"
                )

            # Early stop stage1
            if (ep + 1) >= int(STAGE1_MIN_EPOCHS) and bad >= int(STAGE1_PATIENCE):
                break

            # Optional: if already good enough, you can stop stage1 early
            if (ep + 1) >= int(STAGE1_MIN_EPOCHS) and lvd <= float(RELEASE_TARGET_VAL_DATA):
                # This means Hard-DR is already good; we can proceed to stage2 (or skip it).
                break

        if best_weights is not None:
            model.set_weights(best_weights)

        # Decide whether to do stage2:
        # If HardDR already meets the target, we still MAY do stage2, but it should keep corr~0.
        do_stage2 = True
        # You can choose to skip stage2 when HardDR is excellent:
        # do_stage2 = (best_val > RELEASE_TARGET_VAL_DATA)

        # -------------------------
        # Stage 2: Release correction (gated soft-DR)
        # -------------------------
        if do_stage2:
            best_val2 = np.inf
            best_weights2 = model.get_weights()
            bad2 = 0

            # capture "release baseline" (alpha starts at 0)
            for ep2 in range(int(STAGE2_MAX_EPOCHS)):
                # alpha ramp
                alpha = ramp_alpha(ep2)

                phi_train = tf.gather(phi_tf, train_phi_idx, axis=0)
                phi_val = tf.gather(phi_tf, val_phi_idx, axis=0) if len(val_phi_idx) else None

                with tf.GradientTape() as tape:
                    reh_dr, imh, delta, gate, c0 = model(kin_tf, training=True)

                    # Effective correction with ramp alpha
                    corr = tf.constant(float(alpha), dtype=_FLOATX) * (1.0 - gate) * delta
                    reh_eff = reh_dr + corr

                    xs_train = tf.gather(xs_obs_tf, train_phi_idx, axis=1)
                    bsa_train = tf.gather(bsa_obs_tf, train_phi_idx, axis=1)
                    xs_sig_train = tf.gather(xs_sig_tf, train_phi_idx, axis=1)
                    bsa_sig_train = tf.gather(bsa_sig_tf, train_phi_idx, axis=1)

                    loss_data = compute_softchi2_loss(
                        bkm_op=bkm_op,
                        reh=reh_eff, imh=imh,
                        t_bins=t_tf, xB_bins=xB_tf, Q2_bins=Q2_tf,
                        phi=phi_train,
                        xs_obs=xs_train, bsa_obs=bsa_train,
                        xs_sig=xs_sig_train, bsa_sig=bsa_sig_train,
                    )

                    loss_reg = tf.constant(0.0, dtype=_FLOATX)

                    # smooth ImH always
                    if LAMBDA_SMOOTH_IMH and float(LAMBDA_SMOOTH_IMH) > 0:
                        loss_reg += tf.constant(float(LAMBDA_SMOOTH_IMH), dtype=_FLOATX) * smoothness_penalty(imh)

                    # soft-DR structure penalties (on effective correction!)
                    if LAMBDA_CORR_L2 and float(LAMBDA_CORR_L2) > 0:
                        loss_reg += tf.constant(float(LAMBDA_CORR_L2), dtype=_FLOATX) * tf.reduce_mean(tf.square(corr))

                    if LAMBDA_GATE_PRIOR and float(LAMBDA_GATE_PRIOR) > 0:
                        loss_reg += tf.constant(float(LAMBDA_GATE_PRIOR), dtype=_FLOATX) * tf.reduce_mean(tf.square(1.0 - gate))

                    if LAMBDA_DELTA_L2 and float(LAMBDA_DELTA_L2) > 0:
                        loss_reg += tf.constant(float(LAMBDA_DELTA_L2), dtype=_FLOATX) * tf.reduce_mean(tf.square(delta))

                    if LAMBDA_SMOOTH_DELTA and float(LAMBDA_SMOOTH_DELTA) > 0:
                        loss_reg += tf.constant(float(LAMBDA_SMOOTH_DELTA), dtype=_FLOATX) * smoothness_penalty(delta)

                    loss = loss_data + loss_reg

                grads = tape.gradient(loss, model.trainable_variables)
                gv = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
                opt2.apply_gradients(gv)

                # val
                if len(val_phi_idx):
                    reh_dr_v, imh_v, delta_v, gate_v, c0_v = model(kin_tf, training=False)
                    corr_v = tf.constant(float(alpha), dtype=_FLOATX) * (1.0 - gate_v) * delta_v
                    reh_eff_v = reh_dr_v + corr_v

                    xs_val = tf.gather(xs_obs_tf, val_phi_idx, axis=1)
                    bsa_val = tf.gather(bsa_obs_tf, val_phi_idx, axis=1)
                    xs_sig_val = tf.gather(xs_sig_tf, val_phi_idx, axis=1)
                    bsa_sig_val = tf.gather(bsa_sig_tf, val_phi_idx, axis=1)

                    val_data = compute_softchi2_loss(
                        bkm_op=bkm_op,
                        reh=reh_eff_v, imh=imh_v,
                        t_bins=t_tf, xB_bins=xB_tf, Q2_bins=Q2_tf,
                        phi=phi_val,
                        xs_obs=xs_val, bsa_obs=bsa_val,
                        xs_sig=xs_sig_val, bsa_sig=bsa_sig_val,
                    )
                    val_loss = val_data + loss_reg
                else:
                    val_data = loss_data
                    val_loss = loss

                l = float(loss.numpy())
                ld = float(loss_data.numpy())
                lv = float(val_loss.numpy())
                lvd = float(val_data.numpy())

                gate_mean = float(tf.reduce_mean(gate).numpy())
                corr_rms = float(tf.sqrt(tf.reduce_mean(tf.square(corr)) + 1e-30).numpy())
                c0v = float(c0.numpy())

                history["stage"].append("gated")
                history["epoch"].append(ep2 + 1)
                history["loss"].append(l)
                history["val_loss"].append(lv)
                history["loss_data"].append(ld)
                history["val_loss_data"].append(lvd)
                history["loss_reg"].append(float(loss_reg.numpy()))
                history["corr_rms"].append(corr_rms)
                history["gate_mean"].append(gate_mean)
                history["C0"].append(c0v)

                metric = lvd if len(val_phi_idx) else ld
                if np.isfinite(metric) and metric < best_val2 - 1e-12:
                    best_val2 = metric
                    best_weights2 = model.get_weights()
                    bad2 = 0
                else:
                    if (ep2 + 1) >= int(STAGE2_MIN_EPOCHS):
                        bad2 += 1

                if (ep2 == 0) or ((ep2 + 1) % int(PRINT_EVERY) == 0):
                    print(
                        f"Replica {r+1:03d} | Stage2(GATED) ep {ep2+1:4d} "
                        f"| alpha={alpha:.3f} | loss={l:.6g} val={lv:.6g} | val_data={lvd:.6g} "
                        f"| C0={c0v:+.4g} | gate_mean={gate_mean:.3f} | corr_rms={corr_rms:.3g}"
                    )

                if (ep2 + 1) >= int(STAGE2_MIN_EPOCHS) and bad2 >= int(STAGE2_PATIENCE):
                    break

            model.set_weights(best_weights2)

        # -------------------------
        # Save
        # -------------------------
        base = os.path.join(MODELS_DIR, f"replica_{r+1:03d}_{TAG}")
        model.save_weights(base + ".weights.h5")
        np.savez(
            base + "_meta.npz",
            xi_bins=xi_bins.astype(np.float32),
            xB_bins=xB_bins.astype(np.float32),
            t_bins=t_bins.astype(np.float32),
            Q2_bins=Q2_bins.astype(np.float32),
            phi_grid=phi_grid.astype(np.float32),
            K_used=(K_from_file if K_from_file is not None else build_pv_kernel_trapezoid(xi_bins)),
            config=np.array(
                dict(
                    STAGE1_LR=float(STAGE1_LR),
                    STAGE2_LR=float(STAGE2_LR),
                    RELEASE_TARGET_VAL_DATA=float(RELEASE_TARGET_VAL_DATA),
                    IMH_SCALE=float(IMH_SCALE),
                    DELTA_REH_SCALE=float(DELTA_REH_SCALE),
                    GATE_BIAS_INIT=float(GATE_BIAS_INIT),
                    LAMBDA_CORR_L2=float(LAMBDA_CORR_L2),
                    LAMBDA_GATE_PRIOR=float(LAMBDA_GATE_PRIOR),
                    LAMBDA_DELTA_L2=float(LAMBDA_DELTA_L2),
                    LAMBDA_SMOOTH_IMH=float(LAMBDA_SMOOTH_IMH),
                    LAMBDA_SMOOTH_DELTA=float(LAMBDA_SMOOTH_DELTA),
                    RAMP_EPOCHS=int(RAMP_EPOCHS),
                    FD_EPS=float(FD_EPS),
                ),
                dtype=object,
            ),
        )
        _save_json(os.path.join(HIST_DIR, f"history_replica_{r+1:03d}_{TAG}.json"), history)

        # Postfit residuals
        if PRINT_POSTFIT_RESIDUALS:
            reh_dr, imh, delta, gate, c0 = model(kin_tf, training=False)
            # use full alpha=1 for reporting "final" correction capability
            corr = (1.0 - gate) * delta
            reh_eff = reh_dr + corr

            xs_pred, bsa_pred = bkm_op(reh_eff, imh, t_tf, xB_tf, Q2_tf, phi_tf)
            xs_pred = xs_pred.numpy()
            bsa_pred = bsa_pred.numpy()

            max_dx = float(np.max(np.abs(xs_obs - xs_pred)))
            max_db = float(np.max(np.abs(bsa_obs - bsa_pred)))

            corr_rms = float(np.sqrt(np.mean(corr.numpy()**2)))
            print(
                f"Replica {r+1:03d} postfit: max|ΔXS|={max_dx:.3e}, max|ΔBSA|={max_db:.3e} "
                f"| C0={float(c0.numpy()):+.4g} | gate_mean={float(tf.reduce_mean(gate).numpy()):.3f} "
                f"| corr_rms(alpha=1)={corr_rms:.3g}"
            )

    print("\nSaved weights to:", MODELS_DIR)
    print("Saved histories to:", HIST_DIR)


if __name__ == "__main__":
    main()
