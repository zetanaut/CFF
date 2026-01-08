#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_film.py

Goal
----
Non-bias DR usage via "DR-as-proposal" + FiLM-conditioned gating.

Core idea:
  1) Stage 1: Train a pure data-driven ("free") model -> ReH_free(xi), ImH(xi).
     DR is NOT used anywhere in the prediction.

  2) Stage 2: Introduce a DR proposal ReH_DR(xi)=C0 + PV[ImH](xi) and blend:
        ReH_pred = ReH_free + alpha(xi)*(ReH_DR - ReH_free)
     where alpha(xi) in [0,1] comes from a FiLM-conditioned gate.

  3) Non-bias selection rule:
     Only accept Stage-2 model if it improves validation data loss by >= MIN_IMPROVEMENT.
     Otherwise revert to Stage-1 best weights.

This prevents DR from "pulling away" from what data prefer: DR influence is only kept
if it improves held-out likelihood.

Outputs
-------
<version>/replicas_film_dr_nobias/replica_XXX_<TAG>.weights.h5
<version>/replicas_film_dr_nobias/replica_XXX_<TAG>_meta.npz
<version>/histories_film_dr_nobias/history_replica_XXX_<TAG>.json
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
TAG = "v_1"  # dataset tag: expects <VERSION_DIR>/data/dataset_<TAG>.npz
DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")

MODELS_DIR = os.path.join(VERSION_DIR, "replicas_film_dr_nobias")
HIST_DIR = os.path.join(VERSION_DIR, "histories_film_dr_nobias")

N_REPLICAS = 50
REPLICA_SEED = 20250101

# Train/val split over phi points
TRAIN_FRACTION = 0.8
SPLIT_SEED = 1234

# Optimizers
STAGE1_LR = 3e-3
STAGE2_LR = 1e-3
ADAM_CLIPNORM = 5.0

# Early stopping
STAGE1_EPOCHS = 4000
STAGE1_MIN_EPOCHS = 300
STAGE1_PATIENCE = 600

STAGE2_EPOCHS = 2500
STAGE2_MIN_EPOCHS = 200
STAGE2_PATIENCE = 500

# Accept Stage2 only if it improves validation DATA loss by at least this amount
MIN_IMPROVEMENT = 5e-3  # increase if you want to be more conservative

# FD for bkm gradients
FD_EPS = 5e-3

# Residual clipping after division by sigma_soft
RATIO_CLIP = 1e4

# Soft-chi2 weighting (same philosophy as your earlier script)
USE_POINTWISE_SIGMAS = True
SOFT_XS_REL = 0.02
SOFT_XS_ABS = 0.0
SOFT_BSA_REL = 0.0
SOFT_BSA_ABS = 0.01

# Output parameterization
IMH_SCALE = 8.0
REH_SCALE = 6.0

# FiLM-gate behavior
GATE_WIDTH = 32
FILM_CTX_WIDTH = 16
FILM_GAMMA_SCALE = 0.05
FILM_BETA_SCALE = 0.05

# Gate initialization: alpha ~ sigmoid(-6)=0.0025 (strongly "free" by default)
ALPHA_BIAS_INIT = -6.0

# Regularization (Stage2 only)
LAMBDA_ALPHA = 2.0          # penalize alpha^2 -> keeps DR off unless helpful
LAMBDA_FILM = 0.2           # keep FiLM near identity (gamma~1, beta~0)
LAMBDA_ALPHA_SMOOTH = 1e-4  # optional smoothness penalty on alpha(xi)

# Smoothness always (helps identifiability of xi surface)
LAMBDA_SMOOTH_IMH = 1e-4
LAMBDA_SMOOTH_REH = 1e-4

PRINT_EVERY = 50
PRINT_POSTFIT_RESIDUALS = True

# bkm10 settings (must match generator)
BEAM_ENERGY = 5.75
USING_WW = True
TARGET_POLARIZATION = 0.0
LEPTON_BEAM_POLARIZATION = 0.0

# Fixed nuisance CFFs (must match generator)
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

def smoothness_penalty(v: tf.Tensor) -> tf.Tensor:
    if v.shape[0] is not None and v.shape[0] < 3:
        return tf.constant(0.0, dtype=_FLOATX)
    d2 = v[2:] - 2.0 * v[1:-1] + v[:-2]
    return tf.reduce_mean(tf.square(d2))


# -------------------------
# PV kernel / DR layer
# -------------------------
def build_pv_kernel_trapezoid(xi_nodes: np.ndarray) -> np.ndarray:
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
    def __init__(self, xi_nodes: np.ndarray, K_matrix: np.ndarray, name: str = "pv_dr"):
        super().__init__(name=name)
        xi = np.asarray(xi_nodes, dtype=np.float32).reshape(-1)
        self.B = int(xi.size)
        K_matrix = np.asarray(K_matrix, dtype=np.float32)
        if K_matrix.shape != (self.B, self.B):
            raise ValueError(f"K must be ({self.B},{self.B}), got {K_matrix.shape}")
        self.K = tf.Variable(K_matrix, trainable=False, dtype=_FLOATX, name="K")

    def call(self, imh: tf.Tensor, c0: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(self.K, imh) + c0

class TrainableScalar(tf.keras.layers.Layer):
    def __init__(self, init_value: float = 0.0, name: str = "C0"):
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


# -------------------------
# bkm10 vector op with FD custom gradient
# -------------------------
def _forward_np_single_bin(reh: float, imh: float, t: float, xB: float, Q2: float, phi: np.ndarray):
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
    def _forward_np(reh_np, imh_np, t_np, xB_np, Q2_np, phi_np):
        reh = np.asarray(reh_np, dtype=float).reshape(-1)
        imh = np.asarray(imh_np, dtype=float).reshape(-1)
        t = np.asarray(t_np, dtype=float).reshape(-1)
        xB = np.asarray(xB_np, dtype=float).reshape(-1)
        Q2 = np.asarray(Q2_np, dtype=float).reshape(-1)
        phi = np.asarray(phi_np, dtype=float).reshape(-1)

        xs = np.empty((B, phi.shape[0]), dtype=np.float32)
        bsa = np.empty((B, phi.shape[0]), dtype=np.float32)
        for b in range(B):
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

        for b in range(B):
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
            _forward_np, [reh_tf, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf], [_FLOATX, _FLOATX]
        )
        xs_tf.set_shape((B, None))
        bsa_tf.set_shape((B, None))

        def grad(dxs, dbsa):
            g_reh, g_imh = tf.numpy_function(
                _grads_np, [reh_tf, imh_tf, t_tf, xB_tf, Q2_tf, phi_tf, dxs, dbsa], [_FLOATX, _FLOATX]
            )
            g_reh.set_shape(reh_tf.shape)
            g_imh.set_shape(imh_tf.shape)
            return g_reh, g_imh, None, None, None, None

        return (xs_tf, bsa_tf), grad

    return op


# -------------------------
# Model: free CFFs + DR proposal + FiLM-conditioned alpha gate
# -------------------------
class FiLMGateDRModel(tf.keras.Model):
    """
    Returns:
      reh_free(B,), imh(B,), reh_dr(B,), alpha(B,), c0(scalar), film_g(B,W), film_b(B,W)

    Prediction used in Stage2:
      reh_pred = reh_free + alpha*(reh_dr - reh_free)
    """
    def __init__(self, xi_nodes: np.ndarray, K_matrix: np.ndarray, seed: Optional[int] = None):
        super().__init__(name="FiLMGateDRModel")

        init = tf.keras.initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=seed)

        self.pv = PVDispersionLayer(xi_nodes=xi_nodes, K_matrix=K_matrix, name="pv_dr")
        self.c0_layer = TrainableScalar(init_value=0.0, name="C0")

        # trunk
        self.tr1 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init, name="trunk_dense1")
        self.tr2 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init, name="trunk_dense2")

        # free heads
        self.h_imh = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init, name="imh_raw")
        self.h_reh = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=init, name="reh_free_raw")

        # gate hidden
        self.g1 = tf.keras.layers.Dense(GATE_WIDTH, activation="relu", kernel_initializer=init, name="gate_hidden")

        # FiLM context net
        self.ctx1 = tf.keras.layers.Dense(FILM_CTX_WIDTH, activation="relu", kernel_initializer=init, name="film_ctx")

        # FiLM params should start as identity: gamma=1, beta=0
        self.to_gamma = tf.keras.layers.Dense(
            GATE_WIDTH, activation="linear",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="film_gamma_raw"
        )
        self.to_beta = tf.keras.layers.Dense(
            GATE_WIDTH, activation="linear",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="film_beta_raw"
        )

        # alpha head starts near 0
        self.to_alpha = tf.keras.layers.Dense(
            1, activation="linear",
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Constant(float(ALPHA_BIAS_INIT)),
            name="alpha_logit"
        )

    def call(self, kin: tf.Tensor, training: bool = False):
        # kin: (B,3) = (t, xB, Q2)
        t = kin[:, 0]
        xB = kin[:, 1]
        Q2 = kin[:, 2]
        xi = xB / (2.0 - xB)

        h = self.tr1(kin, training=training)
        h = self.tr2(h, training=training)

        imh_raw = tf.squeeze(self.h_imh(h, training=training), axis=1)
        reh_raw = tf.squeeze(self.h_reh(h, training=training), axis=1)

        imh = tf.constant(float(IMH_SCALE), dtype=_FLOATX) * tf.tanh(imh_raw)
        reh_free = tf.constant(float(REH_SCALE), dtype=_FLOATX) * tf.tanh(reh_raw)

        c0_vec = self.c0_layer(kin)
        c0 = c0_vec[0]

        # DR proposal from the SAME imh (this is important: DR doesn't create a second ImH)
        reh_dr = self.pv(imh, c0)

        # gate hidden features
        gh = self.g1(h, training=training)

        # FiLM conditioning context (DR proposal enters ONLY here)
        ctx = tf.stack([xi, imh, reh_dr], axis=1)  # (B,3)
        ctxh = self.ctx1(ctx, training=training)

        gamma_raw = self.to_gamma(ctxh, training=training)  # (B,W)
        beta_raw = self.to_beta(ctxh, training=training)    # (B,W)

        gamma = 1.0 + tf.constant(float(FILM_GAMMA_SCALE), dtype=_FLOATX) * gamma_raw
        beta = tf.constant(float(FILM_BETA_SCALE), dtype=_FLOATX) * beta_raw

        gh_mod = gamma * gh + beta

        alpha_logit = tf.squeeze(self.to_alpha(gh_mod, training=training), axis=1)
        alpha = tf.sigmoid(alpha_logit)

        return reh_free, imh, reh_dr, alpha, c0, gamma_raw, beta_raw


# -------------------------
# Loss
# -------------------------
def softchi2_loss(
    bkm_op,
    reh: tf.Tensor, imh: tf.Tensor,
    t_bins: tf.Tensor, xB_bins: tf.Tensor, Q2_bins: tf.Tensor,
    phi: tf.Tensor,
    xs_obs: tf.Tensor, bsa_obs: tf.Tensor,
    xs_sig: tf.Tensor, bsa_sig: tf.Tensor
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
# Main
# -------------------------
def main() -> None:
    _safe_mkdir(MODELS_DIR)
    _safe_mkdir(HIST_DIR)

    d = _load_npz(DATA_NPZ)
    X = d["x"].astype(np.float32)
    y_central = d["y_central"].astype(np.float32)
    y_sigma = d["y_sigma"].astype(np.float32)

    sigma_soft = _soft_sigmas(y_central, y_sigma)
    bins_central = group_by_kinematics(X, y_central, sigma_soft)
    phi_grid = assert_common_phi_grid(bins_central)
    Nphi = len(phi_grid)

    # Sort bins by xi increasing
    keys_sorted = sorted(bins_central.keys(), key=lambda k: xi_from_xB(k[1]))
    B = len(keys_sorted)
    if B < 2:
        raise ValueError("Need multiple xB/xi bins for DR example.")

    t_bins = np.array([k[0] for k in keys_sorted], dtype=np.float32)
    xB_bins = np.array([k[1] for k in keys_sorted], dtype=np.float32)
    Q2_bins = np.array([k[2] for k in keys_sorted], dtype=np.float32)
    xi_bins = xi_from_xB(xB_bins).astype(np.float32)

    print(f"Bins: B={B}, xi range=[{xi_bins.min():.6g},{xi_bins.max():.6g}], Nphi={Nphi}")

    # PV kernel: prefer one from NPZ if available
    K = None
    for key in ("K", "K_used", "K_matrix", "pv_K"):
        if key in d.files:
            K = d[key].astype(np.float32)
            print(f"Loaded K from NPZ key '{key}' shape={K.shape}")
            break
    if K is None:
        K = build_pv_kernel_trapezoid(xi_bins)
        print("Built K from xi grid (trapezoid PV).")

    # Pack sigmas (B,Nphi)
    xs_sig = np.stack([bins_central[k]["xs_sig"] for k in keys_sorted], axis=0).astype(np.float32)
    bsa_sig = np.stack([bins_central[k]["bsa_sig"] for k in keys_sorted], axis=0).astype(np.float32)

    # Phi split (shared)
    rng_split = np.random.default_rng(int(SPLIT_SEED))
    idx = np.arange(Nphi)
    rng_split.shuffle(idx)
    n_train = max(1, int(np.floor(float(TRAIN_FRACTION) * Nphi)))
    train_phi_idx = np.sort(idx[:n_train])
    val_phi_idx = np.sort(idx[n_train:]) if n_train < Nphi else np.array([], dtype=int)
    print(f"Phi split: train={len(train_phi_idx)}, val={len(val_phi_idx)}")

    # TF constants
    phi_tf = tf.constant(phi_grid, dtype=_FLOATX)
    t_tf = tf.constant(t_bins, dtype=_FLOATX)
    xB_tf = tf.constant(xB_bins, dtype=_FLOATX)
    Q2_tf = tf.constant(Q2_bins, dtype=_FLOATX)

    xs_sig_tf = tf.constant(xs_sig, dtype=_FLOATX)
    bsa_sig_tf = tf.constant(bsa_sig, dtype=_FLOATX)

    kin_bins = np.stack([t_bins, xB_bins, Q2_bins], axis=1).astype(np.float32)  # (B,3)
    kin_tf = tf.constant(kin_bins, dtype=_FLOATX)

    bkm_op = make_bkm_vector_op(B=B, fd_eps=float(FD_EPS))

    rng_rep = np.random.default_rng(int(REPLICA_SEED))
    Adam = getattr(tf.keras.optimizers, "legacy", tf.keras.optimizers).Adam

    for r in range(int(N_REPLICAS)):
        seed = int(REPLICA_SEED) + 1000 * (r + 1)
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Replica pseudo-data draw
        noise = rng_rep.normal(0.0, 1.0, size=y_central.shape).astype(np.float32)
        y_rep_pointwise = y_central + noise * y_sigma

        bins_rep = group_by_kinematics(X, y_rep_pointwise, sigma_soft)
        xs_obs = np.stack([bins_rep[k]["xs"] for k in keys_sorted], axis=0).astype(np.float32)
        bsa_obs = np.stack([bins_rep[k]["bsa"] for k in keys_sorted], axis=0).astype(np.float32)

        xs_obs_tf = tf.constant(xs_obs, dtype=_FLOATX)
        bsa_obs_tf = tf.constant(bsa_obs, dtype=_FLOATX)

        # Build model
        model = FiLMGateDRModel(xi_nodes=xi_bins, K_matrix=K, seed=seed)
        _ = model(kin_tf, training=False)  # build variables

        opt1 = Adam(learning_rate=float(STAGE1_LR), clipnorm=float(ADAM_CLIPNORM) if ADAM_CLIPNORM else None)
        opt2 = Adam(learning_rate=float(STAGE2_LR), clipnorm=float(ADAM_CLIPNORM) if ADAM_CLIPNORM else None)

        hist = {
            "stage": [], "epoch": [],
            "loss": [], "val_loss": [],
            "loss_data": [], "val_data": [],
            "alpha_mean": [], "C0": [],
        }

        # ----- Stage 1: FREE ONLY -----
        best_val1 = np.inf
        best_w1 = None
        bad = 0

        for ep in range(int(STAGE1_EPOCHS)):
            phi_tr = tf.gather(phi_tf, train_phi_idx, axis=0)
            xs_tr = tf.gather(xs_obs_tf, train_phi_idx, axis=1)
            bsa_tr = tf.gather(bsa_obs_tf, train_phi_idx, axis=1)
            xs_sig_tr = tf.gather(xs_sig_tf, train_phi_idx, axis=1)
            bsa_sig_tr = tf.gather(bsa_sig_tf, train_phi_idx, axis=1)

            with tf.GradientTape() as tape:
                reh_free, imh, reh_dr, alpha, c0, g_raw, b_raw = model(kin_tf, training=True)

                # Stage1 prediction: pure free (NO DR influence)
                reh_pred = reh_free

                loss_data = softchi2_loss(
                    bkm_op, reh_pred, imh, t_tf, xB_tf, Q2_tf, phi_tr, xs_tr, bsa_tr, xs_sig_tr, bsa_sig_tr
                )

                loss_reg = tf.constant(0.0, dtype=_FLOATX)
                if LAMBDA_SMOOTH_IMH and float(LAMBDA_SMOOTH_IMH) > 0:
                    loss_reg += tf.constant(float(LAMBDA_SMOOTH_IMH), dtype=_FLOATX) * smoothness_penalty(imh)
                if LAMBDA_SMOOTH_REH and float(LAMBDA_SMOOTH_REH) > 0:
                    loss_reg += tf.constant(float(LAMBDA_SMOOTH_REH), dtype=_FLOATX) * smoothness_penalty(reh_free)

                loss = loss_data + loss_reg

            grads = tape.gradient(loss, model.trainable_variables)
            gv = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            opt1.apply_gradients(gv)

            # validation (data only)
            if len(val_phi_idx):
                phi_va = tf.gather(phi_tf, val_phi_idx, axis=0)
                xs_va = tf.gather(xs_obs_tf, val_phi_idx, axis=1)
                bsa_va = tf.gather(bsa_obs_tf, val_phi_idx, axis=1)
                xs_sig_va = tf.gather(xs_sig_tf, val_phi_idx, axis=1)
                bsa_sig_va = tf.gather(bsa_sig_tf, val_phi_idx, axis=1)

                reh_free_v, imh_v, _, _, _, _, _ = model(kin_tf, training=False)
                val_data = softchi2_loss(
                    bkm_op, reh_free_v, imh_v, t_tf, xB_tf, Q2_tf, phi_va, xs_va, bsa_va, xs_sig_va, bsa_sig_va
                )
                val_loss = val_data
            else:
                val_data = loss_data
                val_loss = loss

            ld = float(loss_data.numpy())
            lvd = float(val_data.numpy())
            c0v = float(c0.numpy())
            a_mean = float(tf.reduce_mean(alpha).numpy())

            hist["stage"].append("free")
            hist["epoch"].append(ep + 1)
            hist["loss"].append(float(loss.numpy()))
            hist["val_loss"].append(float(val_loss.numpy()))
            hist["loss_data"].append(ld)
            hist["val_data"].append(lvd)
            hist["alpha_mean"].append(a_mean)
            hist["C0"].append(c0v)

            metric = lvd
            if np.isfinite(metric) and metric < best_val1 - 1e-12:
                best_val1 = metric
                best_w1 = model.get_weights()
                bad = 0
            else:
                if (ep + 1) >= int(STAGE1_MIN_EPOCHS):
                    bad += 1

            if (ep == 0) or ((ep + 1) % int(PRINT_EVERY) == 0):
                print(f"Replica {r+1:03d} | Stage1(FREE) ep {ep+1:4d} | data={ld:.6g} val={lvd:.6g} | C0={c0v:+.4g}")

            if (ep + 1) >= int(STAGE1_MIN_EPOCHS) and bad >= int(STAGE1_PATIENCE):
                break

        if best_w1 is not None:
            model.set_weights(best_w1)

        # ----- Stage 2: FiLM-gated DR proposal -----
        best_val2 = np.inf
        best_w2 = model.get_weights()
        bad2 = 0

        for ep2 in range(int(STAGE2_EPOCHS)):
            phi_tr = tf.gather(phi_tf, train_phi_idx, axis=0)
            xs_tr = tf.gather(xs_obs_tf, train_phi_idx, axis=1)
            bsa_tr = tf.gather(bsa_obs_tf, train_phi_idx, axis=1)
            xs_sig_tr = tf.gather(xs_sig_tf, train_phi_idx, axis=1)
            bsa_sig_tr = tf.gather(bsa_sig_tf, train_phi_idx, axis=1)

            with tf.GradientTape() as tape:
                reh_free, imh, reh_dr, alpha, c0, g_raw, b_raw = model(kin_tf, training=True)

                # Stage2 prediction: convex blend toward DR (proposal)
                reh_pred = reh_free + alpha * (reh_dr - reh_free)

                loss_data = softchi2_loss(
                    bkm_op, reh_pred, imh, t_tf, xB_tf, Q2_tf, phi_tr, xs_tr, bsa_tr, xs_sig_tr, bsa_sig_tr
                )

                # Regularization: penalize USING DR unless it helps
                loss_reg = tf.constant(0.0, dtype=_FLOATX)

                # keep surfaces smooth
                if LAMBDA_SMOOTH_IMH and float(LAMBDA_SMOOTH_IMH) > 0:
                    loss_reg += tf.constant(float(LAMBDA_SMOOTH_IMH), dtype=_FLOATX) * smoothness_penalty(imh)
                if LAMBDA_SMOOTH_REH and float(LAMBDA_SMOOTH_REH) > 0:
                    loss_reg += tf.constant(float(LAMBDA_SMOOTH_REH), dtype=_FLOATX) * smoothness_penalty(reh_free)

                # shrink alpha toward 0 (non-bias default)
                loss_reg += tf.constant(float(LAMBDA_ALPHA), dtype=_FLOATX) * tf.reduce_mean(tf.square(alpha))

                # keep FiLM near identity (gamma_raw ~ 0, beta_raw ~ 0)
                loss_reg += tf.constant(float(LAMBDA_FILM), dtype=_FLOATX) * (
                    tf.reduce_mean(tf.square(g_raw)) + tf.reduce_mean(tf.square(b_raw))
                )

                # optional alpha smoothness
                if LAMBDA_ALPHA_SMOOTH and float(LAMBDA_ALPHA_SMOOTH) > 0:
                    loss_reg += tf.constant(float(LAMBDA_ALPHA_SMOOTH), dtype=_FLOATX) * smoothness_penalty(alpha)

                loss = loss_data + loss_reg

            grads = tape.gradient(loss, model.trainable_variables)
            gv = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            opt2.apply_gradients(gv)

            # validation (data only) -> selection metric
            if len(val_phi_idx):
                phi_va = tf.gather(phi_tf, val_phi_idx, axis=0)
                xs_va = tf.gather(xs_obs_tf, val_phi_idx, axis=1)
                bsa_va = tf.gather(bsa_obs_tf, val_phi_idx, axis=1)
                xs_sig_va = tf.gather(xs_sig_tf, val_phi_idx, axis=1)
                bsa_sig_va = tf.gather(bsa_sig_tf, val_phi_idx, axis=1)

                reh_free_v, imh_v, reh_dr_v, alpha_v, c0_v, _, _ = model(kin_tf, training=False)
                reh_pred_v = reh_free_v + alpha_v * (reh_dr_v - reh_free_v)

                val_data = softchi2_loss(
                    bkm_op, reh_pred_v, imh_v, t_tf, xB_tf, Q2_tf, phi_va, xs_va, bsa_va, xs_sig_va, bsa_sig_va
                )
            else:
                val_data = loss_data

            ld = float(loss_data.numpy())
            lvd = float(val_data.numpy())
            c0v = float(c0.numpy())
            a_mean = float(tf.reduce_mean(alpha).numpy())

            hist["stage"].append("film_gate")
            hist["epoch"].append(ep2 + 1)
            hist["loss"].append(float(loss.numpy()))
            hist["val_loss"].append(float(val_data.numpy()))
            hist["loss_data"].append(ld)
            hist["val_data"].append(lvd)
            hist["alpha_mean"].append(a_mean)
            hist["C0"].append(c0v)

            metric = lvd
            if np.isfinite(metric) and metric < best_val2 - 1e-12:
                best_val2 = metric
                best_w2 = model.get_weights()
                bad2 = 0
            else:
                if (ep2 + 1) >= int(STAGE2_MIN_EPOCHS):
                    bad2 += 1

            if (ep2 == 0) or ((ep2 + 1) % int(PRINT_EVERY) == 0):
                print(f"Replica {r+1:03d} | Stage2(FiLM-gate) ep {ep2+1:4d} | data={ld:.6g} val={lvd:.6g} | alpha_mean={a_mean:.3f} | C0={c0v:+.4g}")

            if (ep2 + 1) >= int(STAGE2_MIN_EPOCHS) and bad2 >= int(STAGE2_PATIENCE):
                break

        # Selection rule: accept stage2 only if it improves val_data meaningfully
        use_stage2 = (best_val2 + float(MIN_IMPROVEMENT) < best_val1)
        if use_stage2:
            model.set_weights(best_w2)
            selected = "film_gate"
            best_val = best_val2
        else:
            model.set_weights(best_w1)
            selected = "free"
            best_val = best_val1

        # Save
        base = os.path.join(MODELS_DIR, f"replica_{r+1:03d}_{TAG}")
        model.save_weights(base + ".weights.h5")

        np.savez_compressed(
            base + "_meta.npz",
            xi_bins=xi_bins.astype(np.float32),
            xB_bins=xB_bins.astype(np.float32),
            t_bins=t_bins.astype(np.float32),
            Q2_bins=Q2_bins.astype(np.float32),
            phi_grid=phi_grid.astype(np.float32),
            K_used=K.astype(np.float32),
            config=np.array(
                dict(
                    TAG=TAG,
                    IMH_SCALE=float(IMH_SCALE),
                    REH_SCALE=float(REH_SCALE),
                    GATE_WIDTH=int(GATE_WIDTH),
                    FILM_CTX_WIDTH=int(FILM_CTX_WIDTH),
                    FILM_GAMMA_SCALE=float(FILM_GAMMA_SCALE),
                    FILM_BETA_SCALE=float(FILM_BETA_SCALE),
                    ALPHA_BIAS_INIT=float(ALPHA_BIAS_INIT),
                    LAMBDA_ALPHA=float(LAMBDA_ALPHA),
                    LAMBDA_FILM=float(LAMBDA_FILM),
                    MIN_IMPROVEMENT=float(MIN_IMPROVEMENT),
                    selected=selected,
                    best_val=float(best_val),
                    best_val_stage1=float(best_val1),
                    best_val_stage2=float(best_val2),
                ),
                dtype=object,
            ),
        )

        _save_json(os.path.join(HIST_DIR, f"history_replica_{r+1:03d}_{TAG}.json"), hist)

        if PRINT_POSTFIT_RESIDUALS:
            reh_free, imh, reh_dr, alpha, c0, _, _ = model(kin_tf, training=False)
            reh_pred = reh_free if (selected == "free") else (reh_free + alpha * (reh_dr - reh_free))

            # full phi residuals
            xs_pred, bsa_pred = bkm_op(reh_pred, imh, t_tf, xB_tf, Q2_tf, phi_tf)
            xs_pred = xs_pred.numpy()
            bsa_pred = bsa_pred.numpy()

            max_dx = float(np.max(np.abs(xs_obs - xs_pred)))
            max_db = float(np.max(np.abs(bsa_obs - bsa_pred)))
            print(f"Replica {r+1:03d} selected={selected} | best_val={best_val:.6g} | max|ΔXS|={max_dx:.3e} max|ΔBSA|={max_db:.3e}")

    print("\nSaved weights to:", MODELS_DIR)
    print("Saved histories to:", HIST_DIR)


if __name__ == "__main__":
    main()
