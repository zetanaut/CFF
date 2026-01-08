#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
closure_train_basic.py  (Python 3.9+)

Here is a basic example of ReH and ImH extraction using xsec and bsa
----
Train an ensemble of replica fits to extract ReH and ImH at *fixed kinematics* by
simultaneously fitting:
  - unpolarized cross section XS(phi)
  - beam spin asymmetry BSA(phi)

Key improvement vs closure_train_replicas.py
--------------------------------------------
This version makes the loss well-conditioned even when the dataset uncertainties
are set to zero (XS_err=BSA_err=0). This allows you to start your closure test
at zero experimental error to ensure all is working well with perfect input info.

  sigma_soft = sqrt( sigma_data^2 + (rel_floor * scale)^2 + abs_floor^2 )

- If sigma_data is moderate, sigma_soft ≈ sigma_data (little change).
- If sigma_data is zero, sigma_soft reduces to a sensible default scale, so the
  optimization behaves similarly to a moderate-error case.

You can also make the loss *fully insensitive* to the provided errors by setting:
  USE_POINTWISE_SIGMAS = False
which makes sigma_soft purely determined by the floors/scales.

Inputs (from closure_generate_dataset.py)
-----------------------------------------
<VERSION_DIR>/data/dataset_<TAG>.npz containing:
  x:         (N,4) [t, xB, Q2, phi]  phi in radians
  y_central: (N,2) [XS, BSA]         central values
  y_sigma:   (N,2) [XS_err, BSA_err] per-point 1σ uncertainties (can be zero)

Outputs
-------
<VERSION_DIR>/replicas/replica_XXX_<TAG>.keras
<VERSION_DIR>/histories/history_replica_XXX_<TAG>.json

Run
---
python closure_train_replicas.py
"""

import json
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*")

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs


# =========================
# USER CONFIG (edit here)
# =========================
# careful with naming, the TAG here will be used by evaluation script to find outputs 

VERSION_DIR = "output"
TAG = "v_1"
DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")

MODELS_DIR = os.path.join(VERSION_DIR, "replicas")
HIST_DIR = os.path.join(VERSION_DIR, "histories")

# Replica settings
N_REPLICAS = 50
REPLICA_SEED = 2222  # controls replica noise draws

# Train/val split (random over points). For strict closure debugging, use 1.0.
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42

# Optimizer/training hyperparams
EPOCHS = 2500
PATIENCE = 250
LEARNING_RATE = 3e-3

# Batch size: 0 => full batch
BATCH_SIZE = 0

# Finite-difference step for custom gradient
FD_EPS = 5e-3

# Gradient clipping (recommended for stability)
ADAM_CLIPNORM = 5.0  # set 0 to disable

# Residual clipping in units of sigma_soft (post-division)
RATIO_CLIP = 1e4  # set None/0 to disable

# ---- Soft-chi2 normalization knobs ----
# If True: use provided per-point sigmas (if any) but softened by floors.
# If False: ignore provided sigmas entirely (loss becomes "scale-normalized" MSE).
USE_POINTWISE_SIGMAS = True

# Floors for XS: sigma_soft_xs includes (SOFT_XS_REL * XS_scale) and SOFT_XS_ABS
SOFT_XS_REL = 0.02   # behaves like a ~2% default relative uncertainty when sigma_data=0
SOFT_XS_ABS = 0.0    # absolute XS floor (in XS units)

# Floors for BSA: sigma_soft_bsa includes (SOFT_BSA_REL * BSA_scale) and SOFT_BSA_ABS
SOFT_BSA_REL = 0.0
SOFT_BSA_ABS = 0.01  # absolute asymmetry floor

# Diagnostics
PRINT_POSTFIT_RESIDUALS = True  # per replica: max|ΔXS|, max|ΔBSA| over ALL points

# Fixed kinematics (MUST match generation)
BEAM_ENERGY = 5.75
Q2 = 1.82
XB = 0.34
T = -0.17

# bkm10_lib settings (MUST match generation)
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


def _make_forward_np():
    """Return forward(reh, imh, phi_array)->(xs,bsa) using bkm10_lib."""
    kin = BKM10Inputs(
        lab_kinematics_k=float(BEAM_ENERGY),
        squared_Q_momentum_transfer=float(Q2),
        x_Bjorken=float(XB),
        squared_hadronic_momentum_transfer_t=float(T),
    )

    def forward(reh: float, imh: float, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cff_h = complex(float(reh), float(imh))
        cfg = {
            "kinematics": kin,
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
        xsecs = DifferentialCrossSection(configuration=cfg, verbose=False, debugging=False)
        xs = np.asarray(xsecs.compute_cross_section(phi).real, dtype=np.float32)
        bsa = np.asarray(xsecs.compute_bsa(phi).real, dtype=np.float32)
        return xs, bsa

    return forward


def make_bkm10_custom_op(fd_eps: float):
    forward_np = _make_forward_np()

    def _forward_np(reh_np, imh_np, phi_np):
        reh = float(np.asarray(reh_np).item())
        imh = float(np.asarray(imh_np).item())
        phi = np.asarray(phi_np, dtype=float)
        xs, bsa = forward_np(reh, imh, phi)
        return xs, bsa

    @tf.custom_gradient
    def op(reh_tf: tf.Tensor, imh_tf: tf.Tensor, phi_tf: tf.Tensor):
        xs_tf, bsa_tf = tf.numpy_function(
            func=_forward_np,
            inp=[reh_tf, imh_tf, phi_tf],
            Tout=[_FLOATX, _FLOATX],
        )
        xs_tf.set_shape(phi_tf.shape)
        bsa_tf.set_shape(phi_tf.shape)

        def grad(dxs: tf.Tensor, dbsa: tf.Tensor):
            eps = tf.constant(float(fd_eps), dtype=_FLOATX)

            # d/dReH
            xs_p, bsa_p = tf.numpy_function(_forward_np, [reh_tf + eps, imh_tf, phi_tf], [_FLOATX, _FLOATX])
            xs_m, bsa_m = tf.numpy_function(_forward_np, [reh_tf - eps, imh_tf, phi_tf], [_FLOATX, _FLOATX])
            xs_p.set_shape(phi_tf.shape); xs_m.set_shape(phi_tf.shape)
            bsa_p.set_shape(phi_tf.shape); bsa_m.set_shape(phi_tf.shape)
            d_xs_d_reh = (xs_p - xs_m) / (2.0 * eps)
            d_bsa_d_reh = (bsa_p - bsa_m) / (2.0 * eps)

            # d/dImH
            xs_p2, bsa_p2 = tf.numpy_function(_forward_np, [reh_tf, imh_tf + eps, phi_tf], [_FLOATX, _FLOATX])
            xs_m2, bsa_m2 = tf.numpy_function(_forward_np, [reh_tf, imh_tf - eps, phi_tf], [_FLOATX, _FLOATX])
            xs_p2.set_shape(phi_tf.shape); xs_m2.set_shape(phi_tf.shape)
            bsa_p2.set_shape(phi_tf.shape); bsa_m2.set_shape(phi_tf.shape)
            d_xs_d_imh = (xs_p2 - xs_m2) / (2.0 * eps)
            d_bsa_d_imh = (bsa_p2 - bsa_m2) / (2.0 * eps)

            # Chain rule
            g_reh = tf.reduce_sum(dxs * d_xs_d_reh + dbsa * d_bsa_d_reh)
            g_imh = tf.reduce_sum(dxs * d_xs_d_imh + dbsa * d_bsa_d_imh)

            return g_reh, g_imh, None

        return (xs_tf, bsa_tf), grad

    return op


class SoftChi2Loss(tf.keras.losses.Loss):
    """
    y_true: (N,4) => [XS_obs, BSA_obs, XS_sigma_soft, BSA_sigma_soft]
    y_pred: (N,6) => [ReH_pred, ImH_pred, t, xB, Q2, phi]
    """

    def __init__(self, fd_eps: float = 5e-3, ratio_clip: Optional[float] = None, name: str = "soft_chi2"):
        super().__init__(name=name)
        self.bkm10_op = make_bkm10_custom_op(fd_eps=float(fd_eps))
        self.ratio_clip = float(ratio_clip) if (ratio_clip is not None and ratio_clip > 0) else None

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        xs_obs = y_true[:, 0]
        bsa_obs = y_true[:, 1]
        xs_sig = y_true[:, 2]
        bsa_sig = y_true[:, 3]

        reh_vec = y_pred[:, 0]
        imh_vec = y_pred[:, 1]
        phi = y_pred[:, 5]

        # enforce scalar CFFs (fixed-kinematics extraction)
        reh = tf.reduce_mean(reh_vec)
        imh = tf.reduce_mean(imh_vec)

        xs_pred, bsa_pred = self.bkm10_op(reh, imh, phi)

        # avoid NaNs if bkm10 returns non-finite values
        xs_pred = tf.where(tf.math.is_finite(xs_pred), xs_pred, tf.zeros_like(xs_pred) + 1e30)
        bsa_pred = tf.where(tf.math.is_finite(bsa_pred), bsa_pred, tf.zeros_like(bsa_pred) + 1e30)

        rx = (xs_obs - xs_pred) / xs_sig
        rb = (bsa_obs - bsa_pred) / bsa_sig

        if self.ratio_clip is not None:
            c = tf.constant(self.ratio_clip, dtype=_FLOATX)
            rx = tf.clip_by_value(rx, -c, c)
            rb = tf.clip_by_value(rb, -c, c)

        return 0.5 * (tf.reduce_mean(tf.square(rx)) + tf.reduce_mean(tf.square(rb)))


def cff_model(seed: Optional[int] = None) -> tf.keras.Model:
    init = tf.keras.initializers.RandomUniform(minval=-0.3, maxval=0.3, seed=seed)

    all_in = tf.keras.Input(shape=(4,), name="x")
    kin = tf.keras.layers.Lambda(lambda x: x[:, :3], name="kin")(all_in)  # (t, xB, Q2)

    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init)(kin)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init)(x)
    cffs = tf.keras.layers.Dense(2, activation="linear", kernel_initializer=init, name="cff")(x)

    out = tf.keras.layers.Concatenate(axis=1, name="out")([cffs, all_in])
    return tf.keras.Model(inputs=all_in, outputs=out)


def _soft_sigmas(y_central: np.ndarray, y_sigma: np.ndarray) -> np.ndarray:
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

    # For BSA, median(|BSA|) is ok, but fall back to 90th percentile if median is tiny
    abs_bsa = np.abs(bsa[np.isfinite(bsa)])
    if abs_bsa.size == 0:
        bsa_scale = 1.0
    else:
        bsa_scale = float(np.median(abs_bsa))
        if bsa_scale <= 1e-6:
            bsa_scale = float(np.percentile(abs_bsa, 90))
        bsa_scale = bsa_scale if bsa_scale > 0 else 1.0

    # If we want to ignore provided pointwise errors, set sigma_data=0
    if not bool(USE_POINTWISE_SIGMAS):
        xs_sig = np.zeros_like(xs_sig)
        bsa_sig = np.zeros_like(bsa_sig)

    xs_floor = (float(SOFT_XS_REL) * xs_scale)
    bsa_floor = (float(SOFT_BSA_REL) * bsa_scale)

    xs_soft = np.sqrt(xs_sig**2 + xs_floor**2 + float(SOFT_XS_ABS)**2)
    bsa_soft = np.sqrt(bsa_sig**2 + bsa_floor**2 + float(SOFT_BSA_ABS)**2)

    # Final safety (should never trigger)
    xs_soft = np.where(xs_soft > 0, xs_soft, 1.0)
    bsa_soft = np.where(bsa_soft > 0, bsa_soft, 1.0)

    return np.column_stack([xs_soft, bsa_soft]).astype(np.float32)


def main() -> None:
    _safe_mkdir(MODELS_DIR)
    _safe_mkdir(HIST_DIR)

    d = _load_npz(DATA_NPZ)
    X = d["x"].astype(np.float32)
    y_central = d["y_central"].astype(np.float32)  # (N,2)
    y_sigma = d["y_sigma"].astype(np.float32)      # (N,2) (sampling sigmas)

    N = X.shape[0]
    if X.shape[1] != 4 or y_central.shape != (N, 2) or y_sigma.shape != (N, 2):
        raise ValueError(f"Unexpected shapes: X={X.shape}, y_central={y_central.shape}, y_sigma={y_sigma.shape}")

    # Precompute sigma_soft (loss weights). Replica sampling still uses y_sigma.
    sigma_soft = _soft_sigmas(y_central, y_sigma)

    print("Loss sigma_soft medians (weighting only):")
    print("  XS sigma_soft median  = {:.6g}".format(float(np.median(sigma_soft[:, 0]))))
    print("  BSA sigma_soft median = {:.6g}".format(float(np.median(sigma_soft[:, 1]))))
    print("  USE_POINTWISE_SIGMAS  =", bool(USE_POINTWISE_SIGMAS))
    if float(np.max(y_sigma[:, 0])) == 0.0 and float(np.max(y_sigma[:, 1])) == 0.0:
        print("NOTE: dataset y_sigma is identically zero -> replicas are identical; any spread is optimizer/init only.")
        print("      This script still uses nonzero sigma_soft, so optimization behaves like a moderate-error fit.")

    # Split indices (fixed for all replicas)
    rng_split = np.random.default_rng(int(SPLIT_SEED))
    idx = np.arange(N)
    rng_split.shuffle(idx)
    n_train = int(np.floor(float(TRAIN_FRACTION) * N))
    n_train = max(1, min(N, n_train))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:] if n_train < N else np.array([], dtype=int)

    X_train = X[train_idx]
    X_val = X[val_idx]

    batch_size = int(BATCH_SIZE)
    if batch_size <= 0:
        batch_size = len(X_train)

    loss_fn = SoftChi2Loss(fd_eps=float(FD_EPS), ratio_clip=float(RATIO_CLIP) if RATIO_CLIP else None)

    rng_rep = np.random.default_rng(int(REPLICA_SEED))
    forward_np = _make_forward_np()

    for r in range(int(N_REPLICAS)):
        seed = int(REPLICA_SEED) + 1000 * (r + 1)

        # Replica sampling (uses requested y_sigma; can be zero)
        noise = rng_rep.normal(0.0, 1.0, size=y_central.shape).astype(np.float32)
        y_rep = y_central + noise * y_sigma

        # Pack y_true: [XS_obs, BSA_obs, XS_sigma_soft, BSA_sigma_soft]
        y_pack = np.column_stack([y_rep, sigma_soft]).astype(np.float32)

        y_train = y_pack[train_idx]
        y_val = y_pack[val_idx] if len(val_idx) > 0 else None

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = cff_model(seed=seed)

        if ADAM_CLIPNORM and float(ADAM_CLIPNORM) > 0:
            opt = tf.keras.optimizers.Adam(learning_rate=float(LEARNING_RATE), clipnorm=float(ADAM_CLIPNORM))
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=float(LEARNING_RATE))

        model.compile(optimizer=opt, loss=loss_fn, run_eagerly=False)

        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if (len(X_val) > 0) else "loss",
                patience=int(PATIENCE),
                restore_best_weights=True,
            ),
        ]

        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=int(EPOCHS),
            batch_size=int(batch_size),
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        model_path = os.path.join(MODELS_DIR, f"replica_{r+1:03d}_{TAG}.keras")
        model.save(model_path)

        hist_path = os.path.join(HIST_DIR, f"history_replica_{r+1:03d}_{TAG}.json")
        _save_json(hist_path, {k: [float(x) for x in v] for k, v in hist.history.items()})

        # Extracted point estimate (no truth comparison)
        yhat = model(tf.convert_to_tensor(X, dtype=tf.float32), training=False).numpy()
        reh_hat = float(np.mean(yhat[:, 0]))
        imh_hat = float(np.mean(yhat[:, 1]))

        msg = f"Replica {r+1:03d}: ReH = {reh_hat:.6g}, ImH = {imh_hat:.6g}"

        if PRINT_POSTFIT_RESIDUALS:
            phi_all = X[:, 3].astype(float)
            xs_pred, bsa_pred = forward_np(reh_hat, imh_hat, phi_all)
            xs_obs = y_rep[:, 0].astype(float)
            bsa_obs = y_rep[:, 1].astype(float)
            max_dx = float(np.max(np.abs(xs_obs - xs_pred)))
            max_db = float(np.max(np.abs(bsa_obs - bsa_pred)))
            msg += f" | max|ΔXS|={max_dx:.3e}, max|ΔBSA|={max_db:.3e}"

        print(msg)

    print("\nSaved models to:", MODELS_DIR)
    print("Saved histories to:", HIST_DIR)


if __name__ == "__main__":
    main()
