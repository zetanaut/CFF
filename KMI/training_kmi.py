#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_kmi.py  (Python 3.9+)

Global KMI training WITHOUT DR, fixed to avoid IndexedSlices gradient issues.

Main fixes vs earlier version:
  - Uses dense masks (train_mask, val_mask) instead of tf.gather for loss computation.
    This prevents gradients wrt the bkm_op outputs from becoming IndexedSlices.
  - Also densifies dxs/dbsa inside the custom-gradient as an extra safety net.

Model:
  one shared DNN: (t, xB, Q2) -> (ReH, ImH) per kinematic BIN.
  phi is NEVER an input to the DNN (prevents phi leakage).
  XS/BSA computed by bkm10_lib using nuisance CFFs per bin from the dataset.

Outputs:
  <VERSION_DIR>/replicas_kmi_no_dr/replica_XXX_<TAG>.weights.h5
  <VERSION_DIR>/replicas_kmi_no_dr/replica_XXX_<TAG>_meta.npz
  <VERSION_DIR>/histories_kmi_no_dr/history_replica_XXX_<TAG>.json
"""

import json
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

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
TAG = "v_1"
DATA_NPZ = os.path.join(VERSION_DIR, "data", f"dataset_{TAG}.npz")

MODELS_DIR = os.path.join(VERSION_DIR, "replicas_kmi_no_dr")
HIST_DIR = os.path.join(VERSION_DIR, "histories_kmi_no_dr")

N_REPLICAS = 50
REPLICA_SEED = 20250101

# Hold out whole kinematic bins for validation
N_VAL_BINS = 2
BIN_SPLIT_SEED = 1234

# Training hyperparams
EPOCHS = 3000
MIN_EPOCHS = 200
PATIENCE = 500

LR = 3e-3
ADAM_CLIPNORM = 5.0

# Finite difference step for BKM gradients
FD_EPS = 5e-3

# Soft-chi2 sigma floors
USE_POINTWISE_SIGMAS = True
SOFT_XS_REL = 0.02
SOFT_XS_ABS = 0.0
SOFT_BSA_REL = 0.0
SOFT_BSA_ABS = 0.01

# Output bounding (prevents BKM NaNs)
REH_SCALE = 8.0
IMH_SCALE = 8.0

# L2 regularization on network weights
L2_WEIGHT = 1e-4

# Diagnostics
PRINT_EVERY = 50
PRINT_POSTFIT_RESIDUALS = True

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
        raise FileNotFoundError(f"NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    required = {
        "x", "bin_id", "y_central", "y_sigma",
        "t_bins", "xB_bins", "Q2_bins",
        "nuisance_cff_E", "nuisance_cff_Ht", "nuisance_cff_Et",
    }
    missing = required - set(d.files)
    if missing:
        raise KeyError(f"NPZ missing keys: {missing}. Found: {d.files}")
    return d


def _soft_sigmas(y_central: np.ndarray, y_sigma: np.ndarray) -> np.ndarray:
    """
    sigma_soft = sqrt(sigma_data^2 + (rel_floor*scale)^2 + abs_floor^2)
    """
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


def xi_from_xB(xB: np.ndarray) -> np.ndarray:
    xB = np.asarray(xB, dtype=float)
    return xB / (2.0 - xB)


def build_features(t: np.ndarray, xB: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """
    Input features per BIN (not per point).
    """
    xi = xi_from_xB(xB)
    return np.column_stack([t, xB, np.log(Q2), xi]).astype(np.float32)


def make_cff_model(input_dim: int, feat_mean: np.ndarray, feat_std: np.ndarray, seed: Optional[int] = None) -> tf.keras.Model:
    """
    DNN: (t,xB,logQ2,xi) -> (ReH, ImH) with tanh bounding to keep BKM stable.
    """
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    l2 = tf.keras.regularizers.l2(float(L2_WEIGHT)) if (L2_WEIGHT and float(L2_WEIGHT) > 0) else None

    inp = tf.keras.Input(shape=(input_dim,), name="features")

    mean_tf = tf.constant(feat_mean.reshape(1, -1), dtype=_FLOATX)
    std_tf = tf.constant(feat_std.reshape(1, -1), dtype=_FLOATX)
    x = tf.keras.layers.Lambda(lambda z: (z - mean_tf) / std_tf, name="standardize")(inp)

    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init, kernel_regularizer=l2)(x)

    raw = tf.keras.layers.Dense(2, activation="linear", kernel_initializer=init, name="raw")(x)
    reh = tf.keras.layers.Lambda(lambda r: float(REH_SCALE) * tf.tanh(r), name="ReH")(raw[:, 0])
    imh = tf.keras.layers.Lambda(lambda r: float(IMH_SCALE) * tf.tanh(r), name="ImH")(raw[:, 1])

    out = tf.keras.layers.Concatenate(name="cff")([reh[:, None], imh[:, None]])
    return tf.keras.Model(inputs=inp, outputs=out)


def make_bkm_flat_op(
    t_bins: np.ndarray,
    xB_bins: np.ndarray,
    Q2_bins: np.ndarray,
    phi_bins: List[np.ndarray],
    cffE_bins: np.ndarray,
    cffHt_bins: np.ndarray,
    cffEt_bins: np.ndarray,
    beam_energy: float,
    using_ww: bool,
    target_pol: float,
    beam_pol: float,
    fd_eps: float,
):
    """
    Custom TF op:
      inputs:  reh_bins(B,), imh_bins(B,)
      outputs: xs_flat(N,), bsa_flat(N,)
    where N = sum_i len(phi_bins[i])

    Gradients computed by finite differences per BIN.
    """
    B = len(phi_bins)
    counts = np.array([len(p) for p in phi_bins], dtype=int)
    offsets = np.zeros(B, dtype=int)
    offsets[1:] = np.cumsum(counts[:-1])
    N = int(np.sum(counts))

    def _forward_bin(reh: float, imh: float, i: int) -> Tuple[np.ndarray, np.ndarray]:
        cfg = {
            "kinematics": BKM10Inputs(
                lab_kinematics_k=float(beam_energy),
                squared_Q_momentum_transfer=float(Q2_bins[i]),
                x_Bjorken=float(xB_bins[i]),
                squared_hadronic_momentum_transfer_t=float(t_bins[i]),
            ),
            "cff_inputs": CFFInputs(
                compton_form_factor_h=complex(float(reh), float(imh)),
                compton_form_factor_e=complex(cffE_bins[i].real, cffE_bins[i].imag),
                compton_form_factor_h_tilde=complex(cffHt_bins[i].real, cffHt_bins[i].imag),
                compton_form_factor_e_tilde=complex(cffEt_bins[i].real, cffEt_bins[i].imag),
            ),
            "target_polarization": float(target_pol),
            "lepton_beam_polarization": float(beam_pol),
            "using_ww": bool(using_ww),
        }
        xsecs = DifferentialCrossSection(configuration=cfg, verbose=False, debugging=False)
        phi = phi_bins[i].astype(float)
        xs = np.asarray(xsecs.compute_cross_section(phi).real, dtype=np.float32)
        bsa = np.asarray(xsecs.compute_bsa(phi).real, dtype=np.float32)
        return xs, bsa

    def _forward_np(reh_np, imh_np):
        reh = np.asarray(reh_np, dtype=float).reshape(-1)
        imh = np.asarray(imh_np, dtype=float).reshape(-1)

        xs_flat = np.empty((N,), dtype=np.float32)
        bsa_flat = np.empty((N,), dtype=np.float32)

        for i in range(B):
            xs_i, bsa_i = _forward_bin(reh[i], imh[i], i)
            s = offsets[i]
            e = s + counts[i]
            xs_flat[s:e] = xs_i
            bsa_flat[s:e] = bsa_i

        return xs_flat, bsa_flat

    def _grads_np(reh_np, imh_np, dxs_np, dbsa_np):
        reh = np.asarray(reh_np, dtype=float).reshape(-1)
        imh = np.asarray(imh_np, dtype=float).reshape(-1)

        # dxs_np/dbsa_np MUST be dense arrays here (we densify in TF grad)
        dxs = np.asarray(dxs_np, dtype=float).reshape(-1)
        dbsa = np.asarray(dbsa_np, dtype=float).reshape(-1)

        eps = float(fd_eps)
        g_reh = np.zeros((B,), dtype=np.float32)
        g_imh = np.zeros((B,), dtype=np.float32)

        for i in range(B):
            s = offsets[i]
            e = s + counts[i]
            dxs_i = dxs[s:e]
            dbsa_i = dbsa[s:e]

            xs_p, bsa_p = _forward_bin(reh[i] + eps, imh[i], i)
            xs_m, bsa_m = _forward_bin(reh[i] - eps, imh[i], i)
            d_xs_d_reh = (xs_p - xs_m) / (2.0 * eps)
            d_bsa_d_reh = (bsa_p - bsa_m) / (2.0 * eps)

            xs_p2, bsa_p2 = _forward_bin(reh[i], imh[i] + eps, i)
            xs_m2, bsa_m2 = _forward_bin(reh[i], imh[i] - eps, i)
            d_xs_d_imh = (xs_p2 - xs_m2) / (2.0 * eps)
            d_bsa_d_imh = (bsa_p2 - bsa_m2) / (2.0 * eps)

            g_reh[i] = np.sum(dxs_i * d_xs_d_reh + dbsa_i * d_bsa_d_reh).astype(np.float32)
            g_imh[i] = np.sum(dxs_i * d_xs_d_imh + dbsa_i * d_bsa_d_imh).astype(np.float32)

        return g_reh, g_imh

    @tf.custom_gradient
    def op(reh_tf: tf.Tensor, imh_tf: tf.Tensor):
        xs_tf, bsa_tf = tf.numpy_function(_forward_np, [reh_tf, imh_tf], [_FLOATX, _FLOATX])
        xs_tf.set_shape((N,))
        bsa_tf.set_shape((N,))

        def grad(dxs, dbsa):
            # IMPORTANT: dxs/dbsa can be IndexedSlices if upstream used gather.
            # We *force* them to be dense tensors here.
            dxs_dense = tf.convert_to_tensor(dxs)
            dbsa_dense = tf.convert_to_tensor(dbsa)

            dxs_dense = tf.reshape(dxs_dense, (-1,))
            dbsa_dense = tf.reshape(dbsa_dense, (-1,))

            g_reh, g_imh = tf.numpy_function(
                _grads_np, [reh_tf, imh_tf, dxs_dense, dbsa_dense], [_FLOATX, _FLOATX]
            )
            g_reh.set_shape(reh_tf.shape)
            g_imh.set_shape(imh_tf.shape)
            return g_reh, g_imh

        return (xs_tf, bsa_tf), grad

    return op


def main() -> None:
    _safe_mkdir(MODELS_DIR)
    _safe_mkdir(HIST_DIR)

    d = _load_npz(DATA_NPZ)

    X = d["x"].astype(np.float32)                 # (N,4)
    bin_id = d["bin_id"].astype(np.int32)         # (N,)
    y_central = d["y_central"].astype(np.float32) # (N,2)
    y_sigma = d["y_sigma"].astype(np.float32)     # (N,2)

    beam_energy = float(d["beam_energy"]) if "beam_energy" in d.files else 5.75
    using_ww = bool(int(d["using_ww"])) if "using_ww" in d.files else True
    target_pol = float(d["target_polarization"]) if "target_polarization" in d.files else 0.0
    beam_pol = float(d["lepton_beam_polarization"]) if "lepton_beam_polarization" in d.files else 0.0

    t_bins = d["t_bins"].astype(np.float32)
    xB_bins = d["xB_bins"].astype(np.float32)
    Q2_bins = d["Q2_bins"].astype(np.float32)

    cffE_bins = d["nuisance_cff_E"].astype(np.complex64)
    cffHt_bins = d["nuisance_cff_Ht"].astype(np.complex64)
    cffEt_bins = d["nuisance_cff_Et"].astype(np.complex64)

    B = int(np.max(bin_id)) + 1
    print(f"KMI bins: B={B} | beam={beam_energy}")

    # ---- group indices per bin, sort by phi
    idx_bins: List[np.ndarray] = []
    phi_bins: List[np.ndarray] = []
    for i in range(B):
        idx = np.where(bin_id == i)[0]
        if idx.size == 0:
            raise ValueError(f"Missing bin {i} in dataset.")
        order = np.argsort(X[idx, 3])
        idx = idx[order]
        idx_bins.append(idx)
        phi_bins.append(X[idx, 3].astype(np.float32))

    # ---- flatten in bin order (bin0 block, bin1 block, ...)
    flat_idx = np.concatenate(idx_bins, axis=0)
    X_flat = X[flat_idx]
    y_central_flat = y_central[flat_idx]
    y_sigma_flat = y_sigma[flat_idx]

    sigma_soft_flat = _soft_sigmas(y_central_flat, y_sigma_flat)

    N = X_flat.shape[0]
    print(f"Total points N={N} (ragged phi per bin, flattened).")

    # ---- build per-bin features
    feat_bins = build_features(t_bins.astype(float), xB_bins.astype(float), Q2_bins.astype(float))  # (B,4)
    feat_tf = tf.constant(feat_bins.astype(np.float32), dtype=_FLOATX)

    # ---- KMI bin split: hold out bins
    rng = np.random.default_rng(int(BIN_SPLIT_SEED))
    all_bins = np.arange(B, dtype=int)
    rng.shuffle(all_bins)
    val_bins = np.sort(all_bins[:int(N_VAL_BINS)])
    train_bins = np.sort(all_bins[int(N_VAL_BINS):])

    print("Train bins:", train_bins.tolist())
    print("Val bins:", val_bins.tolist())

    # ---- build dense masks in flattened point space
    counts = np.array([len(p) for p in phi_bins], dtype=int)
    offsets = np.zeros(B, dtype=int)
    offsets[1:] = np.cumsum(counts[:-1])

    train_mask = np.zeros((N,), dtype=np.float32)
    val_mask = np.zeros((N,), dtype=np.float32)

    for i in train_bins:
        s = offsets[i]; e = s + counts[i]
        train_mask[s:e] = 1.0
    for i in val_bins:
        s = offsets[i]; e = s + counts[i]
        val_mask[s:e] = 1.0

    train_count = float(np.sum(train_mask))
    val_count = float(np.sum(val_mask))
    if train_count <= 0 or val_count <= 0:
        raise RuntimeError("Bad split: train_count or val_count is zero.")

    train_mask_tf = tf.constant(train_mask, dtype=_FLOATX)
    val_mask_tf = tf.constant(val_mask, dtype=_FLOATX)
    train_count_tf = tf.constant(train_count, dtype=_FLOATX)
    val_count_tf = tf.constant(val_count, dtype=_FLOATX)

    xs_sig_tf = tf.constant(sigma_soft_flat[:, 0].astype(np.float32), dtype=_FLOATX)
    bsa_sig_tf = tf.constant(sigma_soft_flat[:, 1].astype(np.float32), dtype=_FLOATX)

    # ---- BKM op
    bkm_op = make_bkm_flat_op(
        t_bins=t_bins.astype(float),
        xB_bins=xB_bins.astype(float),
        Q2_bins=Q2_bins.astype(float),
        phi_bins=phi_bins,
        cffE_bins=cffE_bins,
        cffHt_bins=cffHt_bins,
        cffEt_bins=cffEt_bins,
        beam_energy=float(beam_energy),
        using_ww=bool(using_ww),
        target_pol=float(target_pol),
        beam_pol=float(beam_pol),
        fd_eps=float(FD_EPS),
    )

    # ---- replica RNG
    rng_rep = np.random.default_rng(int(REPLICA_SEED))

    # Use legacy Adam on M1/M2 if available
    Adam = getattr(tf.keras.optimizers, "legacy", tf.keras.optimizers).Adam

    for r in range(int(N_REPLICAS)):
        seed = int(REPLICA_SEED) + 1000 * (r + 1)
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Replica pseudo-data
        noise = rng_rep.normal(0.0, 1.0, size=y_central_flat.shape).astype(np.float32)
        y_rep = y_central_flat + noise * y_sigma_flat
        xs_obs_tf = tf.constant(y_rep[:, 0].astype(np.float32), dtype=_FLOATX)
        bsa_obs_tf = tf.constant(y_rep[:, 1].astype(np.float32), dtype=_FLOATX)

        # Standardization from TRAIN bins only
        feat_train = feat_bins[train_bins]
        feat_mean = np.mean(feat_train, axis=0)
        feat_std = np.std(feat_train, axis=0, ddof=0)
        feat_std = np.where(feat_std > 0, feat_std, 1.0)

        model = make_cff_model(input_dim=feat_bins.shape[1], feat_mean=feat_mean, feat_std=feat_std, seed=seed)
        opt = Adam(learning_rate=float(LR), clipnorm=float(ADAM_CLIPNORM) if ADAM_CLIPNORM else None)

        best_val = np.inf
        best_weights = None
        bad = 0

        history = {"epoch": [], "loss": [], "val_loss": []}

        big = tf.constant(1e30, dtype=_FLOATX)

        for ep in range(int(EPOCHS)):
            # ---- TRAIN STEP
            with tf.GradientTape() as tape:
                cff_bins = model(feat_tf, training=True)  # (B,2)
                reh_bins = cff_bins[:, 0]
                imh_bins = cff_bins[:, 1]

                xs_pred, bsa_pred = bkm_op(reh_bins, imh_bins)  # (N,), (N,)

                rx = (xs_obs_tf - xs_pred) / xs_sig_tf
                rb = (bsa_obs_tf - bsa_pred) / bsa_sig_tf

                # Mask to train points only (dense, avoids IndexedSlices)
                rx = rx * train_mask_tf
                rb = rb * train_mask_tf

                # Penalize non-finite residuals
                rx = tf.where(tf.math.is_finite(rx), rx, big * tf.ones_like(rx))
                rb = tf.where(tf.math.is_finite(rb), rb, big * tf.ones_like(rb))

                loss_x = tf.reduce_sum(tf.square(rx)) / train_count_tf
                loss_b = tf.reduce_sum(tf.square(rb)) / train_count_tf
                loss_data = 0.5 * (loss_x + loss_b)

                loss = loss_data + (tf.add_n(model.losses) if model.losses else 0.0)

            grads = tape.gradient(loss, model.trainable_variables)
            gv = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            opt.apply_gradients(gv)

            # ---- VAL LOSS (no grad)
            cff_bins_v = model(feat_tf, training=False)
            xs_pred_v, bsa_pred_v = bkm_op(cff_bins_v[:, 0], cff_bins_v[:, 1])

            rxv = ((xs_obs_tf - xs_pred_v) / xs_sig_tf) * val_mask_tf
            rbv = ((bsa_obs_tf - bsa_pred_v) / bsa_sig_tf) * val_mask_tf
            rxv = tf.where(tf.math.is_finite(rxv), rxv, big * tf.ones_like(rxv))
            rbv = tf.where(tf.math.is_finite(rbv), rbv, big * tf.ones_like(rbv))

            val_x = tf.reduce_sum(tf.square(rxv)) / val_count_tf
            val_b = tf.reduce_sum(tf.square(rbv)) / val_count_tf
            val_loss = 0.5 * (val_x + val_b)

            lv = float(val_loss.numpy())
            ltr = float(loss_data.numpy())

            history["epoch"].append(int(ep + 1))
            history["loss"].append(float(ltr))
            history["val_loss"].append(float(lv))

            if (ep == 0) or ((ep + 1) % int(PRINT_EVERY) == 0):
                reh0 = float(cff_bins_v[0, 0].numpy())
                imh0 = float(cff_bins_v[0, 1].numpy())
                print(f"Replica {r+1:03d} | ep {ep+1:4d} | loss={ltr:.6g} val={lv:.6g} | ReH(bin0)={reh0:+.3f} ImH(bin0)={imh0:+.3f}")

            # Early stopping
            if np.isfinite(lv) and lv < best_val - 1e-12:
                best_val = lv
                best_weights = model.get_weights()
                bad = 0
            else:
                if (ep + 1) >= int(MIN_EPOCHS):
                    bad += 1

            if (ep + 1) >= int(MIN_EPOCHS) and bad >= int(PATIENCE):
                break

        if best_weights is not None:
            model.set_weights(best_weights)

        # Save weights + meta
        base = os.path.join(MODELS_DIR, f"replica_{r+1:03d}_{TAG}")
        model.save_weights(base + ".weights.h5")

        np.savez_compressed(
            base + "_meta.npz",
            TAG=np.array(TAG, dtype=object),
            beam_energy=np.float32(beam_energy),
            using_ww=np.int32(1 if using_ww else 0),
            target_polarization=np.float32(target_pol),
            lepton_beam_polarization=np.float32(beam_pol),
            t_bins=t_bins,
            xB_bins=xB_bins,
            Q2_bins=Q2_bins,
            feat_mean=feat_mean.astype(np.float32),
            feat_std=feat_std.astype(np.float32),
            REH_SCALE=np.float32(REH_SCALE),
            IMH_SCALE=np.float32(IMH_SCALE),
            train_bins=train_bins.astype(np.int32),
            val_bins=val_bins.astype(np.int32),
            nuisance_cff_E=cffE_bins,
            nuisance_cff_Ht=cffHt_bins,
            nuisance_cff_Et=cffEt_bins,
            best_val=np.float32(best_val),
        )

        _save_json(os.path.join(HIST_DIR, f"history_replica_{r+1:03d}_{TAG}.json"), history)

        if PRINT_POSTFIT_RESIDUALS:
            cff_final = model(feat_tf, training=False).numpy().astype(float)
            xs_pred_f, bsa_pred_f = bkm_op(
                tf.constant(cff_final[:, 0], dtype=_FLOATX),
                tf.constant(cff_final[:, 1], dtype=_FLOATX),
            )
            xs_pred_f = xs_pred_f.numpy().astype(float)
            bsa_pred_f = bsa_pred_f.numpy().astype(float)
            xs_obs = y_rep[:, 0].astype(float)
            bsa_obs = y_rep[:, 1].astype(float)
            max_dx = float(np.max(np.abs(xs_obs - xs_pred_f)))
            max_db = float(np.max(np.abs(bsa_obs - bsa_pred_f)))
            print(f"Replica {r+1:03d} postfit: best_val={best_val:.6g} | max|ΔXS|={max_dx:.3e}, max|ΔBSA|={max_db:.3e}")

    print("\nSaved replicas to:", MODELS_DIR)
    print("Saved histories to:", HIST_DIR)


if __name__ == "__main__":
    main()
