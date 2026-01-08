# Basic fixed-kinematics closure example (XS + BSA → ReH, ImH)

This folder contains a minimal *closure test* pipeline that demonstrates how to extract the
DVCS Compton Form Factor **H** at a **single fixed kinematic point** by fitting the full
azimuthal dependence of:

- unpolarized cross section **XS(ϕ)**
- beam spin asymmetry **BSA(ϕ)**

The pipeline is meant to be easy to run and easy to debug.

## What these scripts do

### 1) `closure_generate_dataset.py` — make pseudodata at fixed kinematics
Generates a dataset on a user-defined ϕ grid at fixed (xB, Q², t, beam energy).

**Truth model**
- Uses **Gepard / KM15** to compute the “truth” CFFs (H, E, H~, E~) at the chosen kinematics.
- CFFs should be independent of ϕ; the script records a std-over-ϕ sanity check.

**Forward model (observables)**
- Uses `bkm10_lib` to compute **XS(ϕ)** and **BSA(ϕ)** from those truth CFFs.

**Pseudo-measurements**
- Builds per-point uncertainties via: `sigma = ABS + REL * |y_true|`
- Optionally adds one noisy draw to create `y_central`; replicas are generated later during training.

**Outputs (written under `<OUT_DIR>/data`)**
- `dataset_<TAG>.npz`  (used by training/evaluation)
- `dataset_<TAG>.csv`  (human-readable)
- `truth_<TAG>.json`   (truth for evaluation/plotting only; training does NOT need it)

> Configure: `OUT_DIR`, `TAG`, kinematics, ϕ grid, and uncertainty knobs in the USER CONFIG block.

---

### 2) `closure_train_basic.py` — train replica ensemble to infer ReH and ImH
Trains an ensemble of replica fits that infer **scalar** `(ReH, ImH)` at fixed kinematics by
fitting both XS(ϕ) and BSA(ϕ) simultaneously.

**Key design**
- The model outputs ReH and ImH (as a function of kinematics inputs), but the loss enforces
  that the inferred CFFs are effectively *constant over ϕ* by averaging the outputs over the ϕ grid.
- The XS/BSA predictions come from `bkm10_lib`, wrapped in a TF custom op with finite-difference
  gradients (so we can backprop through the forward model).

**“Soft chi²” loss (important)**
This script is designed to behave well even if `XS_err = BSA_err = 0` in the dataset.
It builds *softened* per-point sigmas for weighting:
