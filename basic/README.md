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
sigma_soft = sqrt( sigma_data^2 + (rel_floor * scale)^2 + abs_floor^2 )
So when the dataset has zero errors, the optimization still behaves like a reasonable
moderate-error fit (instead of becoming ill-conditioned).

You can also ignore the dataset-provided pointwise errors entirely by setting:
- `USE_POINTWISE_SIGMAS = False`

**Outputs**
- `<VERSION_DIR>/replicas/replica_XXX_<TAG>.keras`
- `<VERSION_DIR>/histories/history_replica_XXX_<TAG>.json`

> Configure `VERSION_DIR`, `TAG`, training hyperparameters, and the nuisance CFF choices
> in the USER CONFIG block.

---

### 3) `closure_evaluate.py` — summarize replicas + make plots
Loads the trained replica models and produces:

- histograms of extracted **ReH** and **ImH** across replicas (with the truth line)
- pseudodata plots of **XS(ϕ)** and **BSA(ϕ)** with:
  - the **ensemble-mean inferred curve**
  - optional ±1σ band by propagating each replica through `bkm10_lib`
  - optional truth curve overlay

**Notes**
- Reads truth only for plotting and bias checks; it does not affect training.
- If `ReH_std_over_phi` or `ImH_std_over_phi` are not near zero, it indicates your model is
  leaking ϕ dependence into the CFF prediction (the script warns about this).

**Outputs**
- `<VERSION_DIR>/eval/*.png` and a CSV of per-replica CFFs

---

## Quickstart

From the repo root (or wherever these scripts can import `bkm10_lib` and `gepard`):

### Run scripts
Start wit the generate scrpt, then train, then evaluate
```bash
python3 closure_generate_dataset.py

