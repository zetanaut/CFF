# Hard Dispersion Relation CFF extraction example

You will need TensorFlow, and the BMK library that is compatible with TensorFlow
which you can get by doing
```bash
pip install bmk10
```
If want to use the Gepard you can look here: https://github.com/kkumer/gepard

This directory contains a **closure-test pipeline** demonstrating a *Hard Dispersion
Relation (Hard-DR)* extraction of the DVCS Compton Form Factor **H** using only:

- unpolarized cross section **XS(ϕ)**
- beam–spin asymmetry **BSA(ϕ)**

at **fixed (t, Q², Ebeam)** but for **multiple xB (equivalently multiple ξ)**.

Unlike the basic example, **Re H is not fitted independently**.
Instead it is **computed from Im H via a fixed-t dispersion relation**, enforced
*by construction* during training.

---

## Conceptual difference vs `basic/`

| basic example | Hard-DR example |
|--------------|----------------|
| Fit ReH and ImH independently | Fit **ImH(ξ)** only |
| Single kinematic point | Multiple xB / ξ nodes |
| Local in ξ | **Non-local in ξ (dispersion relation)** |
| DR optional | **DR enforced hard** |

Hard-DR requires multiple ξ points because a dispersion relation connects  
Re H(ξᵢ) to an *integral over Im H(x)*.

---

## What each script does

### 1) `generator.py` — Hard-DR-consistent closure dataset

Generates a **multi-xB closure dataset** that is guaranteed to be compatible
with the Hard-DR training script.

Key features:
- Automatically selects **physical xB nodes** where `bkm10_lib` is finite
- Enforces a **common ϕ grid** across all xB bins
- Constructs Hard-DR truth:
  - **ImH(ξ)** from KM15
  - **ReH(ξ) = C₀ + K · ImH** using the *same discretized DR kernel* used in training
- Uses fixed nuisance CFFs (E, H̃, Ẽ)

Outputs (under `<OUT_DIR>/data`):
- `dataset_<TAG>.npz`
- `dataset_<TAG>.csv`
- `truth_<TAG>.json` *(for evaluation only)*

---

### 2) `HardDR_training.py` — Hard-DR replica training

Trains an ensemble of replica fits enforcing the dispersion relation **hard**.

Model structure:
- Neural network parameterizes **ImH(ξ)**
- A single trainable scalar **C₀** represents the subtraction constant
- **ReH(ξ) = C₀ + K · ImH(ξ)** (fixed kernel, no freedom)

Training details:
- Uses a soft-χ² loss (stable even when dataset errors are zero)
- Replica method for uncertainty estimation
- Uses finite-difference gradients through the BKM forward model
- Saves **weights + metadata** by default (recommended for subclassed models)

Outputs:
- `<VERSION_DIR>/replicas_hard_dr/replica_XXX_<TAG>.weights.h5`
- `<VERSION_DIR>/replicas_hard_dr/replica_XXX_<TAG>_meta.npz`
- training histories in `histories_hard_dr/`

---

### 3) `evaluation.py` — Hard-DR evaluation and diagnostics

Evaluates the trained Hard-DR replicas and compares them to truth.

Produces:
- Histogram of **C₀** across replicas
- Mean ± 1σ bands for **ImH(ξ)** and **ReH(ξ)** vs truth
- XS(ϕ) and BSA(ϕ) plots **per xB bin**
- CSV with per-replica parameters

Supports:
- Loading weights+meta (default) or full `.keras` models
- Truth overlay for closure diagnostics
- Optional ensemble bands for observable curves

Outputs written to:
<VERSION_DIR>/eval_hard_dr/<TAG>/
From the repository root:

### To run, first run generator then training then evaluation:
```bash
python3 generator.py
