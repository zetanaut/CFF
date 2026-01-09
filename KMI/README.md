# KMI (no-DR) CFF extraction example

This directory contains a **Kinematic Model–Independent (KMI)** extraction of the
DVCS Compton Form Factor **H**, performed **without using dispersion relations (DR)**.

The KMI example serves as the **baseline, fully data-driven reference** against which
all DR-based and DR-gated approaches can be compared.

---

## Conceptual role in the example sequence

| Example | Uses DR? | Purpose |
|-------|----------|---------|
| `basic/` | No | minimal single-point demo |
| `hard_dr/` | enforced | test DR consistency |
| `gated_dr_pv_opt/` | preferred | penalized DR |
| `film_dr_nobias/` | proposed | non-bias DR selection |
| **`kmi/`** | No | **pure data baseline** |

This example answers:
> *“What does the data alone tell us, with no theoretical constraints imposed?”*

---

## Core idea (KMI)

- A single neural network learns **ReH and ImH directly** as functions of kinematics
- No dispersion relations, kernels, gates, or DR priors are used
- The model is trained **only through comparison to XS(ϕ) and BSA(ϕ)** via the BKM
  forward model
- Each kinematic bin is treated as an independent degree of freedom

This is the **least biased** extraction possible in this framework.

---

## What each script does

### 1) `generator.py` — robust KMI dataset generation

Generates a **multi-bin DVCS dataset** with realistic kinematics and no DR assumptions.

Key features:
- Automatically selects **physically valid kinematic bins**
  (enforcing \(y<1\), \(W>W_{\min}\), and finite BKM outputs)
- Supports **ragged ϕ coverage**:
  - different bins can have different numbers of ϕ points
- Avoids pathological ϕ endpoints (0°, 180°, 360°)
- Truth CFFs can come from:
  - **KM15 (via gepard)**, or
  - a smooth toy model (fallback)

Outputs:
- `<OUT_DIR>/data/dataset_<TAG>.npz`
- `<OUT_DIR>/data/dataset_<TAG>.csv`
- `<OUT_DIR>/data/truth_<TAG>.json`

This dataset is intentionally realistic and slightly irregular, reflecting
actual experimental conditions.

---

### 2) `training_kmi.py` — global KMI training (no DR)

Trains an ensemble of replicas using a **purely kinematic neural network**.

Model:
- One shared DNN:
(t, xB, log Q², ξ) → (ReH, ImH)
- ϕ is **never** an input to the network (prevents unphysical ϕ leakage)
- XS and BSA are computed through `bkm10_lib` using fixed nuisance CFFs

Training strategy:
- Validation is performed by **holding out entire kinematic bins**
(not individual ϕ points)
- Uses dense masks to avoid TensorFlow `IndexedSlices` gradient pathologies
- Soft-χ² loss with robust sigma floors for numerical stability
- Replica ensemble for uncertainty propagation

Outputs:
- `<VERSION_DIR>/replicas_kmi_no_dr/replica_XXX_<TAG>.weights.h5`
- `<VERSION_DIR>/replicas_kmi_no_dr/replica_XXX_<TAG>_meta.npz`
- training histories in `histories_kmi_no_dr/`

---

### 3) `evaluate.py` — KMI surface evaluation and visualization

Evaluates the trained KMI replicas and compares them to truth.

Produces:

**Bin-level diagnostics**
- CSV of extracted CFFs per kinematic bin

**Surface-level diagnostics**
- Randomly sampled kinematic points across the physical domain
- CSV with mean, σ, truth, pull, and coverage
- Summary JSON with RMSE, bias, and pull statistics

**3D surface plots (main feature)**
- For fixed-t slices:
- truth surface (wireframe)
- ensemble mean surface
- transparent ±1σ uncertainty band
- Surfaces shown over \((x_B, Q^2)\)

Outputs written to:
<VERSION_DIR>/eval_kmi_surface/<TAG>/
