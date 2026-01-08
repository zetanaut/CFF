# Gated-DR CFF extraction example (optimized PV)

This directory contains an **optimized closure-test pipeline** for extracting the DVCS
Compton Form Factor **H** using a **gated dispersion-relation (DR)** model with a
*differentiable principal-value (PV) layer*.

This example generalizes the Hard-DR case by allowing **controlled, data-driven
violations of the dispersion relation** while strongly preferring DR when supported
by the data.

---

## Conceptual position in the sequence

| Example | DR treatment | Re H freedom |
|-------|-------------|--------------|
| `basic/` | none | fully free |
| `hard_dr/` | enforced | none |
| **`gated_dr_pv_opt/`** | **preferred, not enforced** | **controlled + penalized** |

The goal is to:
- recover Hard-DR behavior when DR is valid,
- allow deviations when data demand it,
- avoid instability or basin-hopping during training.

---

## Physics idea (short)

We decompose the real part as
\[
\Re H(\xi) = \Re H_{\rm DR}(\xi) + (1-g(\xi))\,\Delta\Re H(\xi),
\]
where
- \(\Re H_{\rm DR} = C_0 + K \cdot \Im H\) is the dispersion-relation prediction,
- \(g(\xi)\in[0,1]\) is a learned **gate**,
- \(\Delta\Re H\) is a learned correction,
- the *effective* DR violation \((1-g)\Delta\Re H\) is **explicitly penalized**.

Training is staged so that Hard-DR converges *before* any correction is released.

---

## What each script does

### 1) `generator.py` — gated-DR-consistent closure dataset

Generates a **multi-xB closure dataset** with a tunable amount of DR violation.

Truth construction:
- **ImH(ξ)** from KM15
- **ReH_DR(ξ) = C₀ + K · ImH**
- **ReH_free(ξ)** from KM15
- Final truth:
ReH_truth = g_truth * ReH_DR + (1 - g_truth) * ReH_free
- `g_truth = 1` → exact DR closure
- `g_truth = 0` → fully DR-violating truth

Key safeguards:
- Automatic selection of **physical xB nodes** (finite BKM output)
- Common ϕ grid across bins
- PV kernel stored directly in the dataset for closure consistency

Outputs:
- `<OUT_DIR>/data/dataset_<TAG>.npz`
- `<OUT_DIR>/data/dataset_<TAG>.csv`
- `<OUT_DIR>/data/truth_<TAG>.json` *(evaluation only)*

---

### 2) `training.py` — optimized gated-DR replica training

Trains an ensemble of replicas using a **two-stage optimization strategy**.

**Stage 1 — Hard-DR**
- Gate ≈ 1, ΔReH ≈ 0
- Only ImH and C₀ are optimized
- Ensures the DR solution is found first

**Stage 2 — gated release**
- Correction activated smoothly via an **α-ramp**
- Penalizes the *effective* correction:
\[
(1-g)\Delta\Re H
\]
- Strong priors keep DR preferred unless data demand otherwise

Key features:
- Differentiable PV dispersion layer
- Soft-χ² loss stable for zero experimental error
- Explicit penalties on gate, correction magnitude, and roughness
- Replica ensemble for uncertainty propagation

Outputs:
- `<VERSION_DIR>/replicas_gated_dr_pv_opt/replica_XXX_<TAG>.weights.h5`
- `<VERSION_DIR>/replicas_gated_dr_pv_opt/replica_XXX_<TAG>_meta.npz`
- training histories in `histories_gated_dr_pv_opt/`

---

### 3) `evaluate.py` — gated-DR diagnostics and visualization

Evaluates the trained replicas and summarizes the learned DR behavior.

Produces:
- Histograms of:
- \(C_0\)
- mean gate value ⟨g⟩
- RMS effective correction
- Mean ± 1σ bands vs ξ for:
- ImH(ξ)
- ReH_DR(ξ)
- ReH_total(ξ)
- gate(ξ)
- effective correction
- XS(ϕ) and BSA(ϕ) per xB bin with:
- data
- inferred mean curve
- optional ensemble band
- optional truth overlay

Outputs written to:
<VERSION_DIR>/eval_gated_dr_pv_opt/<TAG>/

---

## Quickstart

From the repository root:

### To run start with the generator, then train, then evaluate
```bash
python3 generator.py
