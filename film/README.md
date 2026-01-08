# FiLM-gated non-bias DR CFF extraction example

This directory contains a **non-bias dispersion-relation (DR) extraction pipeline**
for the DVCS Compton Form Factor **H**, using a **FiLM-conditioned gating mechanism**
to decide *whether and where* a dispersion-relation constraint should be applied.

Unlike the Hard-DR and gated-DR examples, **DR is treated as a proposal, not a prior**:
it is only accepted if it improves the description of held-out data.

---

## Conceptual position in the example sequence

| Example | DR usage | Decision rule |
|-------|---------|---------------|
| `basic/` | none | — |
| `hard_dr/` | enforced | fixed |
| `gated_dr_pv_opt/` | preferred | penalized |
| **`film_dr_nobias/`** | **proposed** | **accepted only if data improve** |

This example is designed to answer:
> *“Should DR be used at all — and if so, where?”*

---

## Core idea (non-bias principle)

1. First learn a **purely data-driven solution**:
   \[
   \Re H_{\rm free}(\xi), \quad \Im H(\xi)
   \]

2. Construct a **DR proposal** from the *same* \(\Im H\):
   \[
   \Re H_{\rm DR}(\xi) = C_0 + K \cdot \Im H
   \]

3. Blend them using a **FiLM-conditioned gate**:
   \[
   \Re H_{\rm pred}(\xi)
   =
   \Re H_{\rm free}(\xi)
   + \alpha(\xi)\,[\Re H_{\rm DR}(\xi) - \Re H_{\rm free}(\xi)]
   \]

4. **Accept the DR-gated model only if it improves validation data loss**
   by a configurable threshold.

If DR does *not* help, the final model **reverts to the free solution**.

---

## What each script does

### 1) `generate.py` — FiLM-DR closure dataset

Generates a **multi-ξ closure dataset** with explicit control over DR consistency.

Truth construction:
- Choose a ξ grid (uniform, user-defined)
- Define a smooth **ImH_truth(ξ)**
- Compute **ReH_DR_truth = C0 + K · ImH_truth**
- Optionally inject a **controlled DR violation**:
ReH_truth = ReH_DR + Delta_viol(ξ)

Key features:
- DR-consistent or DR-violating truth modes
- PV kernel stored directly in the dataset
- Clean multi-bin XS(ϕ), BSA(ϕ) generation

Outputs:
- `<OUT_DIR>/data/dataset_<TAG>.npz`
- `<OUT_DIR>/data/dataset_<TAG>.csv`
- `<OUT_DIR>/data/truth_<TAG>.json`

---

### 2) `train_film.py` — FiLM-gated non-bias training

Trains an ensemble of replicas using a **two-stage selection strategy**.

**Stage 1 — free model**
- Learns `ReH_free(ξ)` and `ImH(ξ)`
- No DR information enters the prediction
- Establishes the baseline data likelihood

**Stage 2 — DR as proposal**
- Constructs `ReH_DR` from the learned `ImH`
- Uses a **FiLM-conditioned gate α(ξ)** to blend free and DR solutions
- Applies strong regularization to keep DR *off* unless beneficial

**Non-bias selection rule**
- If Stage-2 improves validation data loss by at least `MIN_IMPROVEMENT`:
→ keep FiLM-gated DR
- Otherwise:
→ revert to pure free solution

This guarantees DR is **never imposed** when unsupported by data.

Outputs:
- `<VERSION_DIR>/replicas_film_dr_nobias/replica_XXX_<TAG>.weights.h5`
- `<VERSION_DIR>/replicas_film_dr_nobias/replica_XXX_<TAG>_meta.npz`
- training histories in `histories_film_dr_nobias/`

---

### 3) `evaluate.py` — non-bias diagnostics and visualization

Evaluates the trained replicas and summarizes **model selection behavior**.

Produces:
- Histograms of:
- \(C_0\)
- mean gate value ⟨α⟩
- validation-loss improvement
- Mean ± 1σ bands vs ξ for:
- ImH(ξ)
- ReH_free(ξ)
- ReH_DR(ξ)
- ReH_pred(ξ)
- α(ξ)
- XS(ϕ) and BSA(ϕ) per xB bin with:
- pseudodata
- inferred mean curve
- optional ensemble band
- optional truth overlay

Outputs written to:
<VERSION_DIR>/eval_film_dr_nobias/<TAG>/

---

## Quickstart

From the repository root:

### To run —
```bash
python generate.py
