# How to Read This Repository

This repository documents an empirical investigation into **learned functional geometry in deep neural networks**. It is organized as a sequence of falsifiable experiments rather than a single monolithic result.

You do **not** need to run the code to understand the core findings. The notebooks and artifacts are the primary evidence; scripts exist to support reproducibility.

---

## 1. Start with the Executive Summary
**File:** `executive_summary.md`

Read this first.

It states:
- the research question,
- what was tested,
- what was confirmed,
- what failed,
- and what is *not* being claimed.

If you only read one file, read this.

---

## 2. Experimental Phases

### Phase 1 — Cartography (Mapping the Geometry)
**Goal:** Determine whether trained networks exhibit coherent internal geometry.

**Notebooks:**
- `01_neuro_cartography.ipynb`
- `02_untrained_baseline_manifold.ipynb`
- `03_pixel_shuffle_control.ipynb`

**Artifacts:**
- `artifacts/ply/phase1_trained_layer4.ply`
- `artifacts/ply/phase1_untrained_layer4.ply`
- `artifacts/ply/phase2_pixelshuffle_layer4.ply`

**Key Result:** Coherent geometry emerges only when semantic structure is learned.

---

### Phase 2 — Biopsy (Causal Intervention)
**Goal:** Test whether geometric proximity predicts functional redundancy.

**Notebooks:**
- `04_failed_biopsy.ipynb` (FAILED / diagnostic)
- `05_neuro_surgeon_biopsy_v2.ipynb`

**Key Result:** Nearby neurons can be merged with minimal impact; distant merges are disruptive.

---

### Phase 3 — Mass Consolidation
**Goal:** Test geometry-guided compression at system scale.

**Notebooks:**
- `06_neuro_surgeon_batch_biopsy.ipynb`
- `07_neuro_surgeon_mass_consolidation.ipynb`

**Key Result:** Geometry-guided consolidation is significantly safer than random consolidation.

---

### Phase 4 — Plasticity / Sleep
**Goal:** Test whether consolidated networks retain learning capacity.

**Notebook:**
- `08_neuro_sleep.ipynb`

**Key Result:** Geometry-merged networks retain plasticity comparable to baseline.

---

## 3. Scripts vs Notebooks

- **notebooks/** — primary evidence and experiments  
- **scripts/** — reusable / reproducibility tooling  

Reviewers should prioritize notebooks.

---

## 4. Artifacts (PLY Files)

Located in `artifacts/ply/`.

These are 3D point clouds of activation manifolds and can be viewed using:
- MeshLab
- CloudCompare
- Blender

They are illustrative evidence, not analysis outputs.

---

## 5. Scope Clarification

**This repository is:**
- Empirical mechanistic interpretability research
- A validated geometry → function mapping
- A reproducible research archive

**This repository is not:**
- A production compression library
- A biological brain model
- A general intelligence theory

Claims are intentionally narrow and evidence-bound.

---

## Suggested Reading Order
1. `executive_summary.md`
2. `README.md`
3. Phase 1 notebooks
4. Phase 2 notebooks (including failure)
5. Phase 3 notebooks
6. Phase 4 notebook
7. PLY artifacts (optional)
