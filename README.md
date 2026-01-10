# Functional Geometry in Deep Neural Networks

## Overview

This repository documents an experimental research arc investigating learned functional geometry in deep neural networks. The project demonstrates the existence, causality, and system-level effects of geometric structure that emerges during training, and explores its implications for model compression, interpretability, and plasticity.

## Core Hypothesis

Neural networks learn functional geometry—structured manifolds in activation space that encode redundancy, stiffness, and causal relationships between neurons. This geometry can be:
- **Mapped** through dimensionality reduction and clustering
- **Validated** through targeted surgical interventions
- **Exploited** for safe model consolidation
- **Perturbed** to study plasticity dynamics

## Experimental Phases

### Phase I: Neuro-Cartography (Manifold Discovery)

**Objective**: Map the functional geometry of trained networks by analyzing activation patterns across layers.

**Method**: 
- Extract activation patterns from multiple layers
- Apply dimensionality reduction (UMAP/t-SNE)
- Identify neuron clusters and manifold structure
- Export geometric representations as 3D point clouds

**Notebook**: [01_neuro_cartography.ipynb](https://colab.research.google.com/drive/1Fq1l2yQtmzHF7zrIWcXd_1iyKWYtFNm6?usp=drive_link)

**Artifacts**: Manifold visualizations (PLY format) in `artifacts/ply/`

### Phase II: Biopsy (Causal Testing)

**Objective**: Validate that geometric proximity predicts functional redundancy through targeted neuron removal.

**Method**:
- Select neurons based on manifold position (proximal vs. distal)
- Perform controlled ablations
- Measure performance degradation
- Compare against random baseline

**Notebooks**:
- [02_biopsy_success.ipynb](https://colab.research.google.com/drive/110sO1CQ5d8Worg3PrXJY2nmGgRtbdR5M?usp=drive_link) - Successful causal validation
- [03_failed_biopsy.ipynb](https://colab.research.google.com/drive/110sO1CQ5d8Worg3PrXJY2nmGgRtbdR5M?usp=drive_link) - **FAILED/DIAGNOSTIC** - Documents unsuccessful intervention

**Key Finding**: Merging geometrically proximal neurons preserves function; merging distant neurons causes instability.

### Phase III: Mass Consolidation

**Objective**: Scale geometry-guided compression to system level.

**Method**:
- Batch process multiple layers
- Apply geometry-informed merge strategies
- Validate preservation of model capabilities
- Compare compression ratios against naive approaches

**Scripts**: `scripts/neuro_surgeon_batch.py`

**Status**: TBD

### Phase IV: Sleep / Plasticity Test

**Objective**: Investigate whether geometric perturbation affects learning dynamics.

**Method**:
- Introduce controlled noise to network weights
- Measure plasticity changes during retraining
- Test hypothesis that damage increases adaptability

**Scripts**: `scripts/neuro_sleep.py`

**Status**: TBD

## Key Findings

1. **Functional geometry exists**: Neurons organize into structured manifolds in activation space
2. **Geometry predicts redundancy**: Proximal neurons in geometric space exhibit functional overlap
3. **Local merges are safe**: Consolidating nearby neurons preserves performance
4. **Distant merges destabilize**: Merging geometrically distant neurons degrades function
5. **Geometry-guided consolidation works**: Compression informed by manifold structure outperforms naive approaches
6. **Geometric perturbation affects plasticity**: TBD
7. **Geometry encodes stability, not learning speed**: TBD

## What This Is / Is Not

**This is**:
- A mechanistic interpretability study
- An exploration of compression priors
- A demonstration of causal validation methods
- An archive of experimental results with controls

**This is not**:
- A claim about biological neural networks
- A general theory of deep learning
- A production-ready compression method
- A complete explanation of network function

## Repository Structure

```
/
├── README.md                          # This file
├── executive_summary.md               # High-level conclusions
├── notebooks/                         # External Colab links
│   ├── 01_neuro_cartography.ipynb
│   ├── 02_biopsy_success.ipynb
│   └── 03_failed_biopsy.ipynb
├── scripts/                           # Analysis and intervention tools
│   ├── neuro_cartographer.py         # Manifold mapping
│   ├── neuro_surgeon.py              # Single-neuron ablation
│   ├── neuro_surgeon_batch.py        # Batch consolidation
│   └── neuro_sleep.py                # Plasticity experiments
├── artifacts/                         # Generated outputs
│   ├── ply/                          # 3D manifold exports
│   └── figures/                      # Plots and visualizations
├── docs/                              # Additional documentation
│   └── research_notes.md
└── LICENSE                            # MIT License
```

## Notebooks & Artifacts

All experimental notebooks are hosted on Google Colab and linked in the respective phase sections above. Manifold visualizations (PLY files) can be found in `artifacts/ply/` and viewed with MeshLab, CloudCompare, or Blender.

## Status & Future Work

**Completed**:
- Phase I: Manifold discovery and visualization
- Phase II: Causal validation through targeted ablation

**In Progress**:
- Phase III: System-scale consolidation

**Planned**:
- Phase IV: Plasticity and relearning dynamics
- Cross-architecture validation
- Theoretical formalization

## Reproducibility / Environment

To ensure consistent experimental results, please use the environment specified in `requirements.txt`.

**Note on UMAP**: Manifold geometry can vary slightly between library versions. It is recommended to use the pinned versions to maintain parity with the documented artifacts.

```bash
pip install -r requirements.txt
```

## Citation

If you use this work, please cite:

```
@misc{functional-geometry-2026,
  title={Functional Geometry in Deep Neural Networks},
  author={Billy O’Flaherty},
  year={2026},
  url={https://github.com/boglim1984/functional-geometry-hebbian-manifold}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
