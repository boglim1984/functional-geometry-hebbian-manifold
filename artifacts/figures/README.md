# Figures and Visualizations

This directory contains plots, charts, and visual analyses generated during the research.

## Contents

Figures are organized by experimental phase:

### Phase I: Manifold Discovery
- Dimensionality reduction plots (UMAP/t-SNE)
- Cluster visualizations
- Layer-wise geometry comparisons
- Activation pattern heatmaps

### Phase II: Causal Validation
- Performance degradation curves
- Geometric distance vs. functional impact
- Baseline comparisons
- Statistical significance plots

### Phase III: System-Scale Consolidation
- Compression ratio charts
- Layer-wise consolidation results
- Performance preservation metrics
- Comparison with naive approaches

### Phase IV: Plasticity Experiments
- Learning curve comparisons
- Plasticity metrics over time
- Perturbation effect visualizations
- Convergence speed analysis

## File Naming Convention

```
<phase>_<experiment>_<plot_type>_<timestamp>.png

Examples:
- phase1_umap_embedding_20260110.png
- phase2_ablation_performance_20260110.png
- phase3_compression_comparison_20260110.png
```

## Current Files

*No figures currently in repository. Figures will be added as experiments are completed and visualizations are generated.*

## Generating Figures

Most figures are generated within the experimental notebooks (see `notebooks/` directory). To regenerate:

1. Open the relevant Colab notebook
2. Run all cells
3. Download generated figures
4. Place in this directory following naming convention

## Citation

If you use these figures in publications or presentations, please cite the main repository and reference the specific experiment.
