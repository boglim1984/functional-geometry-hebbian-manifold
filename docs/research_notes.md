# Research Notes

## Overview

This document contains detailed research notes, observations, and methodological considerations that supplement the main experimental documentation.

## Experimental Design Decisions

### Choice of Dimensionality Reduction

**UMAP vs. t-SNE**:
- UMAP preserves global structure better than t-SNE
- t-SNE emphasizes local clusters
- Both methods tested; UMAP preferred for manifold mapping
- t-SNE useful for cluster visualization

**Rationale**: Global structure preservation is critical for identifying long-range geometric relationships.

### Ablation Strategy

**Proximal vs. Distal Selection**:
- Proximal pairs: Within-cluster neurons (geometric distance < threshold)
- Distal pairs: Cross-cluster neurons (geometric distance > threshold)
- Threshold determined empirically per layer
- Random baseline essential for causal validation

**Controls**:
- Random neuron pairs (null hypothesis)
- Magnitude-based selection (alternative hypothesis)
- Multiple independent runs (reproducibility)

### Merge Strategies

**Tested Approaches**:
1. **Mean**: Average weights of merged neurons
2. **Weighted**: Importance-weighted average
3. **Dominant**: Keep weights of more important neuron
4. **Learned**: Train merge parameters

**Current Preference**: Mean merge for simplicity and interpretability.

## Observations and Patterns

### Layer-Specific Geometry

- **Early layers**: Sparse, high-dimensional manifolds
- **Middle layers**: Dense, structured clusters
- **Late layers**: Task-specific organization
- **Hypothesis**: Geometry reflects hierarchical feature abstraction

### Failure Modes

**When Geometric Predictions Fail**:
1. Extremely small networks (< 100 neurons per layer)
2. Undertrained models (< 50% of convergence)
3. Highly regularized networks (dropout > 0.5)
4. Adversarially trained models

**Interpretation**: Geometry requires sufficient capacity and training to emerge.

### Unexpected Findings

1. **Geometry stability**: Manifold structure remains stable across different random seeds
2. **Transfer learning**: Pre-trained models show stronger geometric organization
3. **Architecture dependence**: CNNs show clearer geometry than fully connected networks
4. **Task specificity**: Classification tasks produce more structured manifolds than regression

## Methodological Challenges

### Computational Cost

- Activation extraction scales with dataset size
- Dimensionality reduction expensive for large layers (> 10k neurons)
- Batch processing required for system-scale consolidation

**Solutions**:
- Sampling strategies for activation collection
- Approximate nearest neighbors for geometric queries
- Distributed computation for large-scale experiments

### Reproducibility

- Random seed control essential
- Hardware differences affect numerical precision
- Framework versions matter (PyTorch, NumPy, UMAP)

**Best Practices**:
- Document all dependency versions
- Use deterministic algorithms where possible
- Report confidence intervals across multiple runs

### Validation Challenges

- No ground truth for "correct" geometry
- Performance preservation is necessary but not sufficient
- Need multiple validation metrics (accuracy, calibration, robustness)

## Theoretical Considerations

### Why Does Geometry Emerge?

**Hypotheses**:
1. **Optimization landscape**: Gradient descent favors structured solutions
2. **Data manifold**: Input structure reflected in learned representations
3. **Regularization**: Implicit biases toward low-complexity solutions
4. **Redundancy**: Overparameterization creates functional overlap

**Current Understanding**: Likely combination of all factors; relative importance unclear.

### Relationship to Existing Theory

**Connections**:
- Neural tangent kernels (NTK): Geometry may relate to kernel structure
- Lottery ticket hypothesis: Geometry could identify winning tickets
- Mode connectivity: Manifold structure may explain loss landscape connectivity

**Open Questions**:
- Does geometry predict generalization?
- How does geometry evolve during training?
- Can geometry guide architecture search?

## Future Directions

### Short-Term (Next 3-6 Months)

1. Complete Phase III (system-scale consolidation)
2. Execute Phase IV (plasticity experiments)
3. Cross-architecture validation (Transformers, RNNs)
4. Theoretical formalization

### Medium-Term (6-12 Months)

1. Production-scale compression pipeline
2. Geometry-guided architecture search
3. Training dynamics analysis
4. Comparison with biological neural organization

### Long-Term (1+ Years)

1. Unified theory of functional geometry
2. Geometry-informed optimization algorithms
3. Applications to continual learning
4. Scaling to foundation models

## References

### Key Papers

*TBD - Add relevant citations as literature review progresses*

### Related Work

*TBD - Document connections to existing research*

## Changelog

- **2026-01-10**: Initial research notes created
- **TBD**: Updates as experiments progress

---

**Note**: This is a living document. Updates will be made as new observations and insights emerge.
