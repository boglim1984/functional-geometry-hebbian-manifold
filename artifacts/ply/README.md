# PLY Manifold Exports

This directory contains 3D point cloud representations of neural network functional geometry, exported in PLY (Polygon File Format) format.

## What These Files Represent

Each PLY file represents the geometric structure of neuron activations from a specific layer:

- **Points**: Individual neurons
- **Coordinates**: Low-dimensional embedding (typically 3D) of high-dimensional activation patterns
- **Clusters**: Groups of functionally related neurons
- **Distances**: Geometric proximity correlates with functional redundancy

## How They Were Generated

1. **Activation Extraction**: Layer activations collected across diverse inputs
2. **Dimensionality Reduction**: UMAP or t-SNE applied to activation matrix
3. **Embedding**: High-dimensional patterns mapped to 3D space
4. **Export**: Coordinates saved as PLY point cloud

See `scripts/neuro_cartographer.py` for implementation details.

## How to Visualize

### MeshLab
```bash
# Install MeshLab (cross-platform)
# macOS: brew install --cask meshlab
# Linux: sudo apt-get install meshlab
# Windows: Download from meshlab.net

# Open PLY file
meshlab <filename>.ply
```

### CloudCompare
```bash
# Install CloudCompare (cross-platform)
# macOS: brew install --cask cloudcompare
# Linux: sudo snap install cloudcompare
# Windows: Download from cloudcompare.org

# Open PLY file
cloudcompare.CloudCompare <filename>.ply
```

### Blender
```python
# Open Blender
# File > Import > Stanford PLY (.ply)
# Select file and import

# Recommended settings:
# - Display as points or mesh
# - Apply color mapping if available
# - Use orthographic view for geometric analysis
```

### Python (Programmatic)
```python
from plyfile import PlyData
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load PLY file
plydata = PlyData.read('layer_manifold.ply')
vertices = plydata['vertex']
x, y, z = vertices['x'], vertices['y'], vertices['z']

# Visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()
```

## File Naming Convention

```
<model>_<layer>_<method>_<timestamp>.ply

Examples:
- resnet50_layer3_umap_20260110.ply
- vgg16_conv4_tsne_20260110.ply
```

## Large File Handling

PLY files can become large for networks with many neurons. If files exceed 100MB, consider:

1. **Git LFS** (Large File Storage):
   ```bash
   git lfs install
   git lfs track "*.ply"
   git add .gitattributes
   ```

2. **External Hosting**: Upload to Zenodo, Figshare, or Google Drive and link here

3. **Compression**: Use `gzip` or `xz` for archival
   ```bash
   gzip layer_manifold.ply  # Creates layer_manifold.ply.gz
   ```

## Current Files

The following PLY manifold files are included in this repository:

### Phase I: Neuro-Cartography

| File | Description | Size | Date Added |
|------|-------------|------|------------|
| `phase1_layer_trained_manifold.ply` | Trained network layer manifold (UMAP embedding) | 22KB | 2026-01-10 |
| `phase1_layer4_untrained_manifold.ply` | Untrained network layer 4 manifold (baseline control) | 24KB | 2026-01-10 |
| `phase1_layer_shuffle_control.ply` | Shuffled weights control manifold (null hypothesis test) | 23KB | 2026-01-10 |

**Experimental Context**:
- **Trained manifold**: Demonstrates learned functional geometry in a trained network
- **Untrained manifold**: Baseline control showing random initialization structure
- **Shuffle control**: Null hypothesis test with randomized weight assignments

These files can be visualized using MeshLab, CloudCompare, Blender, or Python (see visualization instructions above).

## Metadata

Each PLY file may include additional properties:
- `cluster_id`: Neuron cluster assignment
- `importance`: Neuron importance score (if available)
- `layer_depth`: Position in network architecture

Check individual file headers for available properties.

## Citation

If you use these manifold visualizations, please cite the main repository and reference the specific experiment that generated them.
