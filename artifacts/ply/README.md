# PLY Manifold Exports

## ⚠️ Adding PLY Artifacts (Author Action Required)

**STATUS**: PLY files are **NOT yet included** in this repository.

### Action Required from Billy

Please provide the `.ply` manifold files for inclusion in this repository. For each file, supply the following metadata:

**Required Metadata**:
1. **Filename**: Descriptive name (e.g., `resnet50_layer3_trained_umap.ply`)
2. **Experiment Phase**: Phase I, II, III, or IV
3. **Layer Name**: Specific layer identifier (e.g., `layer3`, `conv4`, `fc2`)
4. **Condition**: 
   - `trained` - Trained network
   - `untrained` - Random initialization baseline
   - `pixel-shuffle` - Shuffled weights control
   - `near-merge` - Proximal neuron merge
   - `far-merge` - Distal neuron merge
   - Other (specify)
5. **Short Description**: Brief explanation of what this manifold represents

### Storage Options

Choose one of the following methods for adding PLY files:

#### Option A: Git LFS (Recommended for Versioned Artifacts)

**Best for**: Files that may be updated, files < 2GB, files that need version history

**Setup**:
```bash
# Install Git LFS
git lfs install

# Track PLY files
git lfs track "artifacts/ply/*.ply"

# Add and commit
git add .gitattributes
git add artifacts/ply/your_file.ply
git commit -m "Add PLY artifact: [description]"
git push
```

**Pros**: Version control, easy updates, integrated with repository  
**Cons**: Requires Git LFS setup, bandwidth limits on free tier

#### Option B: GitHub Releases (Recommended for Large Static Files)

**Best for**: Files > 100MB, static archives, large collections

**Process**:
1. Go to: https://github.com/boglim1984/functional-geometry-hebbian-manifold/releases
2. Click "Create a new release"
3. Tag version (e.g., `v1.0-ply-artifacts`)
4. Upload PLY files as release assets
5. Update this README with download links

**Pros**: No size limits, doesn't bloat repository, permanent URLs  
**Cons**: No version control, separate from main repository

### PLY File Manifest Template

Once files are provided, they will be documented here:

| Filename | Phase | Layer | Condition | Description | Size | Date Added | Storage |
|----------|-------|-------|-----------|-------------|------|------------|---------|
| _awaiting files_ | - | - | - | - | - | - | - |

**Example Entry**:
| Filename | Phase | Layer | Condition | Description | Size | Date Added | Storage |
|----------|-------|-------|-----------|-------------|------|------------|---------|
| `resnet50_layer3_trained_umap.ply` | Phase I | layer3 | trained | UMAP embedding of trained ResNet-50 layer 3 activations | 85MB | 2026-01-10 | Git LFS |

---

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
<phase>_<layer>_<condition>_<method>.ply

Examples:
- phase1_layer3_trained_umap.ply
- phase1_layer4_untrained_umap.ply
- phase2_conv4_near_merge_tsne.ply
- phase2_fc2_far_merge_umap.ply
```

## Metadata

Each PLY file may include additional properties:
- `cluster_id`: Neuron cluster assignment
- `importance`: Neuron importance score (if available)
- `layer_depth`: Position in network architecture

Check individual file headers for available properties.

## Citation

If you use these manifold visualizations, please cite the main repository and reference the specific experiment that generated them.
