# PLY Manifold Artifacts

This directory is the canonical location for manifold exports.

### File Naming Convention
Please name all PLY files using the following format:
`phase_condition_layer.ply`
(e.g., `phase1_trained_layer3.ply`, `phase2_untrained_conv4.ply`)

### Large Files & Storage
Manifold files can be large. 
- **Git LFS**: It is highly recommended to enable Git LFS before committing `.ply` files to this directory.
- **Releases**: For exceptionally large static files (>100MB), consider hosting them as GitHub Release assets.

### Included Artifacts

| Filename | Phase | Layer | Condition | Description | Size |
|----------|-------|-------|-----------|-------------|------|
| `phase1_trained_layer1.ply` | Phase I | layer1 | trained | Trained manifold for Phase I cartography | 22KB |
| `phase1_shuffle_layer1.ply` | Phase I | layer1 | shuffle | Shuffle control for Phase I cartography | 23KB |
| `phase1_untrained_layer4.ply` | Phase I | layer4 | untrained | Untrained baseline for layer 4 | 24KB |

---
*Note: PLY files are force-added to this repository despite .gitignore for small file visibility.*

