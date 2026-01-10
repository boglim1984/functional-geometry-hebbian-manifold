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

---
*Note: PLY files are currently excluded by .gitignore until explicitly added by the author.*
