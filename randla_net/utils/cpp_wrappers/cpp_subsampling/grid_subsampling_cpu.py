"""
CPU-only replacement for RandLA-Net's compiled `grid_subsampling` module.

The original repo ships precompiled Linux .so binaries (x86_64-linux-gnu),
which cannot run on Windows regardless of build tools. This module
reproduces the same `compute(points, features=None, classes=None,
sampleDl=..., verbose=0)` interface in pure NumPy.

Method: voxel grid subsampling.
- Points falling in the same grid cell are merged into one point at their
  barycenter (mean position) — same "method = barycenter" behavior as the
  original C++ docstring describes.
- Features (if given) are averaged the same way within each cell.
- Labels/classes (if given) are assigned by majority vote within each cell,
  since averaging integer class labels doesn't make sense.
"""

import numpy as np


def compute(points, features=None, classes=None, sampleDl=0.1, verbose=0):
    points = np.asarray(points, dtype=np.float32)

    cell_indices = np.floor(points / sampleDl).astype(np.int64)
    keys, inverse = np.unique(cell_indices, axis=0, return_inverse=True)
    num_cells = keys.shape[0]

    if verbose:
        print(f"[grid_subsampling_cpu] {points.shape[0]} points -> {num_cells} cells (grid_size={sampleDl})")

    sub_points = np.zeros((num_cells, 3), dtype=np.float32)
    counts = np.bincount(inverse, minlength=num_cells)
    for dim in range(3):
        sub_points[:, dim] = np.bincount(inverse, weights=points[:, dim], minlength=num_cells) / counts

    outputs = [sub_points]

    if features is not None:
        features = np.asarray(features, dtype=np.float32)
        sub_features = np.zeros((num_cells, features.shape[1]), dtype=np.float32)
        for dim in range(features.shape[1]):
            sub_features[:, dim] = np.bincount(inverse, weights=features[:, dim], minlength=num_cells) / counts
        outputs.append(sub_features)

    if classes is not None:
        classes = np.asarray(classes)
        sub_classes = np.zeros(num_cells, dtype=classes.dtype)
        for cell_id in range(num_cells):
            cell_labels = classes[inverse == cell_id]
            values, label_counts = np.unique(cell_labels, return_counts=True)
            sub_classes[cell_id] = values[np.argmax(label_counts)]
        outputs.append(sub_classes)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


if __name__ == "__main__":
    import time

    N = 5000
    points = np.random.rand(N, 3).astype(np.float32) * 10
    features = np.random.rand(N, 4).astype(np.float32)
    labels = np.random.randint(0, 5, size=N)

    start = time.time()
    sub_pts, sub_feat, sub_lbl = compute(points, features=features, classes=labels, sampleDl=0.5, verbose=1)
    elapsed = time.time() - start

    print(f"sub_pts shape: {sub_pts.shape}")
    print(f"sub_feat shape: {sub_feat.shape}")
    print(f"sub_lbl shape: {sub_lbl.shape}")
    print(f"Time: {elapsed:.4f}s")