"""
CPU-only replacement for RandLA-Net's compiled `nearest_neighbors` module.

The original repo uses a C++ extension (nanoflann + Cython) for fast batched
KNN search. That extension needs a C++ compiler with OpenMP support set up
correctly (MSVC flags differ from the gcc flags in the repo's setup.py),
which is painful on Windows and unnecessary for our CPU-only prototype.

This module reproduces the same `knn_batch(pc, query, k, omp=True)` interface
using scipy's cKDTree, so it's a drop-in replacement — nothing calling it
needs to change.
"""

import numpy as np
from scipy.spatial import cKDTree


def knn_batch(support_pts, query_pts, k, omp=True):
    support_pts = np.asarray(support_pts, dtype=np.float32)
    query_pts = np.asarray(query_pts, dtype=np.float32)

    batch_size = support_pts.shape[0]

    all_indices = []
    for b in range(batch_size):
        tree = cKDTree(support_pts[b])
        _, idx = tree.query(query_pts[b], k=k)
        if k == 1:
            idx = idx[:, None]
        all_indices.append(idx.astype(np.int32))

    return np.stack(all_indices, axis=0)


if __name__ == "__main__":
    import time

    batch_size = 4
    num_points = 2000
    K = 16

    pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

    start = time.time()
    neigh_idx = knn_batch(pc, pc, K, omp=True)
    elapsed = time.time() - start

    print(f"Output shape: {neigh_idx.shape}  (expected: ({batch_size}, {num_points}, {K}))")
    print(f"Time: {elapsed:.4f}s for {batch_size} batches of {num_points} points, k={K}")