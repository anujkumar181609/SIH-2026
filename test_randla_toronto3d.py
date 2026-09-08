"""
Step 2 (RandLA-Net half): real-data structural test.

Loads a random chunk of real points from a Toronto-3D tile, applies the
UTM_OFFSET fix documented in the dataset's own README (avoids float
precision loss), and builds the actual multi-scale KNN + subsampling
structure RandLA-Net expects -- using REAL computed neighbor indices via
our knn_cpu.py fallback, not random/synthetic ones like the earlier
architecture-only smoke test.

IMPORTANT CAVEAT (flagging clearly, not hiding it):
Our checkpoint was trained on SemanticKITTI's 19 classes (outdoor driving
scenes). Toronto-3D uses a different 9-class taxonomy (0=unclassified,
1-8=Road/RoadMarking/Natural/Building/UtilityLine/Pole/Car/Fence). So the
model's predicted class IDs here do NOT correspond to Toronto-3D's ground
truth labels -- there is no meaningful accuracy/mIoU to compute from this
run. This script only confirms: real point cloud data can be ingested,
correctly preprocessed through our CPU KNN/subsampling code, and pushed
through the model to produce correctly-shaped output without errors.
"""

import sys
import os
import numpy as np
import torch
from plyfile import PlyData

sys.path.append(os.path.join(os.path.dirname(__file__), 'randla_net'))

from helper_tool import ConfigSemanticKITTI as cfg
from helper_tool import DataProcessing as DP
from RandLANet import Network

PLY_PATH = os.path.join("data", "toronto_3d", "L001.ply")
CHECKPOINT_PATH = os.path.join("randla_net", "output", "checkpoint.tar")
DEVICE = torch.device("cpu")

UTM_OFFSET = np.array([627285.0, 4841948.0, 0.0])

TEST_NUM_POINTS = 4096


def load_real_chunk():
    print(f"Reading {PLY_PATH} ...")
    ply = PlyData.read(PLY_PATH)
    vertex = ply['vertex']
    total_points = len(vertex['x'])
    print(f"Tile has {total_points:,} total points. Sampling {TEST_NUM_POINTS} at random.")

    rng = np.random.default_rng(seed=42)
    idx = rng.choice(total_points, size=TEST_NUM_POINTS, replace=False)

    xyz = np.stack([vertex['x'][idx], vertex['y'][idx], vertex['z'][idx]], axis=-1).astype(np.float64)
    xyz = (xyz - UTM_OFFSET).astype(np.float32)

    labels = vertex['scalar_Label'][idx].astype(np.int32)

    return xyz, labels


def build_multiscale_inputs(xyz):
    import knn_cpu

    end_points = {"xyz": [], "neigh_idx": [], "sub_idx": [], "interp_idx": []}

    current_xyz = xyz[np.newaxis, :, :]  # (1, N, 3)
    layer_xyz_list = [current_xyz]

    for i in range(cfg.num_layers):
        n = current_xyz.shape[1]
        neigh_idx = knn_cpu.knn_batch(current_xyz, current_xyz, cfg.k_n)
        sub_n = n // cfg.sub_sampling_ratio[i]
        sub_xyz = current_xyz[:, :sub_n, :]
        sub_idx = neigh_idx[:, :sub_n, :]

        interp_idx = knn_cpu.knn_batch(sub_xyz, current_xyz, 1)

        end_points["neigh_idx"].append(torch.from_numpy(neigh_idx).long())
        end_points["sub_idx"].append(torch.from_numpy(sub_idx).long())
        end_points["interp_idx"].append(torch.from_numpy(interp_idx).long())

        current_xyz = sub_xyz
        layer_xyz_list.append(current_xyz)

    end_points["xyz"] = [torch.from_numpy(p).float() for p in layer_xyz_list[:-1]]

    return end_points


def build_model():
    model = Network(cfg)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def main():
    xyz, gt_labels = load_real_chunk()
    print(f"Ground-truth Toronto-3D label distribution in this sample: "
          f"{dict(zip(*np.unique(gt_labels, return_counts=True)))}")

    print("Building real multi-scale KNN/subsampling structure (this uses actual "
          "geometry, may take a little while on CPU)...")
    end_points = build_multiscale_inputs(xyz)

    features = xyz[np.newaxis, :, :]  # (1, N, 3)
    end_points["features"] = torch.from_numpy(features).transpose(1, 2).float()  # (1, 3, N)

    model = build_model()
    with torch.no_grad():
        output = model(end_points)

    logits = output["logits"]
    print(f"\nOutput logits shape: {tuple(logits.shape)}")
    print(f"Expected: (1, {cfg.num_classes}, {TEST_NUM_POINTS})")

    pred_classes = torch.argmax(logits, dim=1)
    print(f"Predicted SemanticKITTI class range: {pred_classes.min().item()} to {pred_classes.max().item()}")
    print("\nReminder: predicted class IDs are SemanticKITTI's taxonomy (19 classes, "
          "outdoor driving), not Toronto-3D's (9 classes) -- this run validates the "
          "pipeline structurally on real data, not classification accuracy.")


if __name__ == "__main__":
    main()