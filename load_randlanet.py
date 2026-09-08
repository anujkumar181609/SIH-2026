"""
Load the repo's own pretrained RandLA-Net (SemanticKITTI, 19 classes)
checkpoint and run a single inference pass on a small synthetic point
cloud, to confirm the model + CPU fallback ops work end to end.
"""

import sys
import os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'randla_net'))

from helper_tool import ConfigSemanticKITTI as cfg
from RandLANet import Network

CHECKPOINT_PATH = os.path.join("randla_net", "output", "checkpoint.tar")
DEVICE = torch.device("cpu")


def build_model():
    model = Network(cfg)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, "
          f"loss={checkpoint.get('loss', 'unknown')}")
    return model


def make_synthetic_input(num_points):
    """
    xyz-only input (3 channels), matching this checkpoint's fc0.conv.weight
    shape of (8, 3, 1) -- no RGB, unlike the earlier S3DIS attempt.
    """
    xyz = np.random.rand(1, num_points, 3).astype(np.float32)
    features = xyz.copy()  # (1, N, 3)

    end_points = {
        "xyz": [],
        "neigh_idx": [],
        "sub_idx": [],
        "interp_idx": [],
        "features": torch.from_numpy(features).transpose(1, 2).float(),  # (1, 3, N)
    }

    current_n = num_points
    layer_points = [xyz[0]]
    for i in range(cfg.num_layers):
        neigh_idx = np.random.randint(0, current_n, size=(1, current_n, cfg.k_n)).astype(np.int32)
        sub_n = current_n // cfg.sub_sampling_ratio[i]
        sub_idx = neigh_idx[:, :sub_n, :]
        interp_idx = np.random.randint(0, sub_n, size=(1, current_n, 1)).astype(np.int32)

        end_points["neigh_idx"].append(torch.from_numpy(neigh_idx).long())
        end_points["sub_idx"].append(torch.from_numpy(sub_idx).long())
        end_points["interp_idx"].append(torch.from_numpy(interp_idx).long())

        current_n = sub_n
        layer_points.append(layer_points[-1][:sub_n])

    end_points["xyz"] = [torch.from_numpy(p).unsqueeze(0).float() for p in layer_points[:-1]]

    return end_points


def main():
    model = build_model()

    # Using a smaller point count than cfg.num_points (usually 45056 for
    # SemanticKITTI) for a fast CPU structural smoke test.
    test_num_points = 2000
    end_points = make_synthetic_input(test_num_points)

    with torch.no_grad():
        output = model(end_points)

    logits = output["logits"]
    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Expected: (1, {cfg.num_classes}, {test_num_points})")

    pred_classes = torch.argmax(logits, dim=1)
    print(f"Predicted class range: {pred_classes.min().item()} to {pred_classes.max().item()}")
    print(f"(Should fall within 0-{cfg.num_classes - 1} for the 19 SemanticKITTI classes)")


if __name__ == "__main__":
    main()