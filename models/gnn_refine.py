"""Graph Neural Network refinement module for skeleton-aware keypoint correction.

Takes predicted 17x2 keypoints and refines them using the human skeleton
structure as a graph. Each keypoint is a node, each bone is an edge.
Information propagates along bones so neighboring joints constrain each other.

Residual design: output = original + learned_offset (can only help, not break).
"""

import torch
import torch.nn as nn


# COCO 17-keypoint skeleton bone connections
COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # torso
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]

# Exercise-specific joint importance weights
EXERCISE_WEIGHTS = {
    "general": {},  # All edges weight 1.0
    "squat": {
        (11, 13): 2.5, (13, 15): 2.5,  # hip-knee, knee-ankle
        (12, 14): 2.5, (14, 16): 2.5,
        (5, 11): 2.0, (6, 12): 2.0,    # shoulder-hip (back)
    },
    "pushup": {
        (5, 7): 2.5, (7, 9): 2.5,      # shoulder-elbow, elbow-wrist
        (6, 8): 2.5, (8, 10): 2.5,
        (5, 11): 2.0, (6, 12): 2.0,    # shoulder-hip (core)
    },
    "deadlift": {
        (5, 11): 3.0, (6, 12): 3.0,    # shoulder-hip (back)
        (11, 13): 2.0, (12, 14): 2.0,  # hip-knee
    },
}


def build_adjacency(num_joints=17, bones=COCO_BONES, weights=None):
    """Build adjacency matrix from bone connections with optional weights."""
    adj = torch.eye(num_joints)  # Self-loops
    for i, j in bones:
        w = 1.0
        if weights:
            w = weights.get((i, j), weights.get((j, i), 1.0))
        adj[i, j] = w
        adj[j, i] = w
    # Row-normalize
    row_sum = adj.sum(dim=1, keepdim=True)
    adj = adj / row_sum
    return adj


class GNNRefine(nn.Module):

    def __init__(self, num_joints=17, coord_dim=2, hidden_dim=64):
        super().__init__()
        self.num_joints = num_joints

        # Build adjacency matrices for each exercise
        self.register_buffer("adj_general", build_adjacency(num_joints, COCO_BONES))
        for exercise, weights in EXERCISE_WEIGHTS.items():
            if exercise != "general":
                self.register_buffer(
                    f"adj_{exercise}",
                    build_adjacency(num_joints, COCO_BONES, weights),
                )

        # 1-layer GCN: coords -> hidden -> offset
        self.input_proj = nn.Linear(coord_dim, hidden_dim)
        self.gcn_weight = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, coord_dim)
        self.relu = nn.ReLU(inplace=True)

    def get_adjacency(self, exercise="general"):
        return getattr(self, f"adj_{exercise}", self.adj_general)

    def forward(self, coords, exercise="general"):
        """
        coords: (B, 17, 2) predicted keypoint coordinates
        Returns: (B, 17, 2) refined coordinates
        """
        adj = self.get_adjacency(exercise)  # (17, 17)

        # Project to hidden dim
        h = self.relu(self.input_proj(coords))     # (B, 17, 64)

        # Graph convolution: A @ H @ W
        h = torch.matmul(adj, h)                   # (B, 17, 64) — neighbor aggregation
        h = self.relu(self.gcn_weight(h))           # (B, 17, 64)

        # Project back to coordinate space
        offset = self.output_proj(h)                # (B, 17, 2)

        # Residual — original + learned correction
        return coords + offset


def bone_length_loss(coords, target_coords):
    """Penalize when predicted bone lengths differ from ground truth.

    Encourages anatomically consistent skeleton proportions.
    """
    loss = 0.0
    for i, j in COCO_BONES:
        pred_len = torch.norm(coords[:, i] - coords[:, j], dim=-1)
        gt_len = torch.norm(target_coords[:, i] - target_coords[:, j], dim=-1)
        loss += torch.mean((pred_len - gt_len) ** 2)
    return loss / len(COCO_BONES)
