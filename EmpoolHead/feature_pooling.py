import torch
import torch.nn as nn


class FeaturePooling(nn.Module):
    """Pool node embeddings with a root-aware Deep Sets aggregation."""

    def __init__(self, in_dimension: int):
        super().__init__()
        self.chi = nn.Sequential(
            nn.Linear(in_dimension, in_dimension),
            nn.ReLU(),
            nn.Linear(in_dimension, in_dimension),
        )
        self.phi = nn.Sequential(
            nn.Linear(2 * in_dimension, in_dimension),
            nn.ReLU(),
            nn.Linear(in_dimension, in_dimension),
        )

    def forward(
        self, h: torch.Tensor, roots: torch.Tensor, cluster: torch.Tensor
    ) -> torch.Tensor:
        """Map pre-pooling embeddings h to one feature row per coarse vertex.

        roots[i] is the original vertex ID of coarse vertex i.
        cluster[v] is the coarse vertex ID assigned to original vertex v.
        Both index tensors are one-dimensional and on the same device as h.
        """
        member_features = self.chi(h)
        member_features.index_fill_(0, roots, 0.0)

        member_sums = member_features.new_zeros((roots.numel(), member_features.size(1)))
        member_sums.index_add_(0, cluster, member_features)

        root_features = h.index_select(0, roots)
        return self.phi(torch.cat((root_features, member_sums), dim=-1))
