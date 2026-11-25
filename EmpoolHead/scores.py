import torch
import torch.nn as nn

from ._edge_dot import edge_dot


class RetentionScore(nn.Module):
    def __init__(self, in_dimension: int):
        super().__init__()
        self.linear = nn.Linear(in_dimension, 1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
    

class ContractabilityScore(nn.Module):
    def __init__(self, in_dimension: int):
        super().__init__()
        self.in_dimension = in_dimension
        indices = torch.arange(self.in_dimension)
        low = torch.minimum(indices[:, None], indices[None, :])
        high = torch.maximum(indices[:, None], indices[None, :])

        lookup = low * self.in_dimension - low * (low + 1) // 2 + high

        self.register_buffer("weight_index", lookup.flatten().int(), persistent=False)
        self.packed_weight = nn.Parameter(torch.empty(self.in_dimension * (self.in_dimension + 1) // 2))
        nn.init.normal_(self.packed_weight, std=1.0 / self.in_dimension)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Score each edge from node embeddings h of shape [num_nodes, in_dimension]."""
        d = self.in_dimension
        weight = self.packed_weight.index_select(0, self.weight_index).view(d, d)
        projected = h @ weight
        return edge_dot(projected, h, edge_index)
