from bisect import bisect_right
from itertools import accumulate, chain
from math import exp, isfinite
from random import Random
from typing import NamedTuple

import torch


class RootedForest(NamedTuple):
    """A sampled forest and its assignments for graph and feature pooling.

    edge_index has shape [2, num_forest_edges] and points from child to parent.
    gradient has shape [num_forest_edges, 2], with rows (child, input_edge_id).
    roots[i] is the original vertex ID of coarse vertex i, in ascending order.
    cluster[v] is the coarse vertex ID assigned to original vertex v.
    """

    edge_index: torch.Tensor
    gradient: torch.Tensor
    roots: torch.Tensor
    cluster: torch.Tensor


class ForestAssignments(NamedTuple):
    """The three index tensors needed for pooling.

    gradient has rows (child, logical edge ID). roots[i] is the original
    vertex ID of coarse vertex i, and cluster[v] is the coarse ID of vertex v.
    """

    gradient: torch.Tensor
    roots: torch.Tensor
    cluster: torch.Tensor

    def as_edge_index(self, logical_edge_index: torch.Tensor) -> torch.Tensor:
        """Materialize the child-to-parent COO from the original logical edge list."""
        children, edge_ids = self.gradient.unbind(dim=1)
        src, dst = logical_edge_index.index_select(1, edge_ids)
        parents = torch.where(children == src, dst, src)
        return torch.stack((children, parents))


@torch.no_grad()
def sample_forest_assignments(
    logical_edge_index: torch.Tensor,
    vertex_scores: torch.Tensor,
    edge_scores: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> ForestAssignments:
    """Sample Wilson's forest once for adjacency pooling and feature pooling.

    logical_edge_index is a long tensor of shape [2, num_edges], with one column per
    logical undirected edge. Parallel edges have separate columns. Self-loops
    are skipped during sampling. The finite vertex_scores and edge_scores are
    raw logits with shapes [num_nodes] and [num_edges], respectively.

    The random walks run on the CPU. The optional generator is a CPU PyTorch
    generator. All output tensors are long tensors on logical_edge_index.device.
    Example for a batch of disjoint graphs

        forest = sample_forest_assignments(logical_edge_index, vertex_scores, edge_scores)
        edges = contract_edges(logical_edge_index, forest.cluster, forest.gradient[:, 1])
        features = feature_pool(h, forest.roots, forest.cluster)
        pooled_batch = batch[forest.roots]
    """
    if (
        logical_edge_index.ndim != 2
        or logical_edge_index.size(0) != 2
        or logical_edge_index.dtype != torch.long
    ):
        raise ValueError("logical_edge_index must be a long tensor of shape [2, num_edges].")
    if vertex_scores.ndim != 1 or edge_scores.shape != (logical_edge_index.size(1),):
        raise ValueError("Expected vertex_scores [num_nodes] and edge_scores [num_edges].")
    if not vertex_scores.is_floating_point() or not edge_scores.is_floating_point():
        raise ValueError("Vertex and edge scores must be floating-point tensors.")

    src, dst = logical_edge_index.tolist()
    root_logits = vertex_scores.tolist()
    edge_logits = edge_scores.tolist()
    if not all(map(isfinite, chain(root_logits, edge_logits))):
        raise ValueError("Vertex and edge scores must be finite.")

    num_nodes = len(root_logits)
    incident_edges: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_id, (u, v) in enumerate(zip(src, dst)):
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"Input edge {edge_id} has an endpoint outside the graph.")
        if u != v:
            incident_edges[u].append(edge_id)
            incident_edges[v].append(edge_id)

    # At each vertex, choice zero is the edge to the virtual root. Shift its
    # logit and the incident edge logits by the same amount before exp.
    cumulative_weights = []
    for vertex, incident in enumerate(incident_edges):
        root_logit = root_logits[vertex]
        edge_max = max((edge_logits[e] for e in incident), default=root_logit)
        shift = max(root_logit, edge_max)
        root_weight = exp(root_logit - shift)
        if root_weight == 0.0:
            raise FloatingPointError(f"Root transition at vertex {vertex} underflowed.")
        weights = (exp(edge_logits[e] - shift) for e in incident)
        cumulative_weights.append(list(accumulate(weights, initial=root_weight)))

    seed = torch.randint(2**63 - 1, (), device="cpu", generator=generator).item()
    draw_uniform = Random(seed).random
    virtual_root = num_nodes
    root_of = [-1] * num_nodes + [virtual_root]
    parent = [-1] * num_nodes
    parent_edge = [-1] * num_nodes

    for start in range(num_nodes):
        if root_of[start] >= 0:
            continue

        vertex = start
        last_vertex = start
        while root_of[vertex] < 0:
            cumulative = cumulative_weights[vertex]
            choice = bisect_right(cumulative, draw_uniform() * cumulative[-1])
            if choice == 0:
                next_vertex, edge_id = virtual_root, -1
            else:
                edge_id = incident_edges[vertex][choice - 1]
                next_vertex = dst[edge_id] if src[edge_id] == vertex else src[edge_id]

            # Overwrite earlier exits from this vertex. Following the final
            # exits from start gives the loop-erased path.
            parent[vertex] = next_vertex
            parent_edge[vertex] = edge_id
            last_vertex, vertex = vertex, next_vertex

        root = last_vertex if vertex == virtual_root else root_of[vertex]
        vertex = start
        while root_of[vertex] < 0:
            root_of[vertex] = root
            vertex = parent[vertex]

    roots = [v for v in range(num_nodes) if parent[v] == virtual_root]
    root_rank = {root: rank for rank, root in enumerate(roots)}
    cluster = [root_rank[root_of[v]] for v in range(num_nodes)]
    matching = [(v, edge_id) for v, edge_id in enumerate(parent_edge) if edge_id >= 0]

    device = logical_edge_index.device
    return ForestAssignments(
        gradient=torch.tensor(matching, dtype=torch.long, device=device).reshape(-1, 2),
        roots=torch.tensor(roots, dtype=torch.long, device=device),
        cluster=torch.tensor(cluster, dtype=torch.long, device=device),
    )


def sample_rooted_forest(
    edge_index: torch.Tensor,
    vertex_scores: torch.Tensor,
    edge_scores: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> RootedForest:
    """Sample the full forest, including its child-to-parent COO (original API).

    Inputs and generator follow ``sample_forest_assignments``. Use that function
    when the matching, roots, and cluster IDs suffice for pooling.
    """
    forest = sample_forest_assignments(
        edge_index, vertex_scores, edge_scores, generator=generator
    )
    return RootedForest(
        edge_index=forest.as_edge_index(edge_index),
        gradient=forest.gradient,
        roots=forest.roots,
        cluster=forest.cluster,
    )
