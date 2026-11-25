"""Weighted Wilson sampling. Build the CPU extension once with
``python setup.py build_ext --inplace`` in the repository root.
"""

from importlib import import_module
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


class WilsonTopology:
    """Reusable CSR for one logical undirected COO.

    Keep one instance while the vertex numbering, edge IDs and endpoints stay fixed.
    Each parallel edge has its own ID. Self-loops never enter the forest. Optional
    PyG-style ptr specifies contiguous vertex ranges for independent graphs.
    """

    def __init__(
        self,
        logical_edge_index: torch.Tensor,
        num_nodes: int,
        *,
        ptr: torch.Tensor | None = None,
    ):
        if (
            logical_edge_index.dtype != torch.long
            or logical_edge_index.ndim != 2
            or logical_edge_index.size(0) != 2
        ):
            raise ValueError("Expected logical_edge_index as a long tensor of shape [2, num_edges].")
        if ptr is not None and (ptr.dtype != torch.long or ptr.ndim != 1):
            raise ValueError("ptr must be a one-dimensional long tensor.")
        name = f"{__package__}._wilson_cpu" if __package__ else "_wilson_cpu"
        try:
            native = import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            raise ImportError(
                "Build the Wilson extension from the repository root with "
                "'python setup.py build_ext --inplace'."
            ) from exc

        edges = logical_edge_index.detach().cpu().resolve_neg().contiguous()
        bounds = None if ptr is None else ptr.detach().cpu().resolve_neg().contiguous()
        self._csr = native.Topology(edges, num_nodes, bounds)
        self._num_nodes = num_nodes
        self._num_edges = logical_edge_index.size(1)
        self._device = logical_edge_index.device

    def prepare(
        self, vertex_scores: torch.Tensor, edge_scores: torch.Tensor
    ) -> "WilsonTransitions":
        """Compute the weight table once for the current logits. Reuse it across draws."""
        if vertex_scores.shape != (self._num_nodes,) or edge_scores.shape != (self._num_edges,):
            raise ValueError("Expected vertex_scores [num_nodes] and edge_scores [num_edges].")
        if not vertex_scores.is_floating_point() or not edge_scores.is_floating_point():
            raise ValueError("Vertex and edge scores must be floating-point tensors.")
        table = self._csr.prepare(_host_logits(vertex_scores), _host_logits(edge_scores))
        return WilsonTransitions(table, self._device)


def _host_logits(scores: torch.Tensor) -> torch.Tensor:
    # Transfer in the original dtype. The native CDF accumulates in float64.
    scores = scores.detach().cpu().resolve_neg()
    if scores.dtype not in (torch.float32, torch.float64):
        scores = scores.float()
    return scores.contiguous()


class WilsonTransitions:
    """Weighted transitions for independent draws at a fixed layer state."""

    def __init__(self, table, device: torch.device):
        self._table = table
        self._device = device

    def sample(self, *, generator: torch.Generator | None = None) -> ForestAssignments:
        """Draw one forest with reproducible per-graph streams from a CPU generator."""
        seeds = torch.randint(
            2**63 - 1, (self._table.num_graphs,),
            dtype=torch.long, device="cpu", generator=generator,
        )
        matching, roots, cluster = self._table.sample(seeds)
        return ForestAssignments(
            matching.to(self._device, non_blocking=True),
            roots.to(self._device, non_blocking=True),
            cluster.to(self._device, non_blocking=True),
        )


@torch.no_grad()
def sample_forest_assignments(
    logical_edge_index: torch.Tensor,
    vertex_scores: torch.Tensor,
    edge_scores: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    ptr: torch.Tensor | None = None,
) -> ForestAssignments:
    """Sample from the logical undirected COO using compiled CPU Wilson walks.

    All outputs are long tensors on logical_edge_index.device. Pass batch.ptr to
    parallelize independent graphs in a batch. To reuse both CSR and weights for
    a primary draw and an auxiliary draw, prepare one transition table.

        topology = WilsonTopology(logical_edge_index, vertex_scores.numel(), ptr=batch.ptr)
        transitions = topology.prepare(vertex_scores, edge_scores)
        forest = transitions.sample(generator=generator)
        auxiliary = transitions.sample(generator=generator)

    Prepare again when the scores change. Construct a new topology after contraction.
    """
    if vertex_scores.ndim != 1:
        raise ValueError("Expected vertex_scores [num_nodes].")
    topology = WilsonTopology(logical_edge_index, vertex_scores.numel(), ptr=ptr)
    return topology.prepare(vertex_scores, edge_scores).sample(generator=generator)


def sample_rooted_forest(
    edge_index: torch.Tensor,
    vertex_scores: torch.Tensor,
    edge_scores: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    ptr: torch.Tensor | None = None,
) -> RootedForest:
    """Sample the full forest, including its child-to-parent COO (original API).

    Inputs and generator follow ``sample_forest_assignments``. Use that function
    when the matching, roots, and cluster IDs suffice for pooling.
    """
    forest = sample_forest_assignments(
        edge_index, vertex_scores, edge_scores, generator=generator, ptr=ptr
    )
    return RootedForest(
        edge_index=forest.as_edge_index(edge_index),
        gradient=forest.gradient,
        roots=forest.roots,
        cluster=forest.cluster,
    )
