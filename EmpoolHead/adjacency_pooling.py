import torch


def contract_edges(
    logical_edge_index: torch.Tensor,
    cluster: torch.Tensor,
    matched_edge_ids: torch.Tensor,
) -> torch.Tensor:
    """Contract a logical undirected COO using assignments from one forest draw.

    Pass the same [2, m] edge list used for sampling, with one column per
    undirected edge, cluster[v] the coarse ID of v, and matched_edge_ids
    the column IDs in forest.gradient[:, 1]. The result keeps each unmatched
    column in input order, including parallel edges and induced self-loops.
    Use forest.roots.numel() for the vertex count, including isolated roots.
    """
    if matched_edge_ids.numel():
        keep = torch.ones(
            logical_edge_index.size(1), dtype=torch.bool, device=logical_edge_index.device
        )
        keep.index_fill_(0, matched_edge_ids, False)
        logical_edge_index = logical_edge_index[:, keep]
    return cluster[logical_edge_index]


def logical_edges_from_bidirected(arcs: torch.Tensor) -> torch.Tensor:
    """Take one orientation of each edge in a bidirected message-passing COO.

    Requires one arc in each direction per non-loop edge and one arc per
    logical self-loop. Parallel edges retain their multiplicity. The columns
    of the result define the edge IDs for scoring, sampling and contraction.
    """
    src, dst = arcs
    return arcs[:, src <= dst]


def bidirected_arcs_from_logical(logical_edge_index: torch.Tensor) -> torch.Tensor:
    """Add a reverse message-passing arc per non-loop edge, keeping multiplicity."""
    src, dst = logical_edge_index
    reverse = logical_edge_index[:, src != dst].flip((0,))
    return torch.cat((logical_edge_index, reverse), dim=1)


def reduce_edge_indices(
    edge_index: torch.Tensor,
    num_nodes: int,
    gradient: torch.Tensor,
    *,
    return_assignments: bool = False,
) -> tuple[int, torch.Tensor] | tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Perform Morse reduction from a matching without precomputed assignments.

    Use ``contract_edges`` when Wilson has already supplied the cluster IDs.

    Args:
        edge_index (torch.Tensor): The graph representation in COO, with format [2, num_edges].
        num_nodes (int): Self-explanatory.
        gradient (torch.Tensor): The acyclic matching in [number_of_pairs, 2] format, where each row 
            represents a pair :math:`(vertex_index, edge_index).`
        return_assignments (bool): Include the root IDs and cluster assignments.

    Returns:
        The number of critical vertices and the reduced COO edge index.
        When ``return_assignments`` is true, also return ``roots`` and ``cluster``.
        ``roots[i]`` is the original vertex ID of coarse vertex ``i``.
        ``cluster[v]`` is the coarse vertex ID assigned to original vertex ``v``.
    """
    critical_mask = torch.ones(num_nodes, dtype=torch.bool, device=gradient.device)
    critical_mask.index_fill_(0, gradient[:, 0], False)

    matched_vertices, matched_edges = gradient.unbind(dim=1)
    num_pairs = gradient.size(0)

    src, dst = edge_index[:, matched_edges]
    rho = torch.arange(num_nodes, device=edge_index.device)
    rho[matched_vertices] = torch.where(matched_vertices == src, dst, src)

    rounds = max(0, num_pairs - 1).bit_length()
    for _ in range(rounds):
        rho = rho[rho]

    num_critical_vertices = num_nodes - num_pairs
    root_rank = critical_mask.cumsum(dim=0, dtype=torch.long) - 1
    cluster = root_rank[rho]

    reduced_edge_index = contract_edges(edge_index, cluster, matched_edges)

    if return_assignments:
        roots = critical_mask.nonzero(as_tuple=True)[0]
        return num_critical_vertices, reduced_edge_index, roots, cluster

    return num_critical_vertices, reduced_edge_index
