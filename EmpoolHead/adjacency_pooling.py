import torch


def reduce_edge_indices(
        edge_index:torch.Tensor,
        num_nodes:int,
        gradient:torch.Tensor
) -> tuple[int, torch.Tensor]:
    r"""This function performs Morse reduction on the graph, provided the 
    expected representation and gradients.

    Args:
        edge_index (torch.Tensor): The graph representation in COO, with format [2, num_edges].
        num_nodes (int): Self-explanatory.
        gradient (torch.Tensor): The acyclic matching in [number_of_pairs, 2] format, where each row 
            represents a pair :math:`(vertex_index, edge_index).`

    Returns:
        tuple[int, torch.Tensor]: A tuple where the first component is the number of critical vertices, and 
            the second is the COO representation of the reduced adjacency matrix.
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

    keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    keep.index_fill_(0, matched_edges, False)
    reduced_edge_index = cluster[edge_index[:, keep]]

    return num_critical_vertices, reduced_edge_index
