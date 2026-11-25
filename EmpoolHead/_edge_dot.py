"""Sample row-wise dot products at the edges of a COO graph."""

import torch


# Per-temporary feature-element budget, with a minimum tile of one edge.
_TILE_ELEMENTS = 1 << 20


class _SampledDot(torch.autograd.Function):
    """Use one sampled product in forward and sparse-dense products in backward."""

    @staticmethod
    def forward(ctx, left, right, crow, col):
        n = left.size(0)
        pattern = torch.sparse_csr_tensor(crow, col, left.new_ones(col.shape), (n, n))
        ctx.save_for_backward(
            left if ctx.needs_input_grad[1] else None,
            right if ctx.needs_input_grad[0] else None,
            crow, col,
        )
        with torch.autocast(left.device.type, enabled=False):
            return torch.sparse.sampled_addmm(pattern, left, right.T, beta=0).values()

    @staticmethod
    def backward(ctx, grad):
        left, right, crow, col = ctx.saved_tensors
        n = crow.numel() - 1
        # G[u,v] is the sum of the incoming scalar gradients for this pair.
        # The backward pass of index_select has already combined duplicate edges.
        g = torch.sparse_csr_tensor(crow, col, grad.contiguous(), (n, n))
        grad_left = grad_right = None
        with torch.autocast(grad.device.type, enabled=False):
            if right is not None:
                grad_left = torch.sparse.mm(g, right)
            if left is not None:
                grad_right = torch.sparse.mm(g.transpose(0, 1).to_sparse_csr(), left)
        return grad_left, grad_right, None, None


def _tiles(num_edges, width):
    step = max(1, _TILE_ELEMENTS // width)
    for start in range(0, num_edges, step):
        yield slice(start, start + step)


class _ChunkedDot(torch.autograd.Function):
    """Gather a tile again in backward so its endpoint features can be freed in forward."""

    @staticmethod
    def forward(ctx, left, right, edges):
        src, dst = edges
        ctx.shape = left.shape
        ctx.left_dtype, ctx.right_dtype = left.dtype, right.dtype
        ctx.save_for_backward(
            left if ctx.needs_input_grad[1] else None,
            right if ctx.needs_input_grad[0] else None,
            edges,
        )
        out = left.new_empty(src.shape, dtype=torch.promote_types(left.dtype, right.dtype))
        for tile in _tiles(src.numel(), left.size(1)):
            out[tile] = (left[src[tile]] * right[dst[tile]]).sum(-1)
        return out

    @staticmethod
    def backward(ctx, grad):
        left, right, edges = ctx.saved_tensors
        src, dst = edges
        grad_left = grad.new_zeros(ctx.shape, dtype=ctx.left_dtype) if right is not None else None
        grad_right = grad.new_zeros(ctx.shape, dtype=ctx.right_dtype) if left is not None else None
        for tile in _tiles(src.numel(), ctx.shape[1]):
            u, v = src[tile], dst[tile]
            scale = grad[tile, None]
            if grad_left is not None:
                values = (scale * right[v].conj()).to(ctx.left_dtype)
                grad_left.index_put_((u,), values, accumulate=True)
            if grad_right is not None:
                values = (scale * left[u].conj()).to(ctx.right_dtype)
                grad_right.index_put_((v,), values, accumulate=True)
        return grad_left, grad_right, None


def _supports_sampled_dot(left, right, num_edges):
    # With int32 CSR offsets and columns, the packed (row, col) keys fit int64.
    if max(left.size(0), num_edges) > torch.iinfo(torch.int32).max:
        return False
    # In mixed precision, PyTorch casts each edge contribution to the operand
    # dtype before summing into its vertex gradient. The tiled path preserves this.
    if left.dtype != right.dtype or left.dtype not in (torch.float32, torch.float64):
        return False
    if left.device.type == "cpu":
        return torch.backends.mkl.is_available()
    if left.device.type == "cuda" and torch.version.cuda is not None:
        if torch.are_deterministic_algorithms_enabled():
            return False
        major, minor = torch.version.cuda.split(".")[:2]
        return (int(major), int(minor)) >= (11, 3)
    return False


def _csr_indices(edge_index, n):
    # Indexing the node IDs checks bounds and normalizes negative endpoints
    # before packing them. Reuse the gathered source buffer for the pair keys.
    nodes = torch.arange(n + 1, device=edge_index.device, dtype=torch.long)
    src, dst = edge_index
    keys = nodes[:-1][src].mul_(n).add_(nodes[:-1][dst])
    keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    # Row u starts at the first key >= u*n, including for empty rows.
    crow = torch.searchsorted(keys, nodes.mul_(n), out_int32=True)
    col = keys.remainder(n).to(torch.int32)
    return crow, col, inverse


def edge_dot(left, right, edge_index):
    """Return dot(left[u], right[v]) in input edge order, including parallel edges."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must contain int32 or int64 vertex indices")

    n, width = left.shape
    m = edge_index.size(1)
    if m * width <= _TILE_ELEMENTS:
        src, dst = edge_index
        return (left[src] * right[dst]).sum(-1)

    edge_index = edge_index.to(left.device)
    if not _supports_sampled_dot(left, right, m):
        return _ChunkedDot.apply(left, right, edge_index)

    crow, col, inverse = _csr_indices(edge_index, n)
    scores = _SampledDot.apply(left, right, crow, col)
    return scores.index_select(0, inverse)
