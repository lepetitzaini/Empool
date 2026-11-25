import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from EmpoolHead import _edge_dot
from EmpoolHead.scores import ContractabilityScore


def eager_scores(model, h, edges):
    src, dst = edges
    weight = model.packed_weight.index_select(0, model.weight_index).view(h.size(1), -1)
    return ((h @ weight)[src] * h[dst]).sum(-1)


class EdgeScoreTest(unittest.TestCase):
    def check_scores_and_grads(self, device, dtype, amp_dtype=None):
        n, d = 7, 17
        model = ContractabilityScore(d).to(device=device, dtype=dtype)
        h = torch.randn(d, n, device=device, dtype=dtype).T.requires_grad_()
        # Non-contiguous inputs, parallel edges, both orientations, loops, and isolates.
        edges = torch.tensor(
            [(2, 0), (4, 4), (0, 2), (2, 0), (-1, 1), (4, 4), (1, 0)],
            device=device, dtype=torch.int32,
        ).T
        with patch.object(_edge_dot, "_TILE_ELEMENTS", 2 * d):
            with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
                actual = model(h, edges)
                expected = eager_scores(model, h, edges)
            grad = torch.linspace(-1, 2, actual.numel(), device=device, dtype=actual.dtype)
            inputs = (h, model.packed_weight)
            actual_grads = torch.autograd.grad(actual, inputs, grad)
            expected_grads = torch.autograd.grad(expected, inputs, grad)
        torch.testing.assert_close(actual, expected)
        for observed, reference in zip(actual_grads, expected_grads):
            if amp_dtype is not None:
                torch.testing.assert_close(observed, reference, rtol=2e-2, atol=2e-2)
            else:
                torch.testing.assert_close(observed, reference)

    def test_scores_and_gradients(self):
        self.check_scores_and_grads("cpu", torch.float64)
        self.check_scores_and_grads("cpu", torch.float32, amp_dtype=torch.bfloat16)
        self.check_scores_and_grads("cpu", torch.bfloat16)

    def test_empty_edges(self):
        model = ContractabilityScore(5)
        for n in (0, 4):
            h = torch.randn(n, 5, requires_grad=True)
            scores = model(h, torch.empty((2, 0), dtype=torch.long))
            self.assertEqual(scores.shape, (0,))
            for grad in torch.autograd.grad(scores.sum(), (h, model.packed_weight)):
                torch.testing.assert_close(grad, torch.zeros_like(grad))

    def test_bounds_checked_before_packing(self):
        h = torch.randn(3, 4)
        # (0, 3) would otherwise have the same packed key as (1, 0).
        with patch.object(_edge_dot, "_TILE_ELEMENTS", 1):
            with self.assertRaises(IndexError):
                _edge_dot.edge_dot(h, h, torch.tensor([[0], [3]]))

    def test_large_paths_save_only_node_features(self):
        n, d, m = 13, 17, 211
        edges = torch.randint(n, (2, m))
        for dtype in (torch.float32, torch.bfloat16):
            left = torch.randn(n, d, dtype=dtype, requires_grad=True)
            right = torch.randn_like(left, requires_grad=True)
            saved = []

            def pack(tensor):
                if tensor.ndim == 2 and tensor.size(1) == d:
                    saved.append(tensor.data_ptr())
                return tensor

            with patch.object(_edge_dot, "_TILE_ELEMENTS", 3 * d):
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    scores = _edge_dot.edge_dot(left, right, edges)
                scores.sum().backward()
            self.assertEqual(set(saved), {left.data_ptr(), right.data_ptr()})

    def test_frozen_operand(self):
        edges = torch.tensor([[1, 0, 1, 3], [2, 1, 2, 3]])
        with patch.object(_edge_dot, "_TILE_ELEMENTS", 5):
            for dtype in (torch.float64, torch.bfloat16):
                for train_left in (True, False):
                    left = torch.randn(4, 7, dtype=dtype, requires_grad=train_left)
                    right = torch.randn(4, 7, dtype=dtype, requires_grad=not train_left)
                    trainable = left if train_left else right
                    actual = _edge_dot.edge_dot(left, right, edges).sum()
                    expected = (left[edges[0]] * right[edges[1]]).sum(-1).sum()
                    torch.testing.assert_close(torch.autograd.grad(actual, trainable)[0],
                                               torch.autograd.grad(expected, trainable)[0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_scores_and_gradients(self):
        self.check_scores_and_grads("cuda", torch.float32)
        self.check_scores_and_grads("cuda", torch.float32, amp_dtype=torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_peak_memory(self):
        n, d, m = 4096, 64, 131072
        model = ContractabilityScore(d).cuda()
        h = torch.randn(n, d, device="cuda", requires_grad=True)
        edges = torch.randint(n, (2, m), device="cuda")
        upstream = torch.randn(m, device="cuda")

        def peaks(score, amp):
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = score(model, h, edges)
            torch.cuda.synchronize()
            forward = torch.cuda.max_memory_allocated() - base
            torch.autograd.grad(out, (h, model.packed_weight), upstream)
            torch.cuda.synchronize()
            return forward, torch.cuda.max_memory_allocated() - base

        optimized = ContractabilityScore.forward
        for amp in (False, True):
            peaks(eager_scores, amp)
            peaks(optimized, amp)
            eager = peaks(eager_scores, amp)
            actual = peaks(optimized, amp)
            for phase, old, new in zip(("forward", "through backward"), eager, actual):
                print(f"CUDA peak {phase} above inputs (MiB), AMP={amp}  "
                      f"eager={old / 2**20:.1f}  patched={new / 2**20:.1f}")
                self.assertLess(new, old * 0.75, phase)


if __name__ == "__main__":
    unittest.main()
