import sys
import unittest
from pathlib import Path

import torch

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from EmpoolHead.forest_sampling import (
    WilsonTopology,
    sample_forest_assignments,
    sample_rooted_forest,
)


class WilsonCSRTest(unittest.TestCase):
    def test_weighted_forest_law_with_parallel_edges(self):
        # Each two-vertex graph has root masses (2, 1), edge masses (1, 3),
        # and a high-scoring self-loop which cannot enter a forest.
        draws = 16384
        u = 2 * torch.arange(draws)
        src = torch.stack((u, u + 1, u), dim=1).flatten()
        dst = torch.stack((u + 1, u, u), dim=1).flatten()
        ptr = torch.arange(0, 2 * draws + 1, 2)
        a = torch.tensor([2., 1.], dtype=torch.float64).log().repeat(draws) + 1000.
        b = torch.tensor([1., 3., 1.], dtype=torch.float64).log().repeat(draws) + 1000.
        b[2::3] = 10000.
        topo = WilsonTopology(torch.stack((src, dst)), a.numel(), ptr=ptr)
        forest = topo.prepare(a, b).sample(generator=torch.Generator().manual_seed(6841))

        child, edge_id = forest.gradient.T
        self.assertTrue(torch.equal(child // 2, edge_id // 3))
        self.assertFalse(torch.any(edge_id % 3 == 2))
        self.assertEqual((child // 2).unique().numel(), child.numel())
        outcome = torch.zeros(draws, dtype=torch.long)
        root = 1 - child % 2
        outcome[child // 2] = 1 + 2 * root + edge_id % 3
        observed = outcome.bincount(minlength=5).double() / draws
        # Both roots, then (root 0, edge 0/1), then (root 1, edge 0/1).
        expected = torch.tensor([2, 2, 6, 1, 3], dtype=torch.float64) / 14.
        torch.testing.assert_close(observed, expected, atol=0.02, rtol=0)

    def test_prepared_draws_match_one_shot_and_parallel_paths(self):
        edges = torch.tensor([[1, 0, 2, 3, 2], [0, 1, 3, 4, 2]])
        ptr = torch.tensor([0, 0, 2, 5, 5])
        a, b = torch.zeros(5), torch.zeros(5)
        topo = WilsonTopology(edges, 5, ptr=ptr)
        table = topo.prepare(a, b)
        first = torch.Generator().manual_seed(79)
        replay = torch.Generator().manual_seed(79)
        for _ in range(4):
            compact = table.sample(generator=first)
            full = sample_rooted_forest(edges, a, b, ptr=ptr, generator=replay)
            for actual, expected in zip(compact, full[1:]):
                torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(compact.as_edge_index(edges), full.edge_index)

        # Extra isolated vertices trigger the parallel path without changing the
        # random decisions on the original vertices.
        ptr_padded = ptr.clone()
        ptr_padded[-2:] = 4101
        padded = sample_forest_assignments(
            edges, torch.zeros(4101), b, ptr=ptr_padded, generator=first
        )
        small = table.sample(generator=replay)
        torch.testing.assert_close(padded.gradient, small.gradient)
        torch.testing.assert_close(padded.cluster[:5], small.cluster)
        torch.testing.assert_close(padded.roots[:small.roots.numel()], small.roots)

    def test_topology_and_weights_own_their_data(self):
        edges = torch.tensor([[0, 1], [1, 2]])
        a, b = torch.zeros(3), torch.ones(2, dtype=torch.float64)
        ptr = torch.tensor([0, 3])
        table = WilsonTopology(edges, 3, ptr=ptr).prepare(a, b)
        reference = table.sample(generator=torch.Generator().manual_seed(317))
        edges.zero_()
        ptr.zero_()
        a.fill_(float("nan"))
        b.fill_(float("nan"))
        after = table.sample(generator=torch.Generator().manual_seed(317))
        for actual, expected in zip(after, reference):
            torch.testing.assert_close(actual, expected)

    def test_invalid_native_inputs(self):
        with self.assertRaises(ValueError):
            WilsonTopology(torch.tensor([[0], [2]]), 3, ptr=torch.tensor([0, 1, 3]))
        with self.assertRaises(ValueError):
            WilsonTopology(torch.empty(2, 0, dtype=torch.long), 2, ptr=torch.tensor([0, 3, 2]))
        with self.assertRaises(ValueError):
            WilsonTopology(torch.tensor([[-1], [0]]), 2)
        topology = WilsonTopology(torch.tensor([[0, 1], [1, 1]]), 2)
        with self.assertRaises(ValueError):
            topology.prepare(torch.zeros(2), torch.tensor([0., float("nan")]))
        with self.assertRaises(FloatingPointError):
            topology.prepare(torch.tensor([-1000., 1.]), torch.zeros(2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_inputs_return_the_same_draw_on_cuda(self):
        edges = torch.tensor([[0, 0, 2], [1, 1, 2]])
        a, b = torch.zeros(3), torch.ones(3)
        cpu = sample_forest_assignments(edges, a, b, generator=torch.Generator().manual_seed(5))
        gpu = sample_forest_assignments(
            edges.cuda(), a.cuda(), b.cuda(), generator=torch.Generator().manual_seed(5)
        )
        for actual, expected in zip(gpu, cpu):
            self.assertTrue(actual.is_cuda)
            torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
