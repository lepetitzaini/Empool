import unittest

import torch

from EmpoolHead.adjacency_pooling import (
    bidirected_arcs_from_logical,
    contract_edges,
    logical_edges_from_bidirected,
    reduce_edge_indices,
)
from EmpoolHead.forest_sampling import (
    ForestAssignments,
    sample_forest_assignments,
    sample_rooted_forest,
)


class ForestContractionTest(unittest.TestCase):
    def test_only_matched_edge_ids_are_removed(self):
        edges = torch.tensor([[1, 0, 2, 3, 0, 4, 4, 2], [0, 1, 1, 4, 3, 1, 4, 0]])
        forest = ForestAssignments(
            gradient=torch.tensor([[0, 0], [1, 2], [3, 3]]),
            roots=torch.tensor([2, 4, 5]),
            cluster=torch.tensor([0, 0, 0, 1, 1, 2]),
        )
        coarse = contract_edges(edges, forest.cluster, forest.gradient[:, 1])
        # One parallel edge survives as a loop. Vertex 5 remains an isolated root.
        expected_edges = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]])
        expected_parents = torch.tensor([[0, 1, 3], [1, 2, 4]])
        torch.testing.assert_close(coarse, expected_edges)
        torch.testing.assert_close(forest.as_edge_index(edges), expected_parents)
        n, legacy_edges, roots, cluster = reduce_edge_indices(
            edges, 6, forest.gradient, return_assignments=True
        )
        self.assertEqual(n, forest.roots.numel())
        torch.testing.assert_close(legacy_edges, coarse)
        torch.testing.assert_close(roots, forest.roots)
        torch.testing.assert_close(cluster, forest.cluster)

    def test_compact_draw_agrees_with_full_forest(self):
        empty = torch.empty((2, 0), dtype=torch.long)
        cases = [(0, empty), (3, empty), (5, torch.tensor([[1, 0, 2, 2], [0, 1, 1, 2]]))]
        for n, edges in cases:
            with self.subTest(n=n):
                args = edges, torch.zeros(n), torch.zeros(edges.size(1))
                forest = sample_forest_assignments(
                    *args, generator=torch.Generator().manual_seed(17)
                )
                full = sample_rooted_forest(*args, generator=torch.Generator().manual_seed(17))
                torch.testing.assert_close(full.edge_index, forest.as_edge_index(edges))
                for compact, legacy in zip(forest, full[1:]):
                    torch.testing.assert_close(compact, legacy)
                n_coarse, legacy_edges = reduce_edge_indices(edges, n, forest.gradient)
                self.assertEqual(n_coarse, forest.roots.numel())
                torch.testing.assert_close(
                    contract_edges(edges, forest.cluster, forest.gradient[:, 1]), legacy_edges
                )

    def test_gcn_arcs_keep_loops_and_parallel_multiplicity(self):
        logical = torch.tensor([[0, 0, 2], [1, 1, 2]])
        arcs = bidirected_arcs_from_logical(logical)
        torch.testing.assert_close(arcs, torch.tensor([[0, 0, 2, 1, 1], [1, 1, 2, 0, 0]]))
        torch.testing.assert_close(logical_edges_from_bidirected(arcs), logical)
