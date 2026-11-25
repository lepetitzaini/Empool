#include <c10/macros/Macros.h>

// System PyTorch can include glog without passing its CMake consumer definition.
#if defined(C10_USE_GLOG) && !defined(GLOG_USE_GLOG_EXPORT) && __has_include(<glog/export.h>)
#define GLOG_USE_GLOG_EXPORT 1
#endif

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using Index = int64_t;
using at::Tensor;

struct RootUnderflow : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

bool is_cpu_array(const Tensor& t, int dimensions) {
    return t.device().is_cpu() && t.layout() == at::kStrided &&
           t.is_contiguous() && !t.is_neg() && t.dim() == dimensions;
}

struct Topology {
    Index n, m;
    std::vector<Index> boundaries, row;
    Tensor neighbor, edge_id;

    Topology(const Tensor& edges, Index num_nodes, const std::optional<Tensor>& ptr)
        : n(num_nodes), m(0) {
        require(n >= 0, "num_nodes must be nonnegative.");
        require(is_cpu_array(edges, 2) && edges.scalar_type() == at::kLong && edges.size(0) == 2,
                "Expected a CPU long COO of shape [2, num_edges].");
        m = edges.size(1);
        if (ptr.has_value()) {
            const Tensor& p = *ptr;
            require(is_cpu_array(p, 1) && p.scalar_type() == at::kLong && p.numel() > 0,
                    "ptr must be a nonempty one-dimensional CPU long tensor.");
            const Index* data = p.const_data_ptr<Index>();
            boundaries.assign(data, data + p.numel());
            require(boundaries.front() == 0 && boundaries.back() == n &&
                    std::is_sorted(boundaries.begin(), boundaries.end()),
                    "ptr must be nondecreasing from zero to num_nodes.");
        } else {
            boundaries = {0, n};
        }

        row.resize(static_cast<size_t>(n) + 1, 0);
        auto ends = edges.accessor<Index, 2>();
        for (Index e = 0; e < m; ++e) {
            const Index u = ends[0][e], v = ends[1][e];
            require(u >= 0 && u < n && v >= 0 && v < n, "A logical edge endpoint is out of range.");
            if (u != v) {
                ++row[u + 1];
                ++row[v + 1];
            }
        }
        std::partial_sum(row.begin(), row.end(), row.begin());
        neighbor = at::empty({row.back()}, edges.options());
        edge_id = at::empty_like(neighbor);
        auto* dst = neighbor.data_ptr<Index>();
        auto* ids = edge_id.data_ptr<Index>();
        // Reuse row starts as insertion cursors, then restore the offsets.
        for (Index e = 0; e < m; ++e) {
            const Index u = ends[0][e], v = ends[1][e];
            if (u != v) {
                const Index i = row[u]++, j = row[v]++;
                dst[i] = v;
                dst[j] = u;
                ids[i] = ids[j] = e;
            }
        }
        std::copy_backward(row.begin(), row.end() - 1, row.end());
        row.front() = 0;

        if (boundaries.size() > 2) {
            for (size_t g = 1; g < boundaries.size(); ++g) {
                const Index first = boundaries[g - 1], stop = boundaries[g];
                for (Index i = row[first]; i < row[stop]; ++i) {
                    require(first <= dst[i] && dst[i] < stop, "A logical edge crosses a boundary in ptr.");
                }
            }
        }
    }

    Index graphs() const { return static_cast<Index>(boundaries.size()) - 1; }
};

// SplitMix64. Each graph owns its state, irrespective of the CPU worker it runs on.
double uniform01(uint64_t& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t x = state;
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return static_cast<double>((x ^ (x >> 31)) >> 11) * (1.0 / 9007199254740992.0);
}

struct Transitions {
    std::shared_ptr<const Topology> graph;
    Tensor root_mass, cumulative;

    Transitions(std::shared_ptr<const Topology> topology, const Tensor& a, const Tensor& b)
        : graph(std::move(topology)) {
        auto valid_logits = [](const Tensor& t, Index size) {
            return is_cpu_array(t, 1) && t.numel() == size &&
                   (t.scalar_type() == at::kFloat || t.scalar_type() == at::kDouble);
        };
        require(valid_logits(a, graph->n) && valid_logits(b, graph->m),
                "Expected floating CPU logits of shapes [num_nodes] and [num_edges].");
        root_mass = at::empty({graph->n}, a.options().dtype(at::kDouble));
        cumulative = at::empty({graph->row.back()}, root_mass.options());
        AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "wilson_vertex_logits", [&] {
            const auto* vertex_logits = a.const_data_ptr<scalar_t>();
            AT_DISPATCH_FLOATING_TYPES(b.scalar_type(), "wilson_edge_logits", [&] {
                build_weights(vertex_logits, b.const_data_ptr<scalar_t>());
            });
        });
    }

    template <typename V, typename E>
    void build_weights(const V* a, const E* b) {
        at::parallel_for(0, graph->m, 4096, [&](Index from, Index to) {
            for (Index e = from; e < to; ++e) {
                require(std::isfinite(b[e]), "Edge scores must be finite.");
            }
        });
        const auto* ids = graph->edge_id.const_data_ptr<Index>();
        double* roots = root_mass.data_ptr<double>();
        double* cdf = cumulative.data_ptr<double>();
        at::parallel_for(0, graph->n, 512, [&](Index from, Index to) {
            for (Index v = from; v < to; ++v) {
                const Index first = graph->row[v], stop = graph->row[v + 1];
                require(std::isfinite(a[v]), "Vertex scores must be finite.");
                double shift = a[v];
                for (Index i = first; i < stop; ++i) {
                    shift = std::max(shift, static_cast<double>(b[ids[i]]));
                }
                double weight = std::exp(static_cast<double>(a[v]) - shift);
                if (weight == 0.0) {
                    throw RootUnderflow("A root transition underflowed.");
                }
                roots[v] = weight;
                for (Index i = first; i < stop; ++i) {
                    weight += std::exp(static_cast<double>(b[ids[i]]) - shift);
                    cdf[i] = weight;
                }
            }
        });
    }

    Index walk(Index g, uint64_t seed, Index* root, Index* last_arc) const {
        const auto* dst = graph->neighbor.const_data_ptr<Index>();
        const auto* mass = root_mass.const_data_ptr<double>();
        const auto* cdf = cumulative.const_data_ptr<double>();
        Index count = 0;
        for (Index start = graph->boundaries[g]; start < graph->boundaries[g + 1]; ++start) {
            if (root[start] >= 0) {
                continue;
            }
            Index v = start;
            while (root[v] < 0) {
                const Index begin = graph->row[v], end = graph->row[v + 1];
                if (begin < end) {
                    const double target = uniform01(seed) * cdf[end - 1];
                    if (target >= mass[v]) {
                        // The last bin is the implicit upper bound of the search.
                        const Index i = std::upper_bound(cdf + begin, cdf + end - 1, target) - cdf;
                        last_arc[v] = i;
                        v = dst[i];
                        continue;
                    }
                }
                last_arc[v] = -1;
                root[v] = v;
                ++count;
            }
            const Index sink = root[v];
            // The most recent exit at each vertex gives the loop-erased path.
            for (v = start; root[v] < 0; v = dst[last_arc[v]]) {
                root[v] = sink;
            }
        }
        return count;
    }

    std::tuple<Tensor, Tensor, Tensor> sample(const Tensor& seeds) const {
        const Index groups = graph->graphs();
        require(is_cpu_array(seeds, 1) && seeds.scalar_type() == at::kLong && seeds.numel() == groups,
                "Expected one CPU int64 seed per graph.");
        Tensor cluster = at::full({graph->n}, -1, seeds.options());
        Tensor exits = at::empty_like(cluster);
        auto* root = cluster.data_ptr<Index>();
        auto* last_arc = exits.data_ptr<Index>();
        const auto* seed = seeds.const_data_ptr<Index>();
        std::vector<Index> counts(static_cast<size_t>(groups) + 1, 0);
        // Avoid dispatching a parallel region for very small batches.
        const Index grain = graph->n < 2048 ? groups : 1;
        at::parallel_for(0, groups, grain, [&](Index from, Index to) {
            for (Index g = from; g < to; ++g) {
                counts[g + 1] = walk(g, static_cast<uint64_t>(seed[g]), root, last_arc);
            }
        });
        std::partial_sum(counts.begin(), counts.end(), counts.begin());
        Tensor roots = at::empty({counts.back()}, seeds.options());
        Tensor matching = at::empty({graph->n - counts.back(), 2}, seeds.options());
        auto* roots_out = roots.data_ptr<Index>();
        auto pairs = matching.accessor<Index, 2>();
        const auto* ids = graph->edge_id.const_data_ptr<Index>();
        at::parallel_for(0, groups, grain, [&](Index from, Index to) {
            for (Index g = from; g < to; ++g) {
                Index k = counts[g];
                const Index first = graph->boundaries[g], stop = graph->boundaries[g + 1];
                for (Index v = first; v < stop; ++v) {
                    const Index arc = last_arc[v];
                    if (arc < 0) {
                        roots_out[k] = v;
                        last_arc[v] = k++;
                    } else {
                        pairs[v - k][0] = v;
                        pairs[v - k][1] = ids[arc];
                    }
                }
                // Roots now store compact ranks in the finished exit buffer.
                for (Index v = first; v < stop; ++v) {
                    root[v] = last_arc[root[v]];
                }
            }
        });
        return {matching, roots, cluster};
    }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::register_exception<RootUnderflow>(m, "RootUnderflow", PyExc_FloatingPointError);
    py::class_<Transitions>(m, "Transitions")
        .def_property_readonly("num_graphs", [](const Transitions& t) { return t.graph->graphs(); })
        .def("sample", &Transitions::sample, py::call_guard<py::gil_scoped_release>());
    py::class_<Topology, std::shared_ptr<Topology>>(m, "Topology")
        .def(py::init<const Tensor&, Index, const std::optional<Tensor>&>(),
             py::call_guard<py::gil_scoped_release>())
        .def("prepare",
             [](std::shared_ptr<Topology> g, const Tensor& a, const Tensor& b) {
                 return std::make_unique<Transitions>(std::move(g), a, b);
             },
             py::call_guard<py::gil_scoped_release>());
}
