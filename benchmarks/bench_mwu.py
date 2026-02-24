"""Benchmark: numba-mwu vs scipy.stats.mannwhitneyu.

Compares throughput across:
  - Dense vs sparse matrices
  - Integer vs float data
  - Various matrix sizes (small → large)

Run:
    uv run python benchmarks/bench_mwu.py
"""

import time

import numpy as np
from scipy import sparse, stats

from numba_mwu import mannwhitneyu_columns, mannwhitneyu_sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dense_int(rng, n_cells, n_genes, n_a):
    """Non-negative integer data (raw counts)."""
    data = rng.integers(0, 50, size=(n_cells, n_genes)).astype(np.float64)
    group_a = np.zeros(n_cells, dtype=bool)
    group_a[:n_a] = True
    return data, group_a


def _make_dense_float(rng, n_cells, n_genes, n_a):
    """Non-negative float data (normalized expression)."""
    data = rng.exponential(2.0, size=(n_cells, n_genes))
    # Zero out ~70% to mimic real expression
    mask = rng.random((n_cells, n_genes)) < 0.7
    data[mask] = 0.0
    group_a = np.zeros(n_cells, dtype=bool)
    group_a[:n_a] = True
    return data, group_a


def _make_sparse(rng, n_cells, n_genes, n_a, sparsity, dtype_kind):
    """Sparse CSR matrix with given sparsity level."""
    density = 1.0 - sparsity
    data = np.zeros((n_cells, n_genes), dtype=np.float64)
    mask = rng.random((n_cells, n_genes)) < density
    if dtype_kind == "int":
        data[mask] = rng.integers(1, 100, size=mask.sum()).astype(np.float64)
    else:
        data[mask] = rng.exponential(5.0, size=mask.sum())
    sp = sparse.csr_matrix(data)
    sp.eliminate_zeros()
    group_a = np.zeros(n_cells, dtype=bool)
    group_a[:n_a] = True
    return sp, data, group_a


def _time_fn(fn, *args, n_repeats=3, warmup=1):
    """Time a function, returning median seconds over n_repeats."""
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return np.median(times)


def _scipy_columnwise(data, n_a, n_genes):
    """Run scipy mannwhitneyu on each column (the serial baseline)."""
    for j in range(n_genes):
        stats.mannwhitneyu(data[:n_a, j], data[n_a:, j], method="asymptotic")


def _scipy_columnwise_sparse(sp_data, group_a, n_genes):
    """Run scipy mannwhitneyu on each column of dense-from-sparse."""
    dense = sp_data.toarray()
    a = group_a
    for j in range(n_genes):
        stats.mannwhitneyu(dense[a, j], dense[~a, j], method="asymptotic")


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

DENSE_SCENARIOS = [
    # (label, n_cells, n_genes, n_a)
    ("small (100x50)", 100, 50, 40),
    ("medium (1000x500)", 1000, 500, 400),
    ("large (5000x2000)", 5000, 2000, 2000),
    ("xlarge (10000x5000)", 10000, 5000, 4000),
]

SPARSE_SCENARIOS = [
    # (label, n_cells, n_genes, n_a, sparsity)
    ("small 90% (200x100)", 200, 100, 80, 0.90),
    ("medium 90% (2000x1000)", 2000, 1000, 800, 0.90),
    ("large 95% (5000x2000)", 5000, 2000, 2000, 0.95),
    ("xlarge 95% (10000x5000)", 10000, 5000, 4000, 0.95),
]


def _fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.1f} ms"
    return f"{seconds:8.2f}  s"


def _fmt_speedup(scipy_t, numba_t):
    if numba_t > 0:
        return f"{scipy_t / numba_t:7.1f}x"
    return "     inf"


def run_dense_benchmarks():
    print("=" * 80)
    print("DENSE MATRIX BENCHMARKS")
    print("=" * 80)
    print()

    rng = np.random.default_rng(42)

    for dtype_label, make_fn in [
        ("integer", _make_dense_int),
        ("float", _make_dense_float),
    ]:
        print(f"--- {dtype_label} data ---")
        print(f"{'scenario':<28} {'scipy':>12} {'numba':>12} {'speedup':>10}")
        print("-" * 65)

        for label, n_cells, n_genes, n_a in DENSE_SCENARIOS:
            data, group_a = make_fn(rng, n_cells, n_genes, n_a)

            t_scipy = _time_fn(
                _scipy_columnwise, data, n_a, n_genes, n_repeats=3, warmup=0
            )
            t_numba = _time_fn(mannwhitneyu_columns, data, n_a, n_repeats=3)

            print(
                f"{label:<28} {_fmt_time(t_scipy):>12} {_fmt_time(t_numba):>12} "
                f"{_fmt_speedup(t_scipy, t_numba):>10}"
            )

        print()


def run_sparse_benchmarks():
    print("=" * 80)
    print("SPARSE MATRIX BENCHMARKS")
    print("=" * 80)
    print()

    rng = np.random.default_rng(42)

    for dtype_label in ("integer", "float"):
        print(f"--- {dtype_label} data ---")
        print(
            f"{'scenario':<28} {'scipy (dense)':>14} {'numba sparse':>14} "
            f"{'numba dense':>14} {'sp speedup':>12}"
        )
        print("-" * 85)

        for label, n_cells, n_genes, n_a, sparsity in SPARSE_SCENARIOS:
            sp, dense, group_a = _make_sparse(
                rng, n_cells, n_genes, n_a, sparsity, dtype_label
            )

            t_scipy = _time_fn(
                _scipy_columnwise_sparse, sp, group_a, n_genes, n_repeats=3, warmup=0
            )
            t_numba_sparse = _time_fn(mannwhitneyu_sparse, sp, group_a, n_repeats=3)
            t_numba_dense = _time_fn(mannwhitneyu_columns, dense, n_a, n_repeats=3)

            print(
                f"{label:<28} {_fmt_time(t_scipy):>14} {_fmt_time(t_numba_sparse):>14} "
                f"{_fmt_time(t_numba_dense):>14} {_fmt_speedup(t_scipy, t_numba_sparse):>12}"
            )

        print()


def run_single_test_benchmarks():
    print("=" * 80)
    print("SINGLE PAIR BENCHMARKS (overhead comparison)")
    print("=" * 80)
    print()

    from numba_mwu import mannwhitneyu

    rng = np.random.default_rng(42)

    scenarios = [
        ("n=20 vs n=20", 20, 20),
        ("n=100 vs n=100", 100, 100),
        ("n=500 vs n=500", 500, 500),
        ("n=1000 vs n=1000", 1000, 1000),
    ]

    for dtype_label in ("integer", "float"):
        print(f"--- {dtype_label} data ---")
        print(f"{'scenario':<28} {'scipy':>12} {'numba':>12} {'speedup':>10}")
        print("-" * 65)

        for label, n1, n2 in scenarios:
            if dtype_label == "integer":
                x = rng.integers(0, 50, size=n1).astype(np.float64)
                y = rng.integers(0, 50, size=n2).astype(np.float64)
            else:
                x = rng.exponential(2.0, size=n1)
                y = rng.exponential(2.0, size=n2)

            def run_scipy():
                stats.mannwhitneyu(x, y, method="asymptotic")

            def run_numba():
                mannwhitneyu(x, y)

            t_scipy = _time_fn(run_scipy, n_repeats=50, warmup=5)
            t_numba = _time_fn(run_numba, n_repeats=50, warmup=5)

            print(
                f"{label:<28} {_fmt_time(t_scipy):>12} {_fmt_time(t_numba):>12} "
                f"{_fmt_speedup(t_scipy, t_numba):>10}"
            )

        print()


if __name__ == "__main__":
    print()
    print("numba-mwu benchmarks")
    print("Warming up JIT compilation...")
    print()

    # Warmup JIT once so compilation time isn't included
    from numba_mwu import mannwhitneyu

    _rng = np.random.default_rng(0)
    _x = _rng.standard_normal(10)
    _y = _rng.standard_normal(10)
    mannwhitneyu(_x, _y)

    _dense = _rng.integers(0, 5, size=(20, 3)).astype(np.float64)
    mannwhitneyu_columns(_dense, 10)

    _sp = sparse.csr_matrix(_dense)
    _sp.eliminate_zeros()
    _ga = np.array([True] * 10 + [False] * 10)
    mannwhitneyu_sparse(_sp, _ga)

    print("JIT warmup complete.")
    print()

    run_single_test_benchmarks()
    run_dense_benchmarks()
    run_sparse_benchmarks()
