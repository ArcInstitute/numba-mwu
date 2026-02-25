# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`numba-mwu` is a Numba-accelerated Mann-Whitney U test, designed as a drop-in replacement for `scipy.stats.mannwhitneyu(method="asymptotic")`. Key use case is batch testing for single-cell expression data with native CSR sparse matrix support.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file or test class
uv run pytest tests/test_mannwhitneyu.py::TestSparse

# Run a single test
uv run pytest tests/test_mannwhitneyu.py::TestBasicCorrectness::test_scipy_example

# Run benchmarks
uv run python benchmarks/bench_mwu.py

# Build the package
uv build

# Type check (ty is the type checker configured)
uv run ty check
```

## Architecture

The library is split into four modules under `src/numba_mwu/`:

**`_core.py`** — Single-pair JIT-compiled computation. Implements `_rankdata_avg` (average-method ranking with tie tracking), `_ndtr` (normal CDF via `math.erfc`), and `_mannwhitneyu_single` which computes U statistic, tie-correction term, z-score, and p-value.

**`_batch.py`** — Parallel batch wrappers using `@nb.njit(parallel=True)` + `nb.prange`. Two functions: `_mannwhitneyu_rows` (each row of X vs shared y) and `_mannwhitneyu_columns` (corresponding columns of X vs Y).

**`_sparse.py`** — CSR sparse matrix support. `_build_col_index` constructs a column-order index (col_indptr + col_order permutation) without copying data values. `_sparse_mwu_column` handles zeros analytically (treated as a contiguous block at the start of sorted order), requiring non-negative input. `_sparse_mwu_batch` parallelizes column-wise computation.

**`__init__.py`** — Public API with input validation. Exports: `mannwhitneyu` (1D), `mannwhitneyu_rows` (2D X, 1D y), `mannwhitneyu_columns` (2D X and Y), `mannwhitneyu_sparse` (CSR matrices), and `sparse_column_index` (precompute reusable column index). All functions return `MannWhitneyUResult(statistic, pvalue)`.

### Sparse Column Index Reuse

`sparse_column_index(X)` precomputes the column traversal structure for a CSR matrix, returning a `SparseColumnIndex` namedtuple. This is useful when comparing multiple group matrices against the same reference—pass the precomputed index instead of the raw matrix to avoid repeated index construction.

### Alternative Hypotheses

Encoded as an integer enum in `_core.py`: `TWO_SIDED=0`, `LESS=1`, `GREATER=2`. The public API accepts string alternatives (`"two-sided"`, `"less"`, `"greater"`) via `_validate_alternative()`.

## Testing Conventions

All tests compare against `scipy.stats.mannwhitneyu(..., method='asymptotic')` using `np.isclose()`. The test file is organized into 11 classes, one per functional area. When adding new functionality, follow this pattern: add a new test class or extend the relevant existing class, and always validate against scipy's output.
