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

**`_core.py`** — Single-pair JIT-compiled computation. Implements `_rankdata_avg` (average-method ranking with tie tracking), `_ndtr` (normal CDF via `math.erfc`), `_mwu_stats_from_rank_sum` (reduces a rank sum + tie term into `(U1, pvalue)` — the asymptotic formula shared by every kernel in the library), and `_mannwhitneyu_single` (ranks two concatenated samples, then reduces through `_mwu_stats_from_rank_sum`).

**`_batch.py`** — Parallel batch wrappers using `@nb.njit(parallel=True)` + `nb.prange`. Pairwise: `_mannwhitneyu_rows` (each row of X vs shared y) and `_mannwhitneyu_columns` (corresponding columns of X vs Y). One-vs-rest (dense): `_one_vs_rest_rank_sums_dense` ranks each column once and reduces to per-group rank sums + a tie term; `_mwu_stats_one_vs_rest` reduces those (via `_mwu_stats_from_rank_sum`) into `(U, p)` for every `(group, column)` pair — shared with the sparse one-vs-rest path in `_sparse.py`; `_mannwhitneyu_one_vs_rest_columns` (plain Python) sequences the two.

**`_sparse.py`** — CSR sparse matrix support. `_build_col_index` constructs a column-order index (col_indptr + col_order permutation) without copying data values; `_build_col_index_with_rows` extends it with each entry's originating row (needed by one-vs-rest to look up a nonzero's group). `_sparse_mwu_column`/`_sparse_mwu_batch` are the pairwise path, handling zeros analytically (contiguous block at the start of sorted order), requiring non-negative input. `_one_vs_rest_rank_sums_sparse`/`_mannwhitneyu_one_vs_rest_sparse_batch` are the one-vs-rest counterpart, using the same zero-block trick generalized to N groups.

**`__init__.py`** — Public API with input validation. Exports: `mannwhitneyu` (1D), `mannwhitneyu_rows` (2D X, 1D y), `mannwhitneyu_columns` (2D X and Y), `mannwhitneyu_sparse` (CSR matrices), `mannwhitneyu_one_vs_rest`/`mannwhitneyu_one_vs_rest_sparse` (N groups vs "everything else" in one call), and `sparse_column_index` (precompute reusable column index). All functions return `MannWhitneyUResult(statistic, pvalue)`.

### Sparse Column Index Reuse

`sparse_column_index(X)` precomputes the column traversal structure for a CSR matrix, returning a `SparseColumnIndex` namedtuple. This is useful when comparing multiple group matrices against the same reference—pass the precomputed index instead of the raw matrix to avoid repeated index construction.

### Alternative Hypotheses

Encoded as an integer enum in `_core.py`: `TWO_SIDED=0`, `LESS=1`, `GREATER=2`. The public API accepts string alternatives (`"two-sided"`, `"less"`, `"greater"`) via `_validate_alternative()`.

### One-vs-Rest (N groups tested simultaneously)

`mannwhitneyu_one_vs_rest`/`mannwhitneyu_one_vs_rest_sparse` generalize the pairwise functions to test every group against "all other rows" in a single call. This exploits an invariant specific to one-vs-rest: `group ∪ rest` is always the *entire* input, regardless of which group is being tested. So each column is ranked exactly once (and the tie-correction term, which depends only on the full column's tie structure, is computed once too), and every group's statistic is derived from that single ranking via a cheap per-group rank-sum reduction — instead of the `O(n_groups)` blow-up of calling `mannwhitneyu_columns(group, rest)` once per group (which re-ranks `group + rest` from scratch every time). `labels` is an integer array of group ids in `[0, n_groups)`, e.g. from `pd.factorize`/`pd.Categorical.codes`; callers are expected to drop unlabeled/filtered rows before calling (no `-1`/sentinel handling here).

## Testing Conventions

All tests compare against `scipy.stats.mannwhitneyu(..., method='asymptotic')` using `np.isclose()`. The test file is organized into functional classes (one per area — pairwise, batch, sparse, one-vs-rest, validation). When adding new functionality, follow this pattern: add a new test class or extend the relevant existing class, and always validate against scipy's output.
