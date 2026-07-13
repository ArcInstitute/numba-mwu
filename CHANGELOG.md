# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-13

### Added

- `mannwhitneyu_one_vs_rest` / `mannwhitneyu_one_vs_rest_sparse`: one-shot Mann-Whitney U test for N groups tested simultaneously against "everything else" (the common 1-vs-rest / marker-feature workflow). Each column is ranked exactly once and every group's statistic is derived from that single ranking, instead of the `O(n_groups)` cost of calling `mannwhitneyu_columns`/`mannwhitneyu_sparse` once per group in a loop.
- CI workflow (GitHub Actions): formatting (`ruff format --check`), type checking (`ty check`), and `pytest` across Python 3.11-3.13.
- Publish workflow: builds and publishes to PyPI via trusted publishing when a GitHub Release is published.

### Changed

- Refactored `_mannwhitneyu_single`'s rank-sum-to-stats reduction into a shared `_mwu_stats_from_rank_sum` function, now reused by both the pairwise and one-vs-rest kernels so their numeric results can never drift apart.
- Bumped minimum supported Python version.
- Added `ruff` to dev dependencies.

## [0.1.1] - 2026-02-24

### Added

- `sparse_column_index()` / `SparseColumnIndex`: precompute a CSR column index once and reuse it across multiple `mannwhitneyu_sparse` calls against the same reference matrix, avoiding redundant index construction.

## [0.1.0] - 2026-02-24

### Added

- Initial release.
- `mannwhitneyu(x, y)`: single-pair Mann-Whitney U test, Numba-accelerated asymptotic method equivalent to `scipy.stats.mannwhitneyu(..., method="asymptotic")`.
- `mannwhitneyu_rows(X, y)`: batch test of each row of `X` against a shared reference sample `y`.
- `mannwhitneyu_columns(X, Y)`: batch test of corresponding columns of `X` and `Y`.
- `mannwhitneyu_sparse(X, Y)`: native CSR sparse matrix support, handling zeros analytically without densifying.
- `use_continuity` and `alternative` (`"two-sided"`, `"less"`, `"greater"`) options across all functions.
- Benchmarks against `scipy.stats.mannwhitneyu`.
