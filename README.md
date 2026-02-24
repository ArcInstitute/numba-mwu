# numba-mwu

Numba-accelerated Mann-Whitney U test.
Drop-in replacement for `scipy.stats.mannwhitneyu` with parallel batch operations and native sparse matrix support.

All functions use the asymptotic (normal approximation) method and produce results identical to `scipy.stats.mannwhitneyu(..., method="asymptotic")`.

> Note: This is only supported for 1D and 2D inputs.

## Installation

```bash
uv pip install numba-mwu
```

## API

Every function returns a `MannWhitneyUResult` named tuple with `statistic` and `pvalue` fields. The batch functions return arrays instead of scalars.

All functions accept `use_continuity` (default `True`) and `alternative` (`"two-sided"`, `"less"`, `"greater"`).

### `mannwhitneyu(x, y)`

Single two-sample test. Equivalent to scipy's `mannwhitneyu`.

```python
from numba_mwu import mannwhitneyu

result = mannwhitneyu(x, y)
result.statistic  # U statistic
result.pvalue     # two-sided p-value
```

### `mannwhitneyu_rows(X, y)`

Test each row of a 2-D array `X` against a shared reference sample `y`.
Parallelized across rows.

```python
from numba_mwu import mannwhitneyu_rows

# X: (n_tests, n1), y: (n2,)
result = mannwhitneyu_rows(X, y)
result.statistic  # shape (n_tests,)
result.pvalue     # shape (n_tests,)
```

### `mannwhitneyu_columns(X, Y)`

Test each column of `X` against the corresponding column of `Y`.
Parallelized across columns.
Designed for the common case of slicing a cells-by-genes matrix into two groups:

```python
from numba_mwu import mannwhitneyu_columns

# expression: (n_cells, n_genes), labels: (n_cells,)
X = expression[labels == "A"]  # (n1, n_genes)
Y = expression[labels == "B"]  # (n2, n_genes)

result = mannwhitneyu_columns(X, Y)
result.statistic  # shape (n_genes,)
result.pvalue     # shape (n_genes,)
```

### `mannwhitneyu_sparse(X, Y)`

Same as `mannwhitneyu_columns` but operates directly on CSR sparse matrices without converting to dense.

Memory overhead per matrix is one `int64` array of length `nnz` (column permutation) plus one `int64` array of length `n_genes + 1` (column pointers).
No data values are copied.

Requires non-negative data (raw counts, normalized expression, etc.).

> Note: Call `eliminate_zeros()` on each matrix beforehand if it may contain explicitly stored zeros.

```python
from numba_mwu import mannwhitneyu_sparse

# adata.X is a CSR matrix, adata.obs["group"] has labels
mask = adata.obs["group"] == "A"
X = adata.X[mask]    # CSR row-slice is still CSR
Y = adata.X[~mask]


result = mannwhitneyu_sparse(X, Y)
result.statistic  # shape (n_genes,)
result.pvalue     # shape (n_genes,)
```

## Benchmarks

Run benchmarks with:

```bash
uv run benchmarks/bench_mwu.py
```
