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

```text
================================================================================
SINGLE PAIR BENCHMARKS (overhead comparison)
================================================================================

--- integer data ---
scenario                            scipy        numba    speedup
-----------------------------------------------------------------
n=20 vs n=20                     223.1 us       3.9 us      56.9x
n=100 vs n=100                   224.0 us       5.4 us      41.7x
n=500 vs n=500                   248.3 us      12.6 us      19.7x
n=1000 vs n=1000                 287.2 us      22.7 us      12.7x

--- float data ---
scenario                            scipy        numba    speedup
-----------------------------------------------------------------
n=20 vs n=20                     212.6 us       3.9 us      53.9x
n=100 vs n=100                   220.7 us       5.6 us      39.4x
n=500 vs n=500                   249.4 us      14.7 us      16.9x
n=1000 vs n=1000                 287.3 us      27.4 us      10.5x

================================================================================
DENSE MATRIX BENCHMARKS
================================================================================

--- integer data ---
scenario                            scipy        numba    speedup
-----------------------------------------------------------------
small (100x50)                    11.4 ms      64.1 us     177.8x
medium (1000x500)                139.5 ms       1.5 ms      94.0x
large (5000x2000)                 1.01  s      43.7 ms      23.0x
xlarge (10000x5000)               3.93  s     179.5 ms      21.9x

--- float data ---
scenario                            scipy        numba    speedup
-----------------------------------------------------------------
small (100x50)                    11.1 ms      53.0 us     208.5x
medium (1000x500)                131.5 ms       1.2 ms     109.1x
large (5000x2000)                866.6 ms      36.0 ms      24.1x
xlarge (10000x5000)               3.33  s     151.9 ms      22.0x

================================================================================
SPARSE MATRIX BENCHMARKS
================================================================================

--- integer data ---
scenario                      scipy (dense)   numba sparse    numba dense   sp speedup
-------------------------------------------------------------------------------------
small 90% (200x100)                 22.7 ms        51.3 us        84.3 us       442.3x
medium 90% (2000x1000)             275.5 ms         1.0 ms         3.5 ms       266.9x
large 95% (5000x2000)              746.8 ms         2.6 ms        20.4 ms       282.1x
xlarge 95% (10000x5000)             2.80  s        21.1 ms       117.2 ms       132.6x

--- float data ---
scenario                      scipy (dense)   numba sparse    numba dense   sp speedup
-------------------------------------------------------------------------------------
small 90% (200x100)                 22.7 ms        53.2 us        80.7 us       427.0x
medium 90% (2000x1000)             279.5 ms         1.0 ms         4.3 ms       268.9x
large 95% (5000x2000)              741.1 ms         3.5 ms        23.7 ms       209.4x
xlarge 95% (10000x5000)             2.80  s        21.0 ms       111.5 ms       133.0x
```
