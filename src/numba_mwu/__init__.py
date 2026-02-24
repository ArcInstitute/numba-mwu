"""Numba-accelerated Mann-Whitney U test."""

from collections import namedtuple

import numpy as np

from ._batch import _mannwhitneyu_batch, _mannwhitneyu_columns
from ._core import GREATER, LESS, TWO_SIDED, _mannwhitneyu_single
from ._sparse import _build_col_index, _expand_row_indices, _sparse_mwu_batch

__all__ = [
    "MannWhitneyUResult",
    "mannwhitneyu",
    "mannwhitneyu_batch",
    "mannwhitneyu_columns",
    "mannwhitneyu_sparse",
]

MannWhitneyUResult = namedtuple("MannWhitneyUResult", ("statistic", "pvalue"))

_ALTERNATIVE_MAP = {
    "two-sided": TWO_SIDED,
    "less": LESS,
    "greater": GREATER,
}


def _validate_alternative(alternative):
    alt = alternative.lower()
    if alt not in _ALTERNATIVE_MAP:
        raise ValueError(
            f"`alternative` must be one of {set(_ALTERNATIVE_MAP)}, got {alternative!r}"
        )
    return _ALTERNATIVE_MAP[alt]


def _validate_1d(arr, name):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be 1-dimensional, got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError(f"`{name}` must be of nonzero size.")
    if np.any(np.isnan(arr)):
        raise ValueError(f"`{name}` must not contain NaNs.")
    return arr


def mannwhitneyu(x, y, use_continuity=True, alternative="two-sided"):
    """Perform the Mann-Whitney U rank test on two independent samples.

    Numba-accelerated asymptotic implementation equivalent to
    ``scipy.stats.mannwhitneyu(..., method='asymptotic')``.

    Parameters
    ----------
    x, y : array_like
        1-D arrays of samples.
    use_continuity : bool, optional
        Whether a continuity correction (1/2) should be applied. Default True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.

    Returns
    -------
    result : MannWhitneyUResult
        Named tuple with ``statistic`` (U for sample x) and ``pvalue``.
    """
    x = _validate_1d(x, "x")
    y = _validate_1d(y, "y")
    alt = _validate_alternative(alternative)
    stat, pval = _mannwhitneyu_single(x, y, use_continuity, alt)
    return MannWhitneyUResult(stat, pval)


def mannwhitneyu_batch(X, y, use_continuity=True, alternative="two-sided"):
    """Run Mann-Whitney U test for each row of X against y (parallelized).

    Parameters
    ----------
    X : array_like, shape (n_tests, n1)
        2-D array where each row is a sample to test against y.
    y : array_like, shape (n2,)
        1-D reference sample.
    use_continuity : bool, optional
        Whether a continuity correction (1/2) should be applied. Default True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.

    Returns
    -------
    result : MannWhitneyUResult
        Named tuple with ``statistic`` and ``pvalue`` arrays of shape (n_tests,).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"`X` must be 2-dimensional, got ndim={X.ndim}")
    if X.shape[0] == 0:
        raise ValueError("`X` must have at least one row.")
    if np.any(np.isnan(X)):
        raise ValueError("`X` must not contain NaNs.")
    y = _validate_1d(y, "y")
    alt = _validate_alternative(alternative)
    stats, pvals = _mannwhitneyu_batch(X, y, use_continuity, alt)
    return MannWhitneyUResult(stats, pvals)


def mannwhitneyu_columns(data, n1, use_continuity=True, alternative="two-sided"):
    """Run Mann-Whitney U test on each column, split at row n1 (parallelized).

    Parameters
    ----------
    data : array_like, shape (n1 + n2, n_tests)
        2-D array where each column contains concatenated samples.
    n1 : int
        Number of rows belonging to the first sample.
    use_continuity : bool, optional
        Whether a continuity correction (1/2) should be applied. Default True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.

    Returns
    -------
    result : MannWhitneyUResult
        Named tuple with ``statistic`` and ``pvalue`` arrays of shape (n_tests,).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"`data` must be 2-dimensional, got ndim={data.ndim}")
    n1 = int(n1)
    if n1 <= 0 or n1 >= data.shape[0]:
        raise ValueError(
            f"`n1` must be between 1 and {data.shape[0] - 1}, got {n1}"
        )
    if np.any(np.isnan(data)):
        raise ValueError("`data` must not contain NaNs.")
    alt = _validate_alternative(alternative)
    stats, pvals = _mannwhitneyu_columns(data, n1, use_continuity, alt)
    return MannWhitneyUResult(stats, pvals)


def mannwhitneyu_sparse(X, group_a, use_continuity=True, alternative="two-sided"):
    """Run Mann-Whitney U test for each gene (column) of a sparse matrix.

    Designed for single-cell expression matrices where rows are cells and
    columns are genes. Works directly with CSR format (the standard for
    single-cell data) without converting to CSC — no copy of the data
    array is made. The only allocation is a column-index permutation array
    (one int per nonzero entry) and column pointers (one int per gene).

    Requires non-negative data — zeros must be the smallest values so they
    form a contiguous block at the start of the sorted order. This holds
    for raw counts, normalized expression, and any non-negative transformation.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix or csr_array, shape (n_cells, n_genes)
        Sparse expression matrix in CSR format. Must have non-negative
        values. Call ``X.eliminate_zeros()`` beforehand if the matrix may
        contain explicitly stored zeros.
    group_a : array_like, shape (n_cells,)
        Boolean mask indicating which cells belong to group A.
        Group B is all cells where ``group_a`` is False.
    use_continuity : bool, optional
        Whether to apply continuity correction. Default True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis. Default 'two-sided'.

    Returns
    -------
    result : MannWhitneyUResult
        Named tuple with ``statistic`` and ``pvalue`` arrays of shape
        (n_genes,).
    """
    from scipy.sparse import issparse, isspmatrix_csr

    if not issparse(X):
        raise TypeError("`X` must be a scipy sparse matrix.")
    if not (isspmatrix_csr(X) or X.format == "csr"):
        raise TypeError(
            "`X` must be in CSR format. "
            "Convert with `X.tocsr()` if needed."
        )

    if X.data.size > 0 and X.data.min() < 0:
        raise ValueError(
            "Sparse MWU requires non-negative data. "
            "For data with negative values, convert to dense and use "
            "mannwhitneyu_columns."
        )

    group_a = np.asarray(group_a, dtype=np.bool_)
    if group_a.ndim != 1 or group_a.shape[0] != X.shape[0]:
        raise ValueError(
            f"`group_a` must be a 1-D boolean array with length {X.shape[0]}, "
            f"got shape {group_a.shape}."
        )

    n_a = int(group_a.sum())
    n_b = X.shape[0] - n_a
    if n_a == 0 or n_b == 0:
        raise ValueError("Both groups must have at least one member.")

    alt = _validate_alternative(alternative)

    # Use the CSR arrays directly — no data copy
    csr_indptr = np.ascontiguousarray(X.indptr)
    csr_indices = np.ascontiguousarray(X.indices)
    csr_data = np.ascontiguousarray(X.data, dtype=np.float64)

    n_cells, n_genes = X.shape

    # Build lightweight column index: permutation + column pointers
    # Memory: nnz * 8 bytes (col_order) + (n_genes+1) * 8 bytes (col_indptr)
    col_indptr, col_order = _build_col_index(csr_indptr, csr_indices, n_genes)

    # Expand CSR indptr into flat row indices for O(1) row lookup
    # Memory: nnz * 8 bytes
    row_indices = _expand_row_indices(csr_indptr)

    stats, pvals = _sparse_mwu_batch(
        csr_data, row_indices, col_indptr, col_order,
        n_cells, group_a, n_a, use_continuity, alt
    )
    return MannWhitneyUResult(stats, pvals)
