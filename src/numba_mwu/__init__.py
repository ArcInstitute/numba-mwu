"""Numba-accelerated Mann-Whitney U test."""

from collections import namedtuple

import numpy as np

from ._batch import _mannwhitneyu_columns, _mannwhitneyu_rows
from ._core import GREATER, LESS, TWO_SIDED, _mannwhitneyu_single
from ._sparse import _build_col_index, _sparse_mwu_batch

__all__ = [
    "MannWhitneyUResult",
    "mannwhitneyu",
    "mannwhitneyu_rows",
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


def mannwhitneyu_rows(X, y, use_continuity=True, alternative="two-sided"):
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
    stats, pvals = _mannwhitneyu_rows(X, y, use_continuity, alt)
    return MannWhitneyUResult(stats, pvals)


def _validate_2d(arr, name):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 2-dimensional, got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError(f"`{name}` must have at least one row.")
    if np.any(np.isnan(arr)):
        raise ValueError(f"`{name}` must not contain NaNs.")
    return arr


def mannwhitneyu_columns(X, Y, use_continuity=True, alternative="two-sided"):
    """Run Mann-Whitney U test on each column of X vs corresponding column of Y.

    Designed for the common workflow where a matrix has been sliced by
    group membership into two separate matrices (or views).

    Parameters
    ----------
    X : array_like, shape (n1, n_genes)
        2-D array for group A (one column per gene/feature).
    Y : array_like, shape (n2, n_genes)
        2-D array for group B. Must have the same number of columns as X.
    use_continuity : bool, optional
        Whether a continuity correction (1/2) should be applied. Default True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.

    Returns
    -------
    result : MannWhitneyUResult
        Named tuple with ``statistic`` and ``pvalue`` arrays of shape (n_genes,).
    """
    X = _validate_2d(X, "X")
    Y = _validate_2d(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"`X` and `Y` must have the same number of columns, "
            f"got {X.shape[1]} and {Y.shape[1]}."
        )
    alt = _validate_alternative(alternative)
    stats, pvals = _mannwhitneyu_columns(X, Y, use_continuity, alt)
    return MannWhitneyUResult(stats, pvals)


def _validate_csr(X, name):
    from scipy.sparse import issparse, isspmatrix_csr

    if not issparse(X):
        raise TypeError(f"`{name}` must be a scipy sparse matrix.")
    if not (isspmatrix_csr(X) or X.format == "csr"):
        raise TypeError(
            f"`{name}` must be in CSR format. Convert with `{name}.tocsr()` if needed."
        )
    if X.data.size > 0 and X.data.min() < 0:
        raise ValueError(
            f"Sparse MWU requires non-negative data in `{name}`. "
            "For data with negative values, convert to dense and use "
            "mannwhitneyu_columns."
        )
    return X


def mannwhitneyu_sparse(X, Y, use_continuity=True, alternative="two-sided"):
    """Run Mann-Whitney U test for each gene (column) of two sparse matrices.

    Designed for single-cell expression matrices where rows are cells and
    columns are genes. Each matrix represents one group — typically sliced
    from a full expression matrix by cell labels::

        X = full_matrix[labels == "A"]  # CSR row-slice is still CSR
        Y = full_matrix[labels == "B"]

    Works directly with CSR format without converting to CSC or dense.
    The only allocation per matrix is a column-index permutation array
    (one int per nonzero entry) and column pointers (one int per gene).

    Requires non-negative data — zeros must be the smallest values so they
    form a contiguous block at the start of the sorted order. This holds
    for raw counts, normalized expression, and any non-negative transformation.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix or csr_array, shape (n1, n_genes)
        Sparse expression matrix for group A in CSR format. Must have
        non-negative values. Call ``X.eliminate_zeros()`` beforehand if
        the matrix may contain explicitly stored zeros.
    Y : scipy.sparse.csr_matrix or csr_array, shape (n2, n_genes)
        Sparse expression matrix for group B in CSR format. Must have
        the same number of columns as X.
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
    X = _validate_csr(X, "X")
    Y = _validate_csr(Y, "Y")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"`X` and `Y` must have the same number of columns, "
            f"got {X.shape[1]} and {Y.shape[1]}."
        )
    if X.shape[0] == 0:
        raise ValueError("`X` must have at least one row.")
    if Y.shape[0] == 0:
        raise ValueError("`Y` must have at least one row.")

    alt = _validate_alternative(alternative)
    n_genes = X.shape[1]
    n_a = X.shape[0]
    n_b = Y.shape[0]

    # Build column indices for each matrix — no data copy
    data_a = np.ascontiguousarray(X.data, dtype=np.float64)
    indptr_a = np.ascontiguousarray(X.indptr)
    indices_a = np.ascontiguousarray(X.indices)
    col_indptr_a, col_order_a = _build_col_index(indptr_a, indices_a, n_genes)

    data_b = np.ascontiguousarray(Y.data, dtype=np.float64)
    indptr_b = np.ascontiguousarray(Y.indptr)
    indices_b = np.ascontiguousarray(Y.indices)
    col_indptr_b, col_order_b = _build_col_index(indptr_b, indices_b, n_genes)

    stats, pvals = _sparse_mwu_batch(
        data_a,
        col_indptr_a,
        col_order_a,
        n_a,
        data_b,
        col_indptr_b,
        col_order_b,
        n_b,
        use_continuity,
        alt,
    )
    return MannWhitneyUResult(stats, pvals)
