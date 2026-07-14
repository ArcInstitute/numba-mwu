"""Numba-accelerated Mann-Whitney U test."""

from collections import namedtuple

import numpy as np

from ._batch import (
    _mannwhitneyu_columns,
    _mannwhitneyu_one_vs_rest_columns,
    _mannwhitneyu_rows,
)
from ._core import GREATER, LESS, TWO_SIDED, _mannwhitneyu_single
from ._sparse import (
    _build_col_index,
    _build_col_index_with_rows,
    _mannwhitneyu_one_vs_rest_sparse_batch,
    _sparse_mwu_batch,
)

__all__ = [
    "MannWhitneyUResult",
    "SparseColumnIndex",
    "mannwhitneyu",
    "mannwhitneyu_rows",
    "mannwhitneyu_columns",
    "mannwhitneyu_sparse",
    "mannwhitneyu_one_vs_rest",
    "mannwhitneyu_one_vs_rest_sparse",
    "sparse_column_index",
]

MannWhitneyUResult = namedtuple("MannWhitneyUResult", ("statistic", "pvalue"))
SparseColumnIndex = namedtuple(
    "SparseColumnIndex", ("data", "col_indptr", "col_order", "n_rows", "n_cols")
)

_ALTERNATIVE_MAP = {
    "two-sided": TWO_SIDED,
    "less": LESS,
    "greater": GREATER,
}

_PARALLEL_AXES = {"auto", "groups", "columns"}


def _validate_alternative(alternative):
    alt = alternative.lower()
    if alt not in _ALTERNATIVE_MAP:
        raise ValueError(
            f"`alternative` must be one of {set(_ALTERNATIVE_MAP)}, got {alternative!r}"
        )
    return _ALTERNATIVE_MAP[alt]


def _validate_parallel_axis(parallel_axis):
    if parallel_axis not in _PARALLEL_AXES:
        raise ValueError(
            f"`parallel_axis` must be one of {_PARALLEL_AXES}, got {parallel_axis!r}"
        )
    return parallel_axis


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


def _validate_labels(labels, n_rows, n_groups):
    """Validate a one-vs-rest group-label array.

    Returns
    -------
    labels : int64 array (n_rows,)
    n_groups : int
    group_sizes : int64 array (n_groups,)
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"`labels` must be 1-dimensional, got ndim={labels.ndim}")
    if labels.shape[0] != n_rows:
        raise ValueError(
            f"`labels` must have length {n_rows} (one per row), got {labels.shape[0]}"
        )
    if labels.shape[0] == 0:
        raise ValueError("`labels` must be of nonzero size.")
    if np.issubdtype(labels.dtype, np.floating):
        if np.any(np.isnan(labels)):
            raise ValueError("`labels` must not contain NaNs.")
        if np.any(labels != np.round(labels)):
            raise ValueError("`labels` must contain only integer-valued labels.")
    labels = labels.astype(np.int64)
    if labels.min() < 0:
        raise ValueError("`labels` must not contain negative values.")

    inferred_n_groups = int(labels.max()) + 1
    if n_groups is None:
        n_groups = inferred_n_groups
    elif n_groups < inferred_n_groups:
        raise ValueError(
            f"`n_groups={n_groups}` is smaller than the largest label "
            f"({inferred_n_groups - 1})."
        )
    if n_groups < 2:
        raise ValueError(
            f"One-vs-rest requires at least 2 groups, got n_groups={n_groups}."
        )

    group_sizes = np.bincount(labels, minlength=n_groups).astype(np.int64)
    empty = np.flatnonzero(group_sizes == 0)
    if empty.size:
        raise ValueError(f"Group id(s) {empty.tolist()} have no rows.")

    return labels, n_groups, group_sizes


def mannwhitneyu_one_vs_rest(
    X,
    labels,
    n_groups=None,
    use_continuity=True,
    alternative="two-sided",
    parallel_axis="auto",
):
    """Run a one-vs-rest Mann-Whitney U test across every group in one shot.

    Generalizes ``mannwhitneyu_columns`` from a single pair of groups to N
    groups tested simultaneously against "every other row" (the common
    1-vs-rest / marker-feature workflow). Because ``group ∪ rest`` is always
    the full input regardless of which group is being tested, each column is
    ranked exactly **once** and every group's statistic is derived from that
    single ranking — instead of the ``O(n_groups)`` blow-up a naive loop like
    ``[mannwhitneyu_columns(X[labels == g], X[labels != g]) for g in groups]``
    would pay by re-ranking ``group + rest`` from scratch for every group.

    Parameters
    ----------
    X : array_like, shape (n_rows, n_cols)
        All rows to compare — e.g. an expression matrix already restricted to
        the rows that carry a valid label (drop unlabeled/NaN rows first).
    labels : array_like, shape (n_rows,)
        Integer group id for each row, in ``[0, n_groups)``. Every group must
        have at least one row.
    n_groups : int, optional
        Number of groups. Defaults to ``labels.max() + 1``; pass explicitly
        only if that would under-count (rare).
    use_continuity : bool, optional
        Whether a continuity correction (1/2) should be applied. Default True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
    parallel_axis : {'auto', 'groups', 'columns'}, optional
        Which axis the final reduction step parallelizes over — a pure
        performance knob, never affects the result (every ``(group, column)``
        statistic is computed independently). ``'auto'`` (default) picks
        "groups" once ``n_groups`` reaches the number of numba threads —
        benchmarks show it wins from there on regardless of ``n_cols``,
        since "columns"'s strided access cost scales with ``n_groups``
        independent of how parallel it runs. Below that thread-count
        threshold (e.g. a handful of clusters tested against tens of
        thousands of genes), it picks whichever of ``n_groups``/``n_cols`` is
        larger, to keep as many threads busy as possible. Pass ``'groups'``
        or ``'columns'`` explicitly to override the heuristic (e.g. after
        profiling your own workload).

    Returns
    -------
    result : MannWhitneyUResult
        Named tuple with ``statistic`` and ``pvalue`` arrays of shape
        ``(n_groups, n_cols)``. Row ``g`` is group ``g``'s test against every
        other row in ``X``.
    """
    X = _validate_2d(X, "X")
    labels, n_groups, group_sizes = _validate_labels(labels, X.shape[0], n_groups)
    alt = _validate_alternative(alternative)
    axis = _validate_parallel_axis(parallel_axis)
    stats, pvals = _mannwhitneyu_one_vs_rest_columns(
        X, labels, n_groups, group_sizes, use_continuity, alt, axis
    )
    return MannWhitneyUResult(stats, pvals)


def _validate_csr(X, name):
    from scipy.sparse import issparse, isspmatrix_csr

    if not issparse(X):
        raise TypeError(f"`{name}` must be a scipy sparse matrix.")
    if not (isspmatrix_csr(X) or X.format == "csr"):
        raise TypeError(
            f"`{name}` must be in CSR format. Convert with `{name}.tocsr()` if needed."
        )
    if X.data.size > 0:
        if np.any(np.isnan(X.data)):
            raise ValueError(f"`{name}` must not contain NaNs.")
        if np.any(X.data < 0):
            raise ValueError(
                f"Sparse MWU requires non-negative data in `{name}`. "
                "For data with negative values, convert to dense and use "
                "mannwhitneyu_columns."
            )
        if np.any(X.data == 0):
            raise ValueError(
                f"`{name}` must not contain explicit zero entries. "
                f"Call `{name}.eliminate_zeros()` first."
            )
    return X


def sparse_column_index(X):
    """Precompute a column index for a CSR sparse matrix.

    The returned ``SparseColumnIndex`` can be passed to
    ``mannwhitneyu_sparse`` in place of a raw CSR matrix, avoiding
    redundant index construction when the same matrix is reused
    across many comparisons::

        ref_idx = sparse_column_index(ref_matrix)
        for label in labels:
            result = mannwhitneyu_sparse(group_matrix, ref_idx)

    Parameters
    ----------
    X : scipy.sparse.csr_matrix or csr_array
        Sparse matrix in CSR format with non-negative values.
        Call ``X.eliminate_zeros()`` beforehand if it may contain
        explicitly stored zeros.

    Returns
    -------
    SparseColumnIndex
        Precomputed index that can be passed to ``mannwhitneyu_sparse``.
    """
    X = _validate_csr(X, "X")
    if X.shape[0] == 0:
        raise ValueError("`X` must have at least one row.")
    data = np.ascontiguousarray(X.data, dtype=np.float64)
    indptr = np.ascontiguousarray(X.indptr)
    indices = np.ascontiguousarray(X.indices)
    col_indptr, col_order = _build_col_index(indptr, indices, X.shape[1])
    return SparseColumnIndex(data, col_indptr, col_order, X.shape[0], X.shape[1])


def _resolve_sparse(arg, name):
    """Convert a CSR matrix or SparseColumnIndex to a SparseColumnIndex."""
    if isinstance(arg, SparseColumnIndex):
        return arg
    return sparse_column_index(arg)


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

    Both ``X`` and ``Y`` can be either a CSR matrix or a precomputed
    ``SparseColumnIndex`` (from ``sparse_column_index``). Precomputing
    is useful when the same matrix is compared against many others.

    Parameters
    ----------
    X : csr_matrix, csr_array, or SparseColumnIndex, shape (n1, n_genes)
        Sparse expression matrix for group A. If a CSR matrix, must have
        non-negative values. Call ``X.eliminate_zeros()`` beforehand if
        the matrix may contain explicitly stored zeros.
    Y : csr_matrix, csr_array, or SparseColumnIndex, shape (n2, n_genes)
        Sparse expression matrix for group B. Must have the same number
        of columns as X.
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
    idx_a = _resolve_sparse(X, "X")
    idx_b = _resolve_sparse(Y, "Y")

    if idx_a.n_cols != idx_b.n_cols:
        raise ValueError(
            f"`X` and `Y` must have the same number of columns, "
            f"got {idx_a.n_cols} and {idx_b.n_cols}."
        )

    alt = _validate_alternative(alternative)

    stats, pvals = _sparse_mwu_batch(
        idx_a.data,
        idx_a.col_indptr,
        idx_a.col_order,
        idx_a.n_rows,
        idx_b.data,
        idx_b.col_indptr,
        idx_b.col_order,
        idx_b.n_rows,
        use_continuity,
        alt,
    )
    return MannWhitneyUResult(stats, pvals)


def mannwhitneyu_one_vs_rest_sparse(
    X,
    labels,
    n_groups=None,
    use_continuity=True,
    alternative="two-sided",
    parallel_axis="auto",
):
    """Sparse (CSR) counterpart of ``mannwhitneyu_one_vs_rest``.

    Works directly on a single CSR matrix spanning every group — no need to
    slice it into per-group matrices first — the sparse analogue of the dense
    function's "rank once, not once per group" optimization. Zeros are
    treated analytically (same zero-block trick as ``mannwhitneyu_sparse``),
    so this requires non-negative data.

    Parameters
    ----------
    X : csr_matrix or csr_array, shape (n_rows, n_cols)
        Sparse matrix with non-negative values, covering every labeled row.
        Call ``X.eliminate_zeros()`` beforehand if it may contain explicitly
        stored zeros.
    labels : array_like, shape (n_rows,)
        Integer group id for each row, in ``[0, n_groups)``. Every group must
        have at least one row.
    n_groups : int, optional
        Number of groups. Defaults to ``labels.max() + 1``.
    use_continuity : bool, optional
        Whether to apply continuity correction. Default True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis. Default 'two-sided'.
    parallel_axis : {'auto', 'groups', 'columns'}, optional
        See ``mannwhitneyu_one_vs_rest`` — a pure performance knob for the
        final reduction step, never affects the result. ``'auto'`` (default)
        picks whichever of ``n_groups``/``n_cols`` is larger.

    Returns
    -------
    result : MannWhitneyUResult
        Named tuple with ``statistic`` and ``pvalue`` arrays of shape
        ``(n_groups, n_cols)``.
    """
    X = _validate_csr(X, "X")
    n_rows = X.shape[0]
    labels, n_groups, group_sizes = _validate_labels(labels, n_rows, n_groups)
    alt = _validate_alternative(alternative)
    axis = _validate_parallel_axis(parallel_axis)

    data = np.ascontiguousarray(X.data, dtype=np.float64)
    indptr = np.ascontiguousarray(X.indptr)
    indices = np.ascontiguousarray(X.indices)
    col_indptr, col_order, row_of_nnz = _build_col_index_with_rows(
        indptr, indices, X.shape[1]
    )

    stats, pvals = _mannwhitneyu_one_vs_rest_sparse_batch(
        data,
        col_indptr,
        col_order,
        row_of_nnz,
        labels,
        n_groups,
        group_sizes,
        n_rows,
        use_continuity,
        alt,
        axis,
    )
    return MannWhitneyUResult(stats, pvals)
