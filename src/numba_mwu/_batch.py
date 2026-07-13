"""Parallel batch Mann-Whitney U tests using numba.prange."""

import numba as nb
import numpy as np

from ._core import _mannwhitneyu_single, _mwu_stats_from_rank_sum


@nb.njit(parallel=True)
def _mannwhitneyu_rows(X, y, use_continuity, alternative):
    """Run Mann-Whitney U test for each row of X against y.

    Parameters
    ----------
    X : 2D float64 array (n_tests, n1)
    y : 1D float64 array (n2,)
    use_continuity : bool
    alternative : int (0=two-sided, 1=less, 2=greater)

    Returns
    -------
    U_out : float64 array (n_tests,)
    p_out : float64 array (n_tests,)
    """
    n_tests = X.shape[0]
    U_out = np.empty(n_tests, dtype=np.float64)
    p_out = np.empty(n_tests, dtype=np.float64)
    for i in nb.prange(n_tests):  # type: ignore
        U_out[i], p_out[i] = _mannwhitneyu_single(
            X[i].copy(), y, use_continuity, alternative
        )
    return U_out, p_out


@nb.njit(parallel=True)
def _mannwhitneyu_columns(X, Y, use_continuity, alternative):
    """Run Mann-Whitney U test on each column of X vs corresponding column of Y.

    Parameters
    ----------
    X : 2D float64 array (n1, n_genes)
        First group (one column per gene).
    Y : 2D float64 array (n2, n_genes)
        Second group (one column per gene). Must have same number of columns as X.
    use_continuity : bool
    alternative : int (0=two-sided, 1=less, 2=greater)

    Returns
    -------
    U_out : float64 array (n_genes,)
    p_out : float64 array (n_genes,)
    """
    n_genes = X.shape[1]
    U_out = np.empty(n_genes, dtype=np.float64)
    p_out = np.empty(n_genes, dtype=np.float64)
    for i in nb.prange(n_genes):  # type: ignore
        x = X[:, i].copy()  # ensure contiguous
        y = Y[:, i].copy()
        U_out[i], p_out[i] = _mannwhitneyu_single(x, y, use_continuity, alternative)
    return U_out, p_out


@nb.njit(parallel=True)
def _one_vs_rest_rank_sums_dense(X, labels, n_groups):
    """Rank each column once; reduce to per-group rank sums and a tie term.

    ``labels[i]`` is the group id (``0..n_groups-1``) of row ``i``. In a
    one-vs-rest comparison, "group vs rest" always partitions the *entire*
    input regardless of which group is being tested — so ranking (and the
    tie-correction term, which depends only on the full column's tie
    structure) needs to happen once per column rather than once per
    ``(group, column)`` pair.

    Parameters
    ----------
    X : 2D float64 array (n_rows, n_cols)
        All rows to be compared, e.g. a matrix already restricted to valid
        (labeled) rows.
    labels : 1D int64 array (n_rows,)
        Group id of each row, in ``[0, n_groups)``.
    n_groups : int

    Returns
    -------
    rank_sum : float64 array (n_groups, n_cols)
    tie_term : float64 array (n_cols,)
        Sum(t_i^3 - t_i) over the full column's tie groups; independent of
        the group split.
    """
    n_rows, n_cols = X.shape
    rank_sum = np.zeros((n_groups, n_cols), dtype=np.float64)
    tie_term = np.zeros(n_cols, dtype=np.float64)

    for j in nb.prange(n_cols):  # type: ignore
        col = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            col[i] = X[i, j]
        order = np.argsort(col)

        local_tie_term = 0.0
        i2 = 0
        while i2 < n_rows:
            k = i2
            while k < n_rows - 1 and col[order[k]] == col[order[k + 1]]:
                k += 1
            tie_count = float(k - i2 + 1)
            local_tie_term += tie_count * tie_count * tie_count - tie_count
            avg_rank = (i2 + k) / 2.0 + 1.0
            for m in range(i2, k + 1):
                row = order[m]
                rank_sum[labels[row], j] += avg_rank
            i2 = k + 1
        tie_term[j] = local_tie_term

    return rank_sum, tie_term


@nb.njit(parallel=True)
def _mwu_stats_one_vs_rest(
    rank_sum, tie_term, group_sizes, n_total, use_continuity, alternative
):
    """Convert per-group rank sums into (statistic, pvalue) via ``_mwu_stats_from_rank_sum``.

    Shared by the dense (this module) and sparse (``_sparse.py``) one-vs-rest
    kernels — both produce ``rank_sum``/``tie_term`` differently but reduce
    through this identical formula.

    Parameters
    ----------
    rank_sum : float64 array (n_groups, n_cols)
    tie_term : float64 array (n_cols,)
    group_sizes : int64 array (n_groups,)
    n_total : int
    use_continuity : bool
    alternative : int (0=two-sided, 1=less, 2=greater)

    Returns
    -------
    U_out : float64 array (n_groups, n_cols)
        ``U_out[g]`` is always the U statistic for group ``g`` (the "x" side),
        matching the convention of every other function in this library.
    p_out : float64 array (n_groups, n_cols)
    """
    n_groups, n_cols = rank_sum.shape
    U_out = np.empty((n_groups, n_cols), dtype=np.float64)
    p_out = np.empty((n_groups, n_cols), dtype=np.float64)

    for g in nb.prange(n_groups):  # type: ignore
        n1 = group_sizes[g]
        n2 = n_total - n1
        for j in range(n_cols):
            U_out[g, j], p_out[g, j] = _mwu_stats_from_rank_sum(
                rank_sum[g, j], n1, n2, tie_term[j], use_continuity, alternative
            )
    return U_out, p_out


def _mannwhitneyu_one_vs_rest_columns(
    X, labels, n_groups, group_sizes, use_continuity, alternative
):
    """One-shot one-vs-rest Mann-Whitney U test across every group, dense input.

    Plain Python (not JIT-compiled itself) — just sequences the two JIT-compiled
    kernels below; there is nothing here for numba to parallelize.

    Parameters
    ----------
    X : 2D float64 array (n_rows, n_cols)
    labels : 1D int64 array (n_rows,)
        Group id of each row, in ``[0, n_groups)``.
    n_groups : int
    group_sizes : int64 array (n_groups,)
        Number of rows belonging to each group (``np.bincount(labels)``).
    use_continuity : bool
    alternative : int (0=two-sided, 1=less, 2=greater)

    Returns
    -------
    U_out : float64 array (n_groups, n_cols)
    p_out : float64 array (n_groups, n_cols)
    """
    rank_sum, tie_term = _one_vs_rest_rank_sums_dense(X, labels, n_groups)
    return _mwu_stats_one_vs_rest(
        rank_sum, tie_term, group_sizes, X.shape[0], use_continuity, alternative
    )
