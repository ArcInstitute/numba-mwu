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
def _mwu_stats_one_vs_rest_by_group(
    rank_sum, tie_term, group_sizes, n_total, use_continuity, alternative
):
    """Convert per-group rank sums into (statistic, pvalue) via ``_mwu_stats_from_rank_sum``.

    Parallelizes over groups (outer ``prange``), columns inner — cache-friendly
    (each thread walks a contiguous row of ``rank_sum``) but only spreads work
    across ``n_groups`` parallel tasks, so it under-utilizes available cores
    whenever there are fewer groups than threads. See ``_mwu_stats_one_vs_rest``
    (the dispatcher both this and ``_mwu_stats_one_vs_rest_by_column`` are
    called through) for when each is chosen.

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


@nb.njit(parallel=True)
def _mwu_stats_one_vs_rest_by_column(
    rank_sum, tie_term, group_sizes, n_total, use_continuity, alternative
):
    """Same formula as ``_mwu_stats_one_vs_rest_by_group``, parallelized over
    columns instead of groups. Spreads work across ``n_cols`` parallel tasks —
    better core utilization when there are few groups but many columns (e.g. a
    handful of clusters tested against tens of thousands of genes), at the
    cost of a strided (column-wise) read of ``rank_sum`` per thread instead of
    a contiguous row.

    Parameters and returns are identical to ``_mwu_stats_one_vs_rest_by_group``.
    """
    n_groups, n_cols = rank_sum.shape
    U_out = np.empty((n_groups, n_cols), dtype=np.float64)
    p_out = np.empty((n_groups, n_cols), dtype=np.float64)

    for j in nb.prange(n_cols):  # type: ignore
        tt = tie_term[j]
        for g in range(n_groups):
            n1 = group_sizes[g]
            n2 = n_total - n1
            U_out[g, j], p_out[g, j] = _mwu_stats_from_rank_sum(
                rank_sum[g, j], n1, n2, tt, use_continuity, alternative
            )
    return U_out, p_out


def _select_parallel_axis(n_groups, n_cols):
    """ "auto" heuristic: prefer "groups" unless there aren't enough groups to
    keep every thread busy.

    Empirically benchmarked (see numba-mwu's CLAUDE.md): once ``n_groups``
    reaches the thread count, "groups" wins regardless of ``n_cols`` — the
    strided access "columns" pays for scales with ``n_groups`` (the axis it
    loops over internally), independent of how many column-tasks run in
    parallel, so a larger ``n_cols`` never rescues it. "columns" only wins in
    the specific regime where ``n_groups`` itself can't fill the thread pool
    (the common "few clusters, many genes" marker-feature workflow); there,
    whichever axis is larger determines which keeps more threads busy.
    """
    n_threads = nb.get_num_threads()
    if n_groups >= n_threads:
        return "groups"
    return "columns" if n_cols > n_groups else "groups"


def _mwu_stats_one_vs_rest(
    rank_sum,
    tie_term,
    group_sizes,
    n_total,
    use_continuity,
    alternative,
    parallel_axis="auto",
):
    """Dispatch to the by-group or by-column parallel reduction kernel.

    Shared by the dense (this module) and sparse (``_sparse.py``) one-vs-rest
    pipelines — both produce ``rank_sum``/``tie_term`` differently but reduce
    through one of these two kernels, which compute the identical formula.
    Every ``(group, column)`` cell is computed independently (no cross-element
    reduction), so ``parallel_axis`` only affects performance — never the
    numeric result.

    Parameters
    ----------
    parallel_axis : {'auto', 'groups', 'columns'}, optional
        Which axis to parallelize the reduction over. ``'auto'`` (default)
        picks "groups" once ``n_groups`` reaches the thread count, else
        whichever of ``n_groups``/``n_cols`` is larger — see
        ``_select_parallel_axis``.
    """
    n_groups, n_cols = rank_sum.shape
    axis = (
        _select_parallel_axis(n_groups, n_cols)
        if parallel_axis == "auto"
        else parallel_axis
    )
    if axis == "columns":
        return _mwu_stats_one_vs_rest_by_column(
            rank_sum, tie_term, group_sizes, n_total, use_continuity, alternative
        )
    return _mwu_stats_one_vs_rest_by_group(
        rank_sum, tie_term, group_sizes, n_total, use_continuity, alternative
    )


def _mannwhitneyu_one_vs_rest_columns(
    X,
    labels,
    n_groups,
    group_sizes,
    use_continuity,
    alternative,
    parallel_axis="auto",
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
    parallel_axis : {'auto', 'groups', 'columns'}, optional
        See ``_mwu_stats_one_vs_rest``.

    Returns
    -------
    U_out : float64 array (n_groups, n_cols)
    p_out : float64 array (n_groups, n_cols)
    """
    rank_sum, tie_term = _one_vs_rest_rank_sums_dense(X, labels, n_groups)
    return _mwu_stats_one_vs_rest(
        rank_sum,
        tie_term,
        group_sizes,
        X.shape[0],
        use_continuity,
        alternative,
        parallel_axis,
    )
