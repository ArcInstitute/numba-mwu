"""Sparse-aware Mann-Whitney U test kernels for CSR matrices.

Works directly with CSR format (the standard for single-cell data) by
building a lightweight column index — a permutation array and column
pointers — without copying the data values. Memory overhead is one int
array of length nnz plus one int array of length (n_cols + 1) per matrix.
"""

import math

import numba as nb
import numpy as np

from ._batch import _mwu_stats_one_vs_rest
from ._core import GREATER, LESS, _ndtr


@nb.njit
def _build_col_index(csr_indptr, csr_indices, n_cols):
    """Build column pointers and a permutation from CSR arrays.

    Returns
    -------
    col_indptr : int64 array (n_cols + 1)
        col_indptr[j] .. col_indptr[j+1] gives the range of entries for
        column j in the col_order array.
    col_order : int64 array (nnz)
        Indices into the CSR data arrays, grouped by column.
        data[col_order[col_indptr[j]:col_indptr[j+1]]] gives the nonzero
        values for column j.
    """
    nnz = csr_indices.shape[0]

    # Count entries per column
    col_counts = np.zeros(n_cols, dtype=np.int64)
    for k in range(nnz):
        col_counts[csr_indices[k]] += 1

    # Build col_indptr via cumulative sum
    col_indptr = np.empty(n_cols + 1, dtype=np.int64)
    col_indptr[0] = 0
    for j in range(n_cols):
        col_indptr[j + 1] = col_indptr[j] + col_counts[j]

    # Fill col_order (scatter each entry to its column bucket)
    pos = np.empty(n_cols, dtype=np.int64)
    for j in range(n_cols):
        pos[j] = col_indptr[j]

    col_order = np.empty(nnz, dtype=np.int64)
    n_rows = csr_indptr.shape[0] - 1
    for i in range(n_rows):
        for k in range(csr_indptr[i], csr_indptr[i + 1]):
            c = csr_indices[k]
            col_order[pos[c]] = k
            pos[c] += 1

    return col_indptr, col_order


@nb.njit
def _build_col_index_with_rows(csr_indptr, csr_indices, n_cols):
    """Like ``_build_col_index``, but also returns each entry's originating row.

    The pairwise sparse kernels never need to know which row a nonzero value
    came from (group membership is already fixed by which of the two input
    matrices it appears in). The one-vs-rest kernel merges every group into a
    single matrix, so it needs each nonzero's row to look up its group id.

    Returns
    -------
    col_indptr, col_order : see ``_build_col_index``
    row_of_nnz : int64 array (nnz)
        ``row_of_nnz[m]`` is the original CSR row of the entry at
        ``col_order[m]``.
    """
    col_indptr, col_order = _build_col_index(csr_indptr, csr_indices, n_cols)

    nnz = csr_indices.shape[0]
    n_rows = csr_indptr.shape[0] - 1
    row_of_pos = np.empty(nnz, dtype=np.int64)
    for i in range(n_rows):
        for k in range(csr_indptr[i], csr_indptr[i + 1]):
            row_of_pos[k] = i

    row_of_nnz = np.empty(nnz, dtype=np.int64)
    for m in range(nnz):
        row_of_nnz[m] = row_of_pos[col_order[m]]

    return col_indptr, col_order, row_of_nnz


@nb.njit
def _sparse_mwu_column(vals_a, vals_b, n_a, n_b, use_continuity, alternative):
    """Compute Mann-Whitney U for a single gene from two groups' nonzero values.

    Zeros are treated analytically — they form a contiguous block at the
    start of the sorted order (requires non-negative data).

    Parameters
    ----------
    vals_a : float64 1-D array
        Nonzero values for this gene in group A.
    vals_b : float64 1-D array
        Nonzero values for this gene in group B.
    n_a, n_b : int
        Total number of cells in group A and group B (including zeros).
    use_continuity : bool
    alternative : int (0=two-sided, 1=less, 2=greater)

    Returns
    -------
    U1 : float64
    pvalue : float64
    """
    n = n_a + n_b
    nnz_a = vals_a.shape[0]
    nnz_b = vals_b.shape[0]
    nnz = nnz_a + nnz_b
    nz = n - nnz  # total implicit zeros
    nz_a = n_a - nnz_a  # zeros in group A

    # --- All zeros: no evidence of difference ---
    if nnz == 0:
        return n_a * n_b / 2.0, 1.0

    # --- Merge nonzero values into a single array with group tags ---
    all_vals = np.empty(nnz, dtype=np.float64)
    is_a = np.empty(nnz, dtype=np.int8)
    for k in range(nnz_a):
        all_vals[k] = vals_a[k]
        is_a[k] = 1
    for k in range(nnz_b):
        all_vals[nnz_a + k] = vals_b[k]
        is_a[nnz_a + k] = 0

    # --- Sort nonzero values ---
    sort_idx = np.argsort(all_vals)

    # --- Walk sorted nonzeros: compute local ranks, tie correction,
    #     and sum of global ranks for group A nonzeros ---
    tie_term_nz = 0.0
    sum_global_ranks_a = 0.0

    i = 0
    while i < nnz:
        j = i
        while j < nnz - 1 and all_vals[sort_idx[j]] == all_vals[sort_idx[j + 1]]:
            j += 1

        tie_count = float(j - i + 1)
        tie_term_nz += tie_count * tie_count * tie_count - tie_count

        # Local average rank (1-based among nonzeros), shifted to global
        local_avg_rank = (i + j) / 2.0 + 1.0
        global_avg_rank = nz + local_avg_rank

        for k in range(i, j + 1):
            if is_a[sort_idx[k]] == 1:
                sum_global_ranks_a += global_avg_rank

        i = j + 1

    # --- R1: total rank sum for group A ---
    zero_avg_rank = (nz + 1.0) / 2.0
    R1 = nz_a * zero_avg_rank + sum_global_ranks_a

    # --- U statistic ---
    U1 = R1 - n_a * (n_a + 1.0) / 2.0
    U2 = n_a * n_b - U1

    # --- Tie correction (zero block + nonzero tie groups) ---
    tie_term = (float(nz) * float(nz) * float(nz) - float(nz)) + tie_term_nz

    # --- Variance and z-score ---
    mu = n_a * n_b / 2.0
    denom = float(n) * float(n - 1)
    var_inner = (n + 1.0) - tie_term / denom
    s_sq = n_a * n_b / 12.0 * var_inner

    if s_sq <= 0.0:
        return U1, 1.0

    s = math.sqrt(s_sq)

    if alternative == GREATER:
        U = U1
        f = 1.0
    elif alternative == LESS:
        U = U2
        f = 1.0
    else:  # TWO_SIDED
        U = max(U1, U2)
        f = 2.0

    numerator = U - mu
    if use_continuity:
        numerator -= 0.5

    z = numerator / s
    p = _ndtr(-z) * f

    if p > 1.0:
        p = 1.0
    if p < 0.0:
        p = 0.0

    return U1, p


@nb.njit
def _gather_col_vals(csr_data, col_order, start, end):
    """Gather nonzero values for a single column from CSR data via col_order."""
    n = end - start
    vals = np.empty(n, dtype=np.float64)
    for k in range(n):
        vals[k] = csr_data[col_order[start + k]]
    return vals


@nb.njit(parallel=True)
def _sparse_mwu_batch(
    data_a,
    col_indptr_a,
    col_order_a,
    n_a,
    data_b,
    col_indptr_b,
    col_order_b,
    n_b,
    use_continuity,
    alternative,
):
    """Run sparse MWU test on each gene using two CSR matrices' column indices.

    Parameters
    ----------
    data_a : float64 1-D array — CSR data for group A
    col_indptr_a : int64 1-D array (n_genes + 1) — column pointers for A
    col_order_a : int64 1-D array (nnz_a) — column permutation for A
    n_a : int — total rows in A
    data_b : float64 1-D array — CSR data for group B
    col_indptr_b : int64 1-D array (n_genes + 1) — column pointers for B
    col_order_b : int64 1-D array (nnz_b) — column permutation for B
    n_b : int — total rows in B
    use_continuity : bool
    alternative : int

    Returns
    -------
    U_out : float64 array (n_genes,)
    p_out : float64 array (n_genes,)
    """
    n_genes = col_indptr_a.shape[0] - 1
    U_out = np.empty(n_genes, dtype=np.float64)
    p_out = np.empty(n_genes, dtype=np.float64)

    for j in nb.prange(n_genes):  # type: ignore
        vals_a = _gather_col_vals(
            data_a, col_order_a, col_indptr_a[j], col_indptr_a[j + 1]
        )
        vals_b = _gather_col_vals(
            data_b, col_order_b, col_indptr_b[j], col_indptr_b[j + 1]
        )
        U_out[j], p_out[j] = _sparse_mwu_column(
            vals_a, vals_b, n_a, n_b, use_continuity, alternative
        )

    return U_out, p_out


@nb.njit(parallel=True)
def _one_vs_rest_rank_sums_sparse(
    data, col_indptr, col_order, row_of_nnz, labels, n_groups, group_sizes, n_rows
):
    """Sparse (CSR-column-index) counterpart of ``_one_vs_rest_rank_sums_dense``.

    Zeros form a contiguous block at the bottom of each column's sorted order
    (data is assumed non-negative); a group's zero rows contribute
    ``count * zero_avg_rank`` to its rank sum without visiting them
    individually, mirroring the zero-block trick used by ``_sparse_mwu_column``.

    Parameters
    ----------
    data, col_indptr, col_order, row_of_nnz : see ``_build_col_index_with_rows``
    labels : int64 array (n_rows,)
        Group id of each row, in ``[0, n_groups)``.
    n_groups : int
    group_sizes : int64 array (n_groups,)
    n_rows : int

    Returns
    -------
    rank_sum : float64 array (n_groups, n_cols)
    tie_term : float64 array (n_cols,)
    """
    n_cols = col_indptr.shape[0] - 1
    rank_sum = np.zeros((n_groups, n_cols), dtype=np.float64)
    tie_term = np.zeros(n_cols, dtype=np.float64)

    for j in nb.prange(n_cols):  # type: ignore
        start = col_indptr[j]
        end = col_indptr[j + 1]
        nnz = end - start
        nz = n_rows - nnz

        nnz_count = np.zeros(n_groups, dtype=np.int64)
        group_rank_sum = np.zeros(n_groups, dtype=np.float64)
        tie_term_nz = 0.0

        if nnz > 0:
            vals = np.empty(nnz, dtype=np.float64)
            rows = np.empty(nnz, dtype=np.int64)
            for k in range(nnz):
                vals[k] = data[col_order[start + k]]
                rows[k] = row_of_nnz[start + k]
                nnz_count[labels[rows[k]]] += 1

            order = np.argsort(vals)
            i2 = 0
            while i2 < nnz:
                k = i2
                while k < nnz - 1 and vals[order[k]] == vals[order[k + 1]]:
                    k += 1
                tie_count = float(k - i2 + 1)
                tie_term_nz += tie_count * tie_count * tie_count - tie_count
                global_avg = float(nz) + (i2 + k) / 2.0 + 1.0
                for m in range(i2, k + 1):
                    g = labels[rows[order[m]]]
                    group_rank_sum[g] += global_avg
                i2 = k + 1

        zero_avg_rank = (float(nz) + 1.0) / 2.0
        for g in range(n_groups):
            rank_sum[g, j] = (
                float(group_sizes[g] - nnz_count[g]) * zero_avg_rank + group_rank_sum[g]
            )
        tie_term[j] = (float(nz) * float(nz) * float(nz) - float(nz)) + tie_term_nz

    return rank_sum, tie_term


def _mannwhitneyu_one_vs_rest_sparse_batch(
    data,
    col_indptr,
    col_order,
    row_of_nnz,
    labels,
    n_groups,
    group_sizes,
    n_rows,
    use_continuity,
    alternative,
    parallel_axis="auto",
):
    """One-shot one-vs-rest Mann-Whitney U test across every group, sparse input.

    Plain Python (not JIT-compiled itself) — just sequences the two JIT-compiled
    kernels (rank sums, then the shared reduction from ``_batch.py``).

    Parameters
    ----------
    parallel_axis : {'auto', 'groups', 'columns'}, optional
        See ``_mwu_stats_one_vs_rest`` in ``_batch.py``.

    Returns
    -------
    U_out : float64 array (n_groups, n_cols)
    p_out : float64 array (n_groups, n_cols)
    """
    rank_sum, tie_term = _one_vs_rest_rank_sums_sparse(
        data, col_indptr, col_order, row_of_nnz, labels, n_groups, group_sizes, n_rows
    )
    return _mwu_stats_one_vs_rest(
        rank_sum,
        tie_term,
        group_sizes,
        n_rows,
        use_continuity,
        alternative,
        parallel_axis,
    )
