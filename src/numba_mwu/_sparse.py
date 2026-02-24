"""Sparse-aware Mann-Whitney U test kernels for CSR matrices.

Works directly with CSR format (the standard for single-cell data) by
building a lightweight column index — a permutation array and column
pointers — without copying the data values. Memory overhead is one int
array of length nnz plus one int array of length (n_cols + 1).
"""

import math

import numba as nb
import numpy as np

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
        Indices into the CSR data/indices arrays, grouped by column.
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
def _sparse_mwu_column(
    data, row_indices, order, n_a, n_b, group_a, use_continuity, alternative
):
    """Compute Mann-Whitney U for a single gene from CSR data via indirection.

    Zeros are treated analytically — they form a contiguous block at the
    start of the sorted order (requires non-negative data).

    Parameters
    ----------
    data : float64 1-D array
        Full CSR data array (shared, read-only).
    row_indices : int 1-D array
        Full CSR row-index array built from indptr (shared, read-only).
    order : int64 1-D array
        Indices into data/row_indices for this column's nonzero entries.
    n_a, n_b : int
        Number of cells in group A and group B.
    group_a : bool 1-D array (length n_cells)
        True for cells in group A.
    use_continuity : bool
    alternative : int (0=two-sided, 1=less, 2=greater)

    Returns
    -------
    U1 : float64
    pvalue : float64
    """
    n = n_a + n_b
    nnz = order.shape[0]
    nz = n - nnz  # implicit zeros

    # --- All zeros: no evidence of difference ---
    if nnz == 0:
        return n_a * n_b / 2.0, 1.0

    # --- Gather this column's values into a contiguous work array ---
    col_vals = np.empty(nnz, dtype=np.float64)
    col_rows = np.empty(nnz, dtype=np.int64)
    for k in range(nnz):
        idx = order[k]
        col_vals[k] = data[idx]
        col_rows[k] = row_indices[idx]

    # --- Count nonzeros per group ---
    nnz_a = 0
    for k in range(nnz):
        if group_a[col_rows[k]]:
            nnz_a += 1
    nz_a = n_a - nnz_a

    # --- Sort nonzero values ---
    sort_idx = np.argsort(col_vals)

    # --- Walk sorted nonzeros: compute local ranks, tie correction,
    #     and sum of global ranks for group A nonzeros ---
    tie_term_nz = 0.0
    sum_global_ranks_a = 0.0

    i = 0
    while i < nnz:
        j = i
        while j < nnz - 1 and col_vals[sort_idx[j]] == col_vals[sort_idx[j + 1]]:
            j += 1

        tie_count = float(j - i + 1)
        tie_term_nz += tie_count * tie_count * tie_count - tie_count

        # Local average rank (1-based among nonzeros), shifted to global
        local_avg_rank = (i + j) / 2.0 + 1.0
        global_avg_rank = nz + local_avg_rank

        for k in range(i, j + 1):
            if group_a[col_rows[sort_idx[k]]]:
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


@nb.njit(parallel=True)
def _sparse_mwu_batch(
    csr_data,
    row_indices,
    col_indptr,
    col_order,
    n_cells,
    group_a,
    n_a,
    use_continuity,
    alternative,
):
    """Run sparse MWU test on each gene using CSR data + column index.

    Parameters
    ----------
    csr_data : float64 1-D array
        CSR data array (not copied).
    row_indices : int64 1-D array
        Row index for each entry in csr_data (precomputed from CSR indptr).
    col_indptr : int64 1-D array (n_genes + 1)
        Column pointers into col_order.
    col_order : int64 1-D array (nnz)
        Permutation mapping column-grouped positions to CSR data indices.
    n_cells : int
    group_a : bool 1-D array (n_cells)
    n_a : int
    use_continuity : bool
    alternative : int

    Returns
    -------
    U_out : float64 array (n_genes,)
    p_out : float64 array (n_genes,)
    """
    n_genes = col_indptr.shape[0] - 1
    n_b = n_cells - n_a
    U_out = np.empty(n_genes, dtype=np.float64)
    p_out = np.empty(n_genes, dtype=np.float64)

    for j in nb.prange(n_genes):  # type: ignore
        start = col_indptr[j]
        end = col_indptr[j + 1]
        order = col_order[start:end]
        U_out[j], p_out[j] = _sparse_mwu_column(
            csr_data, row_indices, order, n_a, n_b, group_a, use_continuity, alternative
        )

    return U_out, p_out


@nb.njit
def _expand_row_indices(indptr):
    """Expand CSR indptr into a flat row-index array.

    For each nonzero entry k, row_indices[k] = the row it belongs to.
    This is the inverse of indptr and avoids repeated binary search.
    """
    nnz = indptr[-1]
    n_rows = indptr.shape[0] - 1
    row_indices = np.empty(nnz, dtype=np.int64)
    for i in range(n_rows):
        for k in range(indptr[i], indptr[i + 1]):
            row_indices[k] = i
    return row_indices
