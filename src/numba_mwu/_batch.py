"""Parallel batch Mann-Whitney U tests using numba.prange."""

import numba as nb
import numpy as np

from ._core import _mannwhitneyu_single


@nb.njit(parallel=True)
def _mannwhitneyu_batch(X, y, use_continuity, alternative):
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
