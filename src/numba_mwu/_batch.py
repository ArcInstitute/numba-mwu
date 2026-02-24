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
def _mannwhitneyu_columns(data, n1, use_continuity, alternative):
    """Run Mann-Whitney U test on each column of data, split at row n1.

    Parameters
    ----------
    data : 2D float64 array (n1 + n2, n_tests)
    n1 : int
        Number of rows belonging to the first sample.
    use_continuity : bool
    alternative : int (0=two-sided, 1=less, 2=greater)

    Returns
    -------
    U_out : float64 array (n_tests,)
    p_out : float64 array (n_tests,)
    """
    n_tests = data.shape[1]
    U_out = np.empty(n_tests, dtype=np.float64)
    p_out = np.empty(n_tests, dtype=np.float64)
    for i in nb.prange(n_tests):  # type: ignore
        col = data[:, i].copy()  # ensure contiguous
        x = col[:n1]
        y = col[n1:]
        U_out[i], p_out[i] = _mannwhitneyu_single(x, y, use_continuity, alternative)
    return U_out, p_out
