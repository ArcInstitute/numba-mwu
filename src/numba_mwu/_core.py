"""Numba-accelerated core functions for Mann-Whitney U test."""

import math

import numba as nb
import numpy as np

# Alternative hypothesis encoding
TWO_SIDED = 0
LESS = 1
GREATER = 2


@nb.njit
def _ndtr(x: float) -> float:
    """Normal CDF using math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@nb.njit
def _rankdata_avg(x):
    """Rank data using average method, returning ranks and tie group sizes.

    Parameters
    ----------
    x : 1D float64 array

    Returns
    -------
    ranks : float64 array of same length as x
    tie_sizes : float64 array of length n_groups (number of distinct values)
    """
    n = x.shape[0]
    idx = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    tie_sizes = np.empty(n, dtype=np.float64)
    n_ties = 0

    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[idx[j]] == x[idx[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        tie_count = j - i + 1
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        tie_sizes[n_ties] = float(tie_count)
        n_ties += 1
        i = j + 1

    return ranks, tie_sizes[:n_ties]


@nb.njit
def _mwu_stats_from_rank_sum(R1, n1, n2, tie_term, use_continuity, alternative):
    """Reduce a rank sum and tie-correction term into (U1, pvalue).

    This is the asymptotic-method formula shared by every kernel in the
    library: the single-pair path (`_mannwhitneyu_single`, below) computes
    `R1`/`tie_term` by ranking the concatenation of two samples, while the
    one-vs-rest batch kernels (`_batch.py`, `_sparse.py`) compute the same two
    quantities once per gene across *all* groups simultaneously (ranking is
    invariant to which group is being tested, in a one-vs-rest split). Both
    paths reduce through this single function so their numeric results can
    never drift apart.

    Parameters
    ----------
    R1 : float
        Sum of ranks for sample x within the combined (x, y) ranking.
    n1, n2 : int
        Sample sizes of x and y (y may represent "all other rows").
    tie_term : float
        Sum(t_i^3 - t_i) over tie groups of the full combined ranking.
    use_continuity : bool
        Whether to apply continuity correction.
    alternative : int
        0 = two-sided, 1 = less, 2 = greater

    Returns
    -------
    U1 : float
        U statistic for sample x.
    pvalue : float
    """
    n = n1 + n2
    U1 = R1 - n1 * (n1 + 1.0) / 2.0
    U2 = n1 * n2 - U1

    # Select statistic and multiplier based on alternative
    if alternative == GREATER:
        U = U1
        f = 1.0
    elif alternative == LESS:
        U = U2
        f = 1.0
    else:  # TWO_SIDED
        U = max(U1, U2)
        f = 2.0

    # Standard deviation with tie correction
    mu = n1 * n2 / 2.0
    denom = n * (n - 1.0)
    var_inner = (n + 1.0) - tie_term / denom
    s_sq = n1 * n2 / 12.0 * var_inner

    if s_sq <= 0.0:
        # All values identical — no evidence of difference
        return U1, 1.0

    s = math.sqrt(s_sq)

    numerator = U - mu
    if use_continuity:
        numerator -= 0.5

    z = numerator / s
    p = _ndtr(-z) * f

    # Clip to [0, 1]
    if p > 1.0:
        p = 1.0
    if p < 0.0:
        p = 0.0

    return U1, p


@nb.njit
def _mannwhitneyu_single(x, y, use_continuity, alternative):
    """Compute Mann-Whitney U test for two 1D samples (asymptotic method).

    Parameters
    ----------
    x : 1D float64 array
    y : 1D float64 array
    use_continuity : bool
        Whether to apply continuity correction.
    alternative : int
        0 = two-sided, 1 = less, 2 = greater

    Returns
    -------
    U1 : float
        U statistic for sample x.
    pvalue : float
    """
    n1 = x.shape[0]
    n2 = y.shape[0]
    n = n1 + n2

    # Concatenate and rank
    xy = np.empty(n, dtype=np.float64)
    for i in range(n1):
        xy[i] = x[i]
    for i in range(n2):
        xy[n1 + i] = y[i]

    ranks, t = _rankdata_avg(xy)

    # Sum of ranks for first sample
    R1 = 0.0
    for i in range(n1):
        R1 += ranks[i]

    # Tie correction term: sum(t_i^3 - t_i)
    tie_term = 0.0
    for i in range(t.shape[0]):
        ti = t[i]
        tie_term += ti * ti * ti - ti

    return _mwu_stats_from_rank_sum(R1, n1, n2, tie_term, use_continuity, alternative)
