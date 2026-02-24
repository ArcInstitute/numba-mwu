@xp_capabilities(
    skip_backends=[
        ("cupy", "`repeat` can't handle array second arg"),
        ("dask.array", "no `take_along_axis`"),
    ]
)
def rankdata(a, method="average", *, axis=None, nan_policy="propagate"):
    """Assign ranks to data, dealing with ties appropriately.

    By default (``axis=None``), the data array is first flattened, and a flat
    array of ranks is returned. Separately reshape the rank array to the
    shape of the data array if desired (see Examples).

    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.  See [1]_ for further discussion of ranking methods.

    Parameters
    ----------
    a : array_like
        The array of values to be ranked.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to assign ranks to tied elements.
        The following methods are available (default is 'average'):

        * 'average': The average of the ranks that would have been assigned to
          all the tied values is assigned to each value.
        * 'min': The minimum of the ranks that would have been assigned to all
          the tied values is assigned to each value.  (This is also
          referred to as "competition" ranking.)
        * 'max': The maximum of the ranks that would have been assigned to all
          the tied values is assigned to each value.
        * 'dense': Like 'min', but the rank of the next highest element is
          assigned the rank immediately after those assigned to the tied
          elements.
        * 'ordinal': All values are given a distinct rank, corresponding to
          the order that the values occur in `a`.

    axis : {None, int}, optional
        Axis along which to perform the ranking. If ``None``, the data array
        is first flattened.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': propagates nans through the rank calculation
        * 'omit': performs the calculations ignoring nan values
        * 'raise': raises an error

        .. note::

            When `nan_policy` is 'propagate', the output is an array of *all*
            nans because ranks relative to nans in the input are undefined.
            When `nan_policy` is 'omit', nans in `a` are ignored when ranking
            the other values, and the corresponding locations of the output
            are nan.

        .. versionadded:: 1.10

    Returns
    -------
    ranks : ndarray
         An array of size equal to the size of `a`, containing rank
         scores. The dtype is the result dtype of `a` and a Python float.

    References
    ----------
    .. [1] "Ranking", https://en.wikipedia.org/wiki/Ranking

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import rankdata
    >>> rankdata([0, 2, 3, 2])
    array([1. , 2.5, 4. , 2.5])
    >>> rankdata([0, 2, 3, 2], method='min')
    array([1., 2., 4., 2.])
    >>> rankdata([0, 2, 3, 2], method='max')
    array([1., 3., 4., 3.])
    >>> rankdata([0, 2, 3, 2], method='dense')
    array([1., 2., 3., 2.])
    >>> rankdata([0, 2, 3, 2], method='ordinal')
    array([1., 2., 4., 3.])
    >>> rankdata([[0, 2], [3, 2]]).reshape(2, 2)
    array([[1. , 2.5],
           [4. , 2.5]])
    >>> rankdata([[0, 2, 2], [3, 2, 5]], axis=1)
    array([[1. , 2.5, 2.5],
           [2. , 1. , 3. ]])
    >>> rankdata([0, 2, 3, np.nan, -2, np.nan], nan_policy="propagate")
    array([nan, nan, nan, nan, nan, nan])
    >>> rankdata([0, 2, 3, np.nan, -2, np.nan], nan_policy="omit")
    array([ 2.,  3.,  4., nan,  1., nan])

    """
    methods = ("average", "min", "max", "dense", "ordinal")
    if method not in methods:
        raise ValueError(f'unknown method "{method}"')

    xp = array_namespace(a)
    x = xp.asarray(a)

    if axis is None:
        x = xp_ravel(x)
        axis = -1

    if xp_size(x) == 0:
        dtype = xp_result_type(x, force_floating=True, xp=xp)
        return xp.empty_like(x, dtype=dtype)

    contains_nan = _contains_nan(x, nan_policy)

    x = xp_swapaxes(x, axis, -1, xp=xp)
    ranks = _rankdata(x, method, xp=xp)

    # JIT won't allow use of `contains_nan` for control flow here, so we always have to
    # run this with JIT.
    if is_lazy_array(x) or contains_nan:
        i_nan = (
            xp.isnan(x)
            if nan_policy == "omit"
            else xp.any(xp.isnan(x), axis=-1, keepdims=True)
        )
        i_nan = xp.broadcast_to(i_nan, ranks.shape)
        ranks = xpx.at(ranks)[i_nan].set(xp.nan)

    ranks = xp_swapaxes(ranks, axis, -1, xp=xp)
    return ranks


def _order_ranks(ranks, j, *, xp):
    # Reorder ascending order `ranks` according to `j`
    xp = array_namespace(ranks) if xp is None else xp
    if is_numpy(xp) or is_cupy(xp):
        ordered_ranks = xp.empty(j.shape, dtype=ranks.dtype)
        xp.put_along_axis(ordered_ranks, j, ranks, axis=-1)
    else:
        # `put_along_axis` not in array API (data-apis/array-api#177)
        #  so argsort the argsort and take_along_axis...
        j_inv = xp.argsort(j, axis=-1, stable=True)
        ordered_ranks = xp.take_along_axis(ranks, j_inv, axis=-1)
    return ordered_ranks


def _rankdata(x, method, return_sorted=False, return_ties=False, xp=None):
    # Rank data `x` by desired `method`; `return_ties`/`return_sorted` data  if desired
    xp = array_namespace(x) if xp is None else xp
    dtype = xp_result_type(x, force_floating=True, xp=xp)

    if is_jax(xp):
        import jax.scipy.stats as jax_stats

        ranks = jax_stats.rankdata(x, method=method, axis=-1)
        ranks = xp.astype(ranks, dtype, copy=False)
        out = [ranks]
        y = xp.sort(x, axis=-1) if (return_ties or return_sorted) else None
        if return_sorted:
            out.append(y)
        if return_ties:
            max_ranks = jax_stats.rankdata(y, method="max", axis=-1)
            t = xp.diff(max_ranks, axis=-1, prepend=0)
            t = xp.astype(t, dtype, copy=False)
            out.append(t)
        return out[0] if len(out) == 1 else tuple(out)

    shape = x.shape

    # Get sort order
    j = xp.argsort(x, axis=-1, stable=True)
    ordinal_ranks = xp.broadcast_to(xp.arange(1, shape[-1] + 1, dtype=dtype), shape)

    # Ordinal ranks is very easy because ties don't matter. We're done.
    if method == "ordinal":
        return _order_ranks(ordinal_ranks, j, xp=xp)  # never return ties or sorted data

    # Sort array
    y = xp.take_along_axis(x, j, axis=-1)
    # Logical indices of unique elements
    i = xp.concat(
        [xp.ones(shape[:-1] + (1,), dtype=xp.bool), y[..., :-1] != y[..., 1:]], axis=-1
    )

    # Integer indices of unique elements
    indices = xp.arange(xp_size(y))[xp.reshape(i, (-1,))]  # i gets raveled
    # Counts of unique elements
    counts = xp.diff(indices, append=xp.asarray([xp_size(y)], dtype=indices.dtype))

    # Compute `'min'`, `'max'`, and `'mid'` ranks of unique elements
    if method == "min":
        ranks = ordinal_ranks[i]
    elif method == "max":
        ranks = ordinal_ranks[i] + xp.astype(counts, dtype) - 1
    elif method == "average":
        # array API doesn't promote integers to floats
        ranks = ordinal_ranks[i] + (xp.astype(counts, dtype) - 1) / 2
    elif method == "dense":
        ranks = xp.cumulative_sum(xp.astype(i, dtype, copy=False), axis=-1)[i]

    ranks = xp.reshape(xp.repeat(ranks, counts), shape)
    ranks = _order_ranks(ranks, j, xp=xp)
    if not (return_sorted or return_ties):
        return ranks

    out = [ranks]

    if return_sorted:
        out.append(y)

    if return_ties:
        # Tie information is returned in a format that is useful to functions that
        # rely on this (private) function. Example:
        # >>> x = np.asarray([3, 2, 1, 2, 2, 2, 1])
        # >>> _, t = _rankdata(x, 'average', return_ties=True)
        # >>> t  # array([2., 0., 4., 0., 0., 0., 1.])  # two 1s, four 2s, and one 3
        # Unlike ranks, tie counts are *not* reordered to correspond with the order of
        # the input; e.g. the number of appearances of the lowest rank element comes
        # first. This is a useful format because:
        # - The shape of the result is the shape of the input. Different slices can
        #   have different numbers of tied elements but not result in a ragged array.
        # - Functions that use `t` usually don't need to which each element of the
        #   original array is associated with each tie count; they perform a reduction
        #   over the tie counts onnly. The tie counts are naturally computed in a
        #   sorted order, so this does not unnecessarily reorder them.
        # - One exception is `wilcoxon`, which needs the number of zeros. Zeros always
        #   have the lowest rank, so it is easy to find them at the zeroth index.
        t = xp.zeros(shape, dtype=dtype)
        t = xpx.at(t)[i].set(xp.astype(counts, dtype, copy=False))
        out.append(t)

    return out
