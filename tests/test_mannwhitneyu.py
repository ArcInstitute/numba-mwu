"""Tests validating numba_mwu against scipy.stats.mannwhitneyu."""

import numpy as np
import pytest
from scipy import stats

from numba_mwu import mannwhitneyu, mannwhitneyu_batch, mannwhitneyu_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scipy_mwu(x, y, use_continuity=True, alternative="two-sided"):
    """Wrapper around scipy for the asymptotic method."""
    return stats.mannwhitneyu(
        x, y, use_continuity=use_continuity, alternative=alternative, method="asymptotic"
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    """Compare single-test output against scipy."""

    def test_small_no_ties(self):
        """Scipy docs example: males vs females diagnosis age."""
        x = np.array([19, 22, 16, 29, 24], dtype=np.float64)
        y = np.array([20, 11, 17, 12], dtype=np.float64)
        result = mannwhitneyu(x, y)
        expected = _scipy_mwu(x, y)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_with_ties(self):
        """Integer data with repeated values."""
        x = np.array([1, 2, 3, 3, 4, 5], dtype=np.float64)
        y = np.array([2, 3, 3, 4, 5, 6], dtype=np.float64)
        result = mannwhitneyu(x, y)
        expected = _scipy_mwu(x, y)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_many_ties(self):
        """Heavy ties."""
        x = np.array([1, 1, 1, 2, 2, 3], dtype=np.float64)
        y = np.array([1, 2, 2, 2, 3, 3], dtype=np.float64)
        result = mannwhitneyu(x, y)
        expected = _scipy_mwu(x, y)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_large_samples(self):
        """Larger random samples."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(150)
        y = rng.standard_normal(120) + 0.3
        result = mannwhitneyu(x, y)
        expected = _scipy_mwu(x, y)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_large_integer_samples(self):
        """Large samples drawn from integers (many ties)."""
        rng = np.random.default_rng(99)
        x = rng.integers(0, 20, size=100).astype(np.float64)
        y = rng.integers(5, 25, size=80).astype(np.float64)
        result = mannwhitneyu(x, y)
        expected = _scipy_mwu(x, y)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)


# ---------------------------------------------------------------------------
# Alternative hypotheses
# ---------------------------------------------------------------------------

class TestAlternatives:
    def setup_method(self):
        self.x = np.array([19, 22, 16, 29, 24], dtype=np.float64)
        self.y = np.array([20, 11, 17, 12], dtype=np.float64)

    def test_two_sided(self):
        result = mannwhitneyu(self.x, self.y, alternative="two-sided")
        expected = _scipy_mwu(self.x, self.y, alternative="two-sided")
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_less(self):
        result = mannwhitneyu(self.x, self.y, alternative="less")
        expected = _scipy_mwu(self.x, self.y, alternative="less")
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_greater(self):
        result = mannwhitneyu(self.x, self.y, alternative="greater")
        expected = _scipy_mwu(self.x, self.y, alternative="greater")
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_alternatives_large(self):
        """All alternatives with larger data."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(50)
        y = rng.standard_normal(60) + 0.5
        for alt in ("two-sided", "less", "greater"):
            result = mannwhitneyu(x, y, alternative=alt)
            expected = _scipy_mwu(x, y, alternative=alt)
            assert np.isclose(result.statistic, expected.statistic), f"stat mismatch for {alt}"
            assert np.isclose(result.pvalue, expected.pvalue), f"pvalue mismatch for {alt}"


# ---------------------------------------------------------------------------
# Continuity correction
# ---------------------------------------------------------------------------

class TestContinuityCorrection:
    def test_continuity_on(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([3, 4, 5, 6, 7], dtype=np.float64)
        result = mannwhitneyu(x, y, use_continuity=True)
        expected = _scipy_mwu(x, y, use_continuity=True)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_continuity_off(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([3, 4, 5, 6, 7], dtype=np.float64)
        result = mannwhitneyu(x, y, use_continuity=False)
        expected = _scipy_mwu(x, y, use_continuity=False)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_continuity_makes_difference(self):
        """Verify continuity on vs off gives different p-values."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([3, 4, 5, 6, 7], dtype=np.float64)
        r_on = mannwhitneyu(x, y, use_continuity=True)
        r_off = mannwhitneyu(x, y, use_continuity=False)
        assert r_on.statistic == r_off.statistic  # U doesn't change
        assert r_on.pvalue != r_off.pvalue  # p-value does


# ---------------------------------------------------------------------------
# Log-transformation invariance
# ---------------------------------------------------------------------------

class TestLogTransformInvariance:
    """Rankings are invariant under monotone transforms like log.
    Therefore U and p must be identical for data and log(data)."""

    def test_log_invariance_no_ties(self):
        x = np.array([1, 3, 5, 7, 9], dtype=np.float64)
        y = np.array([2, 4, 6, 8], dtype=np.float64)
        r_orig = mannwhitneyu(x, y)
        r_log = mannwhitneyu(np.log(x), np.log(y))
        assert r_orig.statistic == r_log.statistic
        assert r_orig.pvalue == r_log.pvalue

    def test_log_invariance_with_ties(self):
        x = np.array([1, 2, 2, 3, 5], dtype=np.float64)
        y = np.array([2, 3, 3, 4], dtype=np.float64)
        r_orig = mannwhitneyu(x, y)
        r_log = mannwhitneyu(np.log(x), np.log(y))
        assert r_orig.statistic == r_log.statistic
        assert r_orig.pvalue == r_log.pvalue

    def test_log_invariance_large(self):
        """Large integer data: int input vs log(float) input."""
        rng = np.random.default_rng(12)
        x_int = rng.integers(1, 100, size=80)
        y_int = rng.integers(1, 100, size=60)
        x_f = x_int.astype(np.float64)
        y_f = y_int.astype(np.float64)
        r_orig = mannwhitneyu(x_f, y_f)
        r_log = mannwhitneyu(np.log(x_f), np.log(y_f))
        assert r_orig.statistic == r_log.statistic
        assert r_orig.pvalue == r_log.pvalue

    def test_log_invariance_all_alternatives(self):
        x = np.array([1, 3, 5, 7, 9, 11], dtype=np.float64)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float64)
        for alt in ("two-sided", "less", "greater"):
            r_orig = mannwhitneyu(x, y, alternative=alt)
            r_log = mannwhitneyu(np.log(x), np.log(y), alternative=alt)
            assert r_orig.statistic == r_log.statistic, f"stat mismatch for {alt}"
            assert r_orig.pvalue == r_log.pvalue, f"pvalue mismatch for {alt}"


# ---------------------------------------------------------------------------
# Float vs int input consistency
# ---------------------------------------------------------------------------

class TestFloatIntConsistency:
    """Same data as int list and float array should produce identical results."""

    def test_int_list_input(self):
        x_int = [19, 22, 16, 29, 24]
        y_int = [20, 11, 17, 12]
        x_float = np.array(x_int, dtype=np.float64)
        y_float = np.array(y_int, dtype=np.float64)
        r_int = mannwhitneyu(x_int, y_int)
        r_float = mannwhitneyu(x_float, y_float)
        assert r_int.statistic == r_float.statistic
        assert r_int.pvalue == r_float.pvalue

    def test_int32_vs_float64(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        y = np.array([3, 4, 5, 6, 7], dtype=np.int32)
        r_int = mannwhitneyu(x, y)
        r_float = mannwhitneyu(x.astype(np.float64), y.astype(np.float64))
        assert r_int.statistic == r_float.statistic
        assert r_int.pvalue == r_float.pvalue


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_identical(self):
        """All values the same → p = 1.0."""
        x = np.array([5, 5, 5, 5], dtype=np.float64)
        y = np.array([5, 5, 5, 5], dtype=np.float64)
        result = mannwhitneyu(x, y)
        assert result.pvalue == 1.0

    def test_single_element_samples(self):
        x = np.array([1.0])
        y = np.array([2.0])
        result = mannwhitneyu(x, y)
        expected = _scipy_mwu(x, y)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)

    def test_clear_separation(self):
        """Non-overlapping samples should give very small p-value."""
        x = np.array([10, 11, 12, 13, 14, 15], dtype=np.float64)
        y = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        result = mannwhitneyu(x, y, alternative="greater")
        expected = _scipy_mwu(x, y, alternative="greater")
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)
        assert result.pvalue < 0.01

    def test_asymmetric_sizes(self):
        """Very different sample sizes."""
        rng = np.random.default_rng(55)
        x = rng.standard_normal(5)
        y = rng.standard_normal(200)
        result = mannwhitneyu(x, y)
        expected = _scipy_mwu(x, y)
        assert np.isclose(result.statistic, expected.statistic)
        assert np.isclose(result.pvalue, expected.pvalue)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_x(self):
        with pytest.raises(ValueError, match="nonzero size"):
            mannwhitneyu(np.array([]), np.array([1.0]))

    def test_empty_y(self):
        with pytest.raises(ValueError, match="nonzero size"):
            mannwhitneyu(np.array([1.0]), np.array([]))

    def test_nan_in_x(self):
        with pytest.raises(ValueError, match="NaN"):
            mannwhitneyu(np.array([1.0, np.nan]), np.array([1.0]))

    def test_invalid_alternative(self):
        with pytest.raises(ValueError, match="alternative"):
            mannwhitneyu(np.array([1.0]), np.array([1.0]), alternative="bad")


# ---------------------------------------------------------------------------
# Batch functions
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_matches_single(self):
        """mannwhitneyu_batch should match row-by-row single calls."""
        rng = np.random.default_rng(42)
        n_tests = 50
        n1, n2 = 30, 25
        X = rng.standard_normal((n_tests, n1))
        y = rng.standard_normal(n2)

        batch_result = mannwhitneyu_batch(X, y)

        for i in range(n_tests):
            single = mannwhitneyu(X[i], y)
            assert np.isclose(batch_result.statistic[i], single.statistic), f"stat mismatch at {i}"
            assert np.isclose(batch_result.pvalue[i], single.pvalue), f"pvalue mismatch at {i}"

    def test_batch_matches_scipy(self):
        """mannwhitneyu_batch results should match scipy."""
        rng = np.random.default_rng(77)
        n_tests = 20
        X = rng.standard_normal((n_tests, 15))
        y = rng.standard_normal(10) + 0.5

        batch_result = mannwhitneyu_batch(X, y)

        for i in range(n_tests):
            expected = _scipy_mwu(X[i], y)
            assert np.isclose(batch_result.statistic[i], expected.statistic)
            assert np.isclose(batch_result.pvalue[i], expected.pvalue)

    def test_batch_alternatives(self):
        rng = np.random.default_rng(33)
        X = rng.standard_normal((10, 20))
        y = rng.standard_normal(15)
        for alt in ("two-sided", "less", "greater"):
            batch = mannwhitneyu_batch(X, y, alternative=alt)
            for i in range(X.shape[0]):
                single = mannwhitneyu(X[i], y, alternative=alt)
                assert np.isclose(batch.statistic[i], single.statistic)
                assert np.isclose(batch.pvalue[i], single.pvalue)


class TestColumns:
    def test_columns_matches_single(self):
        """mannwhitneyu_columns should match column-by-column single calls."""
        rng = np.random.default_rng(42)
        n1, n2, n_tests = 20, 15, 40
        data = rng.standard_normal((n1 + n2, n_tests))

        col_result = mannwhitneyu_columns(data, n1)

        for i in range(n_tests):
            x = data[:n1, i]
            y = data[n1:, i]
            single = mannwhitneyu(x, y)
            assert np.isclose(col_result.statistic[i], single.statistic), f"stat mismatch at {i}"
            assert np.isclose(col_result.pvalue[i], single.pvalue), f"pvalue mismatch at {i}"

    def test_columns_matches_scipy(self):
        """mannwhitneyu_columns results should match scipy."""
        rng = np.random.default_rng(88)
        n1, n2, n_tests = 12, 10, 15
        data = rng.standard_normal((n1 + n2, n_tests))

        col_result = mannwhitneyu_columns(data, n1)

        for i in range(n_tests):
            x = data[:n1, i]
            y = data[n1:, i]
            expected = _scipy_mwu(x, y)
            assert np.isclose(col_result.statistic[i], expected.statistic)
            assert np.isclose(col_result.pvalue[i], expected.pvalue)

    def test_columns_alternatives(self):
        rng = np.random.default_rng(44)
        n1 = 10
        data = rng.standard_normal((25, 8))
        for alt in ("two-sided", "less", "greater"):
            col = mannwhitneyu_columns(data, n1, alternative=alt)
            for i in range(data.shape[1]):
                single = mannwhitneyu(data[:n1, i], data[n1:, i], alternative=alt)
                assert np.isclose(col.statistic[i], single.statistic)
                assert np.isclose(col.pvalue[i], single.pvalue)
