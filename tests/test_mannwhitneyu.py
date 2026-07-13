"""Tests validating numba_mwu against scipy.stats.mannwhitneyu."""

import numpy as np
import pytest
from scipy import sparse, stats

from numba_mwu import (
    mannwhitneyu,
    mannwhitneyu_columns,
    mannwhitneyu_one_vs_rest,
    mannwhitneyu_one_vs_rest_sparse,
    mannwhitneyu_rows,
    mannwhitneyu_sparse,
    sparse_column_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scipy_mwu(x, y, use_continuity=True, alternative="two-sided"):
    """Wrapper around scipy for the asymptotic method."""
    return stats.mannwhitneyu(
        x,
        y,
        use_continuity=use_continuity,
        alternative=alternative,
        method="asymptotic",
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
            assert np.isclose(result.statistic, expected.statistic), (
                f"stat mismatch for {alt}"
            )
            assert np.isclose(result.pvalue, expected.pvalue), (
                f"pvalue mismatch for {alt}"
            )


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
# Row and column batch functions
# ---------------------------------------------------------------------------


class TestRows:
    def test_rows_matches_single(self):
        """mannwhitneyu_rows should match row-by-row single calls."""
        rng = np.random.default_rng(42)
        n_tests = 50
        n1, n2 = 30, 25
        X = rng.standard_normal((n_tests, n1))
        y = rng.standard_normal(n2)

        row_result = mannwhitneyu_rows(X, y)

        for i in range(n_tests):
            single = mannwhitneyu(X[i], y)
            assert np.isclose(row_result.statistic[i], single.statistic), (
                f"stat mismatch at {i}"
            )
            assert np.isclose(row_result.pvalue[i], single.pvalue), (
                f"pvalue mismatch at {i}"
            )

    def test_rows_matches_scipy(self):
        """mannwhitneyu_rows results should match scipy."""
        rng = np.random.default_rng(77)
        n_tests = 20
        X = rng.standard_normal((n_tests, 15))
        y = rng.standard_normal(10) + 0.5

        row_result = mannwhitneyu_rows(X, y)

        for i in range(n_tests):
            expected = _scipy_mwu(X[i], y)
            assert np.isclose(row_result.statistic[i], expected.statistic)
            assert np.isclose(row_result.pvalue[i], expected.pvalue)

    def test_rows_alternatives(self):
        rng = np.random.default_rng(33)
        X = rng.standard_normal((10, 20))
        y = rng.standard_normal(15)
        for alt in ("two-sided", "less", "greater"):
            rows = mannwhitneyu_rows(X, y, alternative=alt)
            for i in range(X.shape[0]):
                single = mannwhitneyu(X[i], y, alternative=alt)
                assert np.isclose(rows.statistic[i], single.statistic)
                assert np.isclose(rows.pvalue[i], single.pvalue)


class TestColumns:
    def test_columns_matches_single(self):
        """mannwhitneyu_columns should match column-by-column single calls."""
        rng = np.random.default_rng(42)
        n1, n2, n_genes = 20, 15, 40
        X = rng.standard_normal((n1, n_genes))
        Y = rng.standard_normal((n2, n_genes))

        col_result = mannwhitneyu_columns(X, Y)

        for i in range(n_genes):
            single = mannwhitneyu(X[:, i], Y[:, i])
            assert np.isclose(col_result.statistic[i], single.statistic), (
                f"stat mismatch at {i}"
            )
            assert np.isclose(col_result.pvalue[i], single.pvalue), (
                f"pvalue mismatch at {i}"
            )

    def test_columns_matches_scipy(self):
        """mannwhitneyu_columns results should match scipy."""
        rng = np.random.default_rng(88)
        n1, n2, n_genes = 12, 10, 15
        X = rng.standard_normal((n1, n_genes))
        Y = rng.standard_normal((n2, n_genes))

        col_result = mannwhitneyu_columns(X, Y)

        for i in range(n_genes):
            expected = _scipy_mwu(X[:, i], Y[:, i])
            assert np.isclose(col_result.statistic[i], expected.statistic)
            assert np.isclose(col_result.pvalue[i], expected.pvalue)

    def test_columns_alternatives(self):
        rng = np.random.default_rng(44)
        n1, n2 = 10, 15
        X = rng.standard_normal((n1, 8))
        Y = rng.standard_normal((n2, 8))
        for alt in ("two-sided", "less", "greater"):
            col = mannwhitneyu_columns(X, Y, alternative=alt)
            for i in range(X.shape[1]):
                single = mannwhitneyu(X[:, i], Y[:, i], alternative=alt)
                assert np.isclose(col.statistic[i], single.statistic)
                assert np.isclose(col.pvalue[i], single.pvalue)

    def test_columns_as_views(self):
        """Slicing a matrix into two views should work (the primary use case)."""
        rng = np.random.default_rng(55)
        n1, n2, n_genes = 20, 30, 10
        full = rng.standard_normal((n1 + n2, n_genes))
        X = full[:n1]  # view, not copy
        Y = full[n1:]  # view, not copy

        col_result = mannwhitneyu_columns(X, Y)

        for i in range(n_genes):
            expected = _scipy_mwu(X[:, i], Y[:, i])
            assert np.isclose(col_result.statistic[i], expected.statistic)
            assert np.isclose(col_result.pvalue[i], expected.pvalue)


# ---------------------------------------------------------------------------
# Sparse matrix support
# ---------------------------------------------------------------------------


class TestSparse:
    """Validate mannwhitneyu_sparse against dense path and scipy."""

    def test_sparse_matches_dense(self):
        """Small dense matrix → two CSR matrices, compare against dense."""
        rng = np.random.default_rng(42)
        n_a, n_b, n_genes = 10, 8, 5
        dense = rng.integers(0, 10, size=(n_a + n_b, n_genes)).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()
        sp_result = mannwhitneyu_sparse(X_sp, Y_sp)

        for i in range(n_genes):
            expected = mannwhitneyu(dense[:n_a, i], dense[n_a:, i])
            assert np.isclose(sp_result.statistic[i], expected.statistic), (
                f"stat mismatch col {i}"
            )
            assert np.isclose(sp_result.pvalue[i], expected.pvalue), (
                f"pvalue mismatch col {i}"
            )

    def test_sparse_matches_scipy(self):
        """Compare each gene against scipy.stats.mannwhitneyu."""
        rng = np.random.default_rng(77)
        n_a, n_b, n_genes = 15, 12, 8
        dense = rng.integers(0, 8, size=(n_a + n_b, n_genes)).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()
        sp_result = mannwhitneyu_sparse(X_sp, Y_sp)

        for i in range(n_genes):
            expected = _scipy_mwu(dense[:n_a, i], dense[n_a:, i])
            assert np.isclose(sp_result.statistic[i], expected.statistic), (
                f"stat mismatch col {i}"
            )
            assert np.isclose(sp_result.pvalue[i], expected.pvalue), (
                f"pvalue mismatch col {i}"
            )

    def test_sparse_highly_sparse(self):
        """~95% zeros, realistic for scRNA-seq."""
        rng = np.random.default_rng(99)
        n_a, n_b, n_genes = 80, 120, 50
        dense = np.zeros((n_a + n_b, n_genes), dtype=np.float64)
        mask = rng.random((n_a + n_b, n_genes)) < 0.05
        dense[mask] = rng.integers(1, 100, size=mask.sum()).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()
        sp_result = mannwhitneyu_sparse(X_sp, Y_sp)

        for i in range(n_genes):
            expected = _scipy_mwu(dense[:n_a, i], dense[n_a:, i])
            assert np.isclose(sp_result.statistic[i], expected.statistic), (
                f"stat mismatch col {i}"
            )
            assert np.isclose(sp_result.pvalue[i], expected.pvalue), (
                f"pvalue mismatch col {i}"
            )

    def test_sparse_all_zeros_column(self):
        """A column that is entirely zero → p = 1.0."""
        n_a, n_b, n_genes = 10, 10, 3
        dense = np.zeros((n_a + n_b, n_genes), dtype=np.float64)
        # Only put data in columns 0 and 2, leave column 1 all zeros
        dense[:5, 0] = [1, 2, 3, 4, 5]
        dense[n_a : n_a + 5, 2] = [6, 7, 8, 9, 10]

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()
        result = mannwhitneyu_sparse(X_sp, Y_sp)

        assert result.pvalue[1] == 1.0

    def test_sparse_no_zeros_column(self):
        """A column with no zeros (fully dense in sparse matrix)."""
        rng = np.random.default_rng(55)
        n_a, n_b, n_genes = 10, 10, 3
        dense = rng.integers(1, 50, size=(n_a + n_b, n_genes)).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()
        sp_result = mannwhitneyu_sparse(X_sp, Y_sp)

        for i in range(n_genes):
            expected = mannwhitneyu(dense[:n_a, i], dense[n_a:, i])
            assert np.isclose(sp_result.statistic[i], expected.statistic)
            assert np.isclose(sp_result.pvalue[i], expected.pvalue)

    def test_sparse_alternatives(self):
        """All three alternative hypotheses."""
        rng = np.random.default_rng(33)
        n_a, n_b, n_genes = 15, 12, 6
        dense = rng.integers(0, 10, size=(n_a + n_b, n_genes)).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()

        for alt in ("two-sided", "less", "greater"):
            sp_result = mannwhitneyu_sparse(X_sp, Y_sp, alternative=alt)
            for i in range(n_genes):
                expected = _scipy_mwu(dense[:n_a, i], dense[n_a:, i], alternative=alt)
                assert np.isclose(sp_result.statistic[i], expected.statistic), (
                    f"stat mismatch col {i}, alt={alt}"
                )
                assert np.isclose(sp_result.pvalue[i], expected.pvalue), (
                    f"pvalue mismatch col {i}, alt={alt}"
                )

    def test_sparse_continuity(self):
        """With and without continuity correction."""
        rng = np.random.default_rng(44)
        n_a, n_b, n_genes = 12, 10, 4
        dense = rng.integers(0, 8, size=(n_a + n_b, n_genes)).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()

        for cont in (True, False):
            sp_result = mannwhitneyu_sparse(X_sp, Y_sp, use_continuity=cont)
            for i in range(n_genes):
                expected = _scipy_mwu(
                    dense[:n_a, i], dense[n_a:, i], use_continuity=cont
                )
                assert np.isclose(sp_result.statistic[i], expected.statistic)
                assert np.isclose(sp_result.pvalue[i], expected.pvalue)

    def test_sparse_explicit_zeros_handled(self):
        """Explicit stored zeros should be handled by eliminate_zeros()."""
        n_a = 5
        dense = np.array(
            [
                [0, 1],
                [2, 0],
                [0, 3],
                [4, 0],
                [0, 5],
                [6, 0],
                [0, 7],
                [8, 0],
                [0, 9],
                [10, 0],
            ],
            dtype=np.float64,
        )

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()
        sp_result = mannwhitneyu_sparse(X_sp, Y_sp)

        for i in range(dense.shape[1]):
            expected = mannwhitneyu(dense[:n_a, i], dense[n_a:, i])
            assert np.isclose(sp_result.statistic[i], expected.statistic)
            assert np.isclose(sp_result.pvalue[i], expected.pvalue)

    def test_sparse_negative_values_rejected(self):
        """Negative stored values should raise ValueError."""
        X_sp = sparse.csr_matrix(np.array([[1, 2]], dtype=np.float64))
        Y_sp = sparse.csr_matrix(np.array([[2, -1]], dtype=np.float64))
        with pytest.raises(ValueError, match="non-negative"):
            mannwhitneyu_sparse(X_sp, Y_sp)

    def test_sparse_non_csr_rejected(self):
        """Non-CSR format should raise TypeError."""
        X_sp = sparse.csr_matrix(np.array([[1, 2]], dtype=np.float64))
        Y_csc = sparse.csc_matrix(np.array([[3, 4]], dtype=np.float64))
        with pytest.raises(TypeError, match="CSR"):
            mannwhitneyu_sparse(X_sp, Y_csc)

    def test_sparse_large(self):
        """Larger matrix (5000 cells, 200 genes, ~90% sparse)."""
        rng = np.random.default_rng(123)
        n_a, n_b, n_genes = 2000, 3000, 200

        dense = np.zeros((n_a + n_b, n_genes), dtype=np.float64)
        mask = rng.random((n_a + n_b, n_genes)) < 0.10
        dense[mask] = rng.integers(1, 500, size=mask.sum()).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()
        sp_result = mannwhitneyu_sparse(X_sp, Y_sp)

        # Spot-check 20 random columns against scipy
        check_cols = rng.choice(n_genes, size=20, replace=False)
        for i in check_cols:
            expected = _scipy_mwu(dense[:n_a, i], dense[n_a:, i])
            assert np.isclose(sp_result.statistic[i], expected.statistic), (
                f"stat mismatch col {i}"
            )
            assert np.isclose(sp_result.pvalue[i], expected.pvalue), (
                f"pvalue mismatch col {i}"
            )

    def test_sparse_shuffled_groups(self):
        """Slicing by shuffled group labels (the primary use case)."""
        rng = np.random.default_rng(66)
        n_cells, n_genes = 40, 5
        dense = rng.integers(0, 10, size=(n_cells, n_genes)).astype(np.float64)

        group_a = np.zeros(n_cells, dtype=bool)
        group_a[rng.choice(n_cells, size=20, replace=False)] = True

        # This is the realistic workflow: slice by labels into two CSR matrices
        X_sp = sparse.csr_matrix(dense[group_a])
        Y_sp = sparse.csr_matrix(dense[~group_a])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()
        sp_result = mannwhitneyu_sparse(X_sp, Y_sp)

        for i in range(n_genes):
            expected = _scipy_mwu(dense[group_a, i], dense[~group_a, i])
            assert np.isclose(sp_result.statistic[i], expected.statistic), f"col {i}"
            assert np.isclose(sp_result.pvalue[i], expected.pvalue), f"col {i}"


# ---------------------------------------------------------------------------
# Precomputed sparse column index
# ---------------------------------------------------------------------------


class TestSparseColumnIndex:
    """Validate that precomputed SparseColumnIndex produces identical results."""

    def test_precomputed_matches_raw(self):
        """Precomputed index should give identical results to raw CSR."""
        rng = np.random.default_rng(42)
        n_a, n_b, n_genes = 15, 12, 8
        dense = rng.integers(0, 10, size=(n_a + n_b, n_genes)).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()

        raw = mannwhitneyu_sparse(X_sp, Y_sp)

        X_idx = sparse_column_index(X_sp)
        Y_idx = sparse_column_index(Y_sp)
        pre = mannwhitneyu_sparse(X_idx, Y_idx)

        np.testing.assert_array_equal(raw.statistic, pre.statistic)
        np.testing.assert_array_equal(raw.pvalue, pre.pvalue)

    def test_precomputed_reuse_reference(self):
        """Reusing a precomputed reference across multiple comparisons."""
        rng = np.random.default_rng(77)
        n_ref, n_genes = 50, 10
        ref_dense = rng.integers(0, 8, size=(n_ref, n_genes)).astype(np.float64)
        ref_sp = sparse.csr_matrix(ref_dense)
        ref_sp.eliminate_zeros()
        ref_idx = sparse_column_index(ref_sp)

        for _ in range(5):
            n_test = rng.integers(10, 30)
            test_dense = rng.integers(0, 8, size=(n_test, n_genes)).astype(np.float64)
            test_sp = sparse.csr_matrix(test_dense)
            test_sp.eliminate_zeros()

            raw = mannwhitneyu_sparse(test_sp, ref_sp)
            pre = mannwhitneyu_sparse(test_sp, ref_idx)

            np.testing.assert_array_equal(raw.statistic, pre.statistic)
            np.testing.assert_array_equal(raw.pvalue, pre.pvalue)

    def test_mixed_precomputed_and_raw(self):
        """One argument precomputed, the other raw CSR."""
        rng = np.random.default_rng(33)
        n_a, n_b, n_genes = 20, 15, 6
        dense = rng.integers(0, 10, size=(n_a + n_b, n_genes)).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()

        raw = mannwhitneyu_sparse(X_sp, Y_sp)

        # Precomputed X, raw Y
        X_idx = sparse_column_index(X_sp)
        mixed1 = mannwhitneyu_sparse(X_idx, Y_sp)
        np.testing.assert_array_equal(raw.statistic, mixed1.statistic)
        np.testing.assert_array_equal(raw.pvalue, mixed1.pvalue)

        # Raw X, precomputed Y
        Y_idx = sparse_column_index(Y_sp)
        mixed2 = mannwhitneyu_sparse(X_sp, Y_idx)
        np.testing.assert_array_equal(raw.statistic, mixed2.statistic)
        np.testing.assert_array_equal(raw.pvalue, mixed2.pvalue)

    def test_precomputed_alternatives(self):
        """All alternatives work with precomputed index."""
        rng = np.random.default_rng(44)
        n_a, n_b, n_genes = 12, 10, 5
        dense = rng.integers(0, 8, size=(n_a + n_b, n_genes)).astype(np.float64)

        X_sp = sparse.csr_matrix(dense[:n_a])
        Y_sp = sparse.csr_matrix(dense[n_a:])
        X_sp.eliminate_zeros()
        Y_sp.eliminate_zeros()

        Y_idx = sparse_column_index(Y_sp)

        for alt in ("two-sided", "less", "greater"):
            raw = mannwhitneyu_sparse(X_sp, Y_sp, alternative=alt)
            pre = mannwhitneyu_sparse(X_sp, Y_idx, alternative=alt)
            np.testing.assert_array_equal(raw.statistic, pre.statistic)
            np.testing.assert_array_equal(raw.pvalue, pre.pvalue)


# ---------------------------------------------------------------------------
# One-vs-rest (N groups tested simultaneously against "everything else")
# ---------------------------------------------------------------------------


def _assert_one_vs_rest_matches_scipy(
    result, dense, labels, use_continuity=True, alternative="two-sided", skip_cols=()
):
    """Cross-check every (group, column) pair against an independently computed
    scipy.stats.mannwhitneyu — validates the multi-group rank-sum reduction
    from first principles, not just by diffing against another numba_mwu path."""
    n_groups = int(labels.max()) + 1
    for g in range(n_groups):
        mask = labels == g
        for j in range(dense.shape[1]):
            if j in skip_cols:
                continue
            expected = _scipy_mwu(
                dense[mask, j],
                dense[~mask, j],
                use_continuity=use_continuity,
                alternative=alternative,
            )
            assert np.isclose(result.statistic[g, j], expected.statistic), (
                f"stat mismatch group={g} col={j}"
            )
            assert np.isclose(result.pvalue[g, j], expected.pvalue), (
                f"pvalue mismatch group={g} col={j}"
            )


class TestOneVsRest:
    """Validate mannwhitneyu_one_vs_rest (dense) against scipy."""

    def test_matches_scipy_multi_group(self):
        rng = np.random.default_rng(2)
        n_rows, n_cols = 30, 10
        X = rng.standard_normal((n_rows, n_cols))
        labels = np.array([0] * 8 + [1] * 12 + [2] * 10)
        # Shift group 1 so there's a real effect to detect
        X[8:20] += 1.5

        result = mannwhitneyu_one_vs_rest(X, labels)
        assert result.statistic.shape == (3, n_cols)
        assert result.pvalue.shape == (3, n_cols)
        _assert_one_vs_rest_matches_scipy(result, X, labels)

    def test_matches_sequential_columns_calls(self):
        """Result must equal calling mannwhitneyu_columns(group, rest) per group."""
        rng = np.random.default_rng(3)
        n_rows, n_cols = 25, 6
        X = rng.standard_normal((n_rows, n_cols))
        labels = np.array([0] * 5 + [1] * 9 + [2] * 4 + [3] * 7)

        result = mannwhitneyu_one_vs_rest(X, labels)
        for g in range(4):
            mask = labels == g
            expected = mannwhitneyu_columns(X[mask], X[~mask])
            np.testing.assert_allclose(result.statistic[g], expected.statistic)
            np.testing.assert_allclose(result.pvalue[g], expected.pvalue)

    def test_group_of_size_one(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((10, 3))
        labels = np.array([0] + [1] * 9)
        result = mannwhitneyu_one_vs_rest(X, labels)
        _assert_one_vs_rest_matches_scipy(result, X, labels)

    def test_ties_spanning_groups(self):
        """Ties across group boundaries share one tie-correction term."""
        X = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [1.0]])
        labels = np.array([0, 0, 1, 1, 2, 2])
        result = mannwhitneyu_one_vs_rest(X, labels)
        _assert_one_vs_rest_matches_scipy(result, X, labels)

    def test_all_identical_column_gives_pvalue_one(self):
        """A fully-tied column needs no special-casing: s_sq <= 0 -> p = 1.0."""
        X = np.full((9, 1), 5.0)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = mannwhitneyu_one_vs_rest(X, labels)
        np.testing.assert_allclose(result.pvalue, 1.0)
        assert np.isfinite(result.statistic).all()

    def test_alternatives(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 4))
        labels = np.array([0] * 6 + [1] * 8 + [2] * 6)
        for alt in ("two-sided", "less", "greater"):
            result = mannwhitneyu_one_vs_rest(X, labels, alternative=alt)
            _assert_one_vs_rest_matches_scipy(result, X, labels, alternative=alt)

    def test_continuity(self):
        rng = np.random.default_rng(6)
        X = rng.integers(0, 8, size=(18, 3)).astype(np.float64)
        labels = np.array([0] * 6 + [1] * 6 + [2] * 6)
        for cont in (True, False):
            result = mannwhitneyu_one_vs_rest(X, labels, use_continuity=cont)
            _assert_one_vs_rest_matches_scipy(result, X, labels, use_continuity=cont)

    def test_explicit_n_groups(self):
        """n_groups can exceed the largest observed label (all-present here)."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((12, 2))
        labels = np.array([0] * 4 + [1] * 4 + [2] * 4)
        result = mannwhitneyu_one_vs_rest(X, labels, n_groups=3)
        _assert_one_vs_rest_matches_scipy(result, X, labels)


class TestOneVsRestSparse:
    """Validate mannwhitneyu_one_vs_rest_sparse against dense and scipy."""

    def test_matches_dense(self):
        rng = np.random.default_rng(8)
        n_rows, n_cols = 40, 12
        dense = rng.integers(0, 6, size=(n_rows, n_cols)).astype(np.float64)
        dense[dense < 2] = 0.0  # induce sparsity
        labels = np.array([0] * 10 + [1] * 15 + [2] * 15)

        X_sp = sparse.csr_matrix(dense)
        X_sp.eliminate_zeros()

        dense_result = mannwhitneyu_one_vs_rest(dense, labels)
        sparse_result = mannwhitneyu_one_vs_rest_sparse(X_sp, labels)
        np.testing.assert_allclose(
            dense_result.statistic, sparse_result.statistic, rtol=1e-8
        )
        np.testing.assert_allclose(
            dense_result.pvalue, sparse_result.pvalue, rtol=1e-8, atol=1e-10
        )

    def test_matches_scipy(self):
        rng = np.random.default_rng(9)
        n_rows, n_cols = 35, 8
        dense = rng.integers(0, 5, size=(n_rows, n_cols)).astype(np.float64)
        dense[dense < 1.5] = 0.0
        labels = np.array([0] * 8 + [1] * 12 + [2] * 15)

        X_sp = sparse.csr_matrix(dense)
        X_sp.eliminate_zeros()
        result = mannwhitneyu_one_vs_rest_sparse(X_sp, labels)
        _assert_one_vs_rest_matches_scipy(result, dense, labels)

    def test_all_zero_column_gives_pvalue_one(self):
        n_rows, n_cols = 12, 3
        dense = np.zeros((n_rows, n_cols), dtype=np.float64)
        dense[:6, 0] = [1, 2, 3, 4, 5, 6]
        dense[6:, 2] = [7, 8, 9, 10, 11, 12]
        labels = np.array([0] * 4 + [1] * 4 + [2] * 4)

        X_sp = sparse.csr_matrix(dense)
        X_sp.eliminate_zeros()
        result = mannwhitneyu_one_vs_rest_sparse(X_sp, labels)
        np.testing.assert_allclose(result.pvalue[:, 1], 1.0)
        assert np.isfinite(result.statistic[:, 1]).all()

    def test_no_zeros_column(self):
        """A column with no zeros at all (fully dense within a sparse matrix)."""
        rng = np.random.default_rng(10)
        dense = rng.integers(1, 50, size=(15, 3)).astype(np.float64)
        labels = np.array([0] * 5 + [1] * 5 + [2] * 5)

        X_sp = sparse.csr_matrix(dense)
        X_sp.eliminate_zeros()
        result = mannwhitneyu_one_vs_rest_sparse(X_sp, labels)
        _assert_one_vs_rest_matches_scipy(result, dense, labels)

    def test_alternatives(self):
        rng = np.random.default_rng(11)
        dense = rng.integers(0, 8, size=(24, 4)).astype(np.float64)
        dense[dense < 2] = 0.0
        labels = np.array([0] * 8 + [1] * 8 + [2] * 8)

        X_sp = sparse.csr_matrix(dense)
        X_sp.eliminate_zeros()
        for alt in ("two-sided", "less", "greater"):
            result = mannwhitneyu_one_vs_rest_sparse(X_sp, labels, alternative=alt)
            _assert_one_vs_rest_matches_scipy(result, dense, labels, alternative=alt)

    def test_negative_values_rejected(self):
        X_sp = sparse.csr_matrix(np.array([[1.0, -2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="non-negative"):
            mannwhitneyu_one_vs_rest_sparse(X_sp, np.array([0, 1]))

    def test_non_csr_rejected(self):
        X_csc = sparse.csc_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(TypeError, match="CSR"):
            mannwhitneyu_one_vs_rest_sparse(X_csc, np.array([0, 1]))


# ---------------------------------------------------------------------------
# One-vs-rest label validation
# ---------------------------------------------------------------------------


class TestOneVsRestValidation:
    def test_single_group_raises(self):
        X = np.zeros((6, 2))
        labels = np.zeros(6, dtype=int)
        with pytest.raises(ValueError, match="at least 2 groups"):
            mannwhitneyu_one_vs_rest(X, labels)

    def test_empty_group_raises(self):
        """n_groups larger than the observed labels leaves a group with no rows."""
        X = np.zeros((6, 2))
        labels = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="no rows"):
            mannwhitneyu_one_vs_rest(X, labels, n_groups=3)

    def test_negative_label_raises(self):
        X = np.zeros((4, 2))
        labels = np.array([0, 1, -1, 1])
        with pytest.raises(ValueError, match="negative"):
            mannwhitneyu_one_vs_rest(X, labels)

    def test_wrong_length_raises(self):
        X = np.zeros((5, 2))
        labels = np.array([0, 1, 1])
        with pytest.raises(ValueError, match="length"):
            mannwhitneyu_one_vs_rest(X, labels)

    def test_2d_labels_raises(self):
        X = np.zeros((4, 2))
        labels = np.zeros((4, 1))
        with pytest.raises(ValueError, match="1-dimensional"):
            mannwhitneyu_one_vs_rest(X, labels)

    def test_n_groups_smaller_than_max_label_raises(self):
        X = np.zeros((4, 2))
        labels = np.array([0, 1, 2, 0])
        with pytest.raises(ValueError, match="smaller than the largest label"):
            mannwhitneyu_one_vs_rest(X, labels, n_groups=2)
