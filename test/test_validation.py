"""Kiểm thử module validation và hàm chỉ số thị trường của download_market (không gọi mạng)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import causal_tail as CT
import download_market as DM
import validation as V


# ----------------------------------------------------------------------------- nhân tố thị trường
def test_index_log_returns():
    df = pd.DataFrame({"time": ["2020-01-02", "2020-01-01", "2020-01-03"], "close": [110.0, 100.0, 121.0]})
    r = DM.index_log_returns(df)
    assert list(r.index) == list(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    assert np.allclose(r.values, np.log(1.1)) and r.name == "VNI"
    with pytest.raises(ValueError):
        DM.index_log_returns(pd.DataFrame({"time": ["2020-01-01"], "close": [0.0]}))


def test_loo_market_excludes_own_column():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 9.0]})
    M = V.loo_market(X)
    assert M.loc[0, "a"] == pytest.approx(4.0) and M.loc[1, "c"] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        V.loo_market(X[["a"]])


def test_loo_market_adjust_removes_common_factor():
    rng = np.random.default_rng(10)
    f = rng.standard_normal(4000)
    X = pd.DataFrame({c: f + 0.5 * rng.standard_normal(4000) for c in "abcdefgh"})
    E, beta = V.loo_market_adjust(X)
    assert (beta > 0.8).all()
    assert np.abs(np.corrcoef(E.values, rowvar=False)[np.triu_indices(8, 1)]).max() < 0.2


def test_swap_test_pairs_matches_single_and_is_parallel_safe():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.standard_t(3, (800, 3)), columns=["VHM", "NVL", "Á"])
    out = V.swap_test_pairs(X, [("VHM", "NVL"), ("NVL", "Á")], k=15, B=49, seed=3, n_jobs=2)
    d, null = V.swap_null_asymmetry(X["NVL"].values, X["Á"].values, k=15, B=49, seed=13)
    assert out.loc[1, "D"] == pytest.approx(d) and out.loc[1, "p"] == pytest.approx(V.perm_pvalue(d, null))
    assert set(out.columns) >= {"cap", "D", "p", "p_holm", "chieu"}


def test_market_adjust_removes_factor():
    rng = np.random.default_rng(0)
    f = pd.Series(rng.standard_normal(3000))
    X = pd.DataFrame({"a": 0.8 * f + rng.standard_normal(3000), "b": -0.3 * f + rng.standard_normal(3000)})
    E, beta = V.market_adjust(X, f)
    assert beta["a"] == pytest.approx(0.8, abs=0.05) and beta["b"] == pytest.approx(-0.3, abs=0.05)
    assert abs(np.corrcoef(E["a"], f)[0, 1]) < 1e-10


# ----------------------------------------------------------------------------- hoán vị
def test_independence_null_centred_at_one_half():
    rng = np.random.default_rng(1)
    f = rng.standard_t(3, 1500)
    X = np.column_stack([f + rng.standard_t(3, 1500) for _ in range(3)])
    null = V.permutation_null_independence(X, k=20, B=60, seed=0)
    off = ~np.eye(3, dtype=bool)
    assert abs(np.nanmean(null[:, off]) - 0.5) < 0.03
    obs = CT.tail_coef_matrix(X, k=20)
    assert np.all(obs[off] > np.quantile(null[:, off], 0.99))           # phụ thuộc thật vượt xa H0


def test_swap_test_detects_direction_and_not_symmetric_pair():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.standard_t(1.5, n)
    y = x + rng.standard_t(1.5, n)
    d, null = V.swap_null_asymmetry(x, y, k=CT.default_k(n), B=199, seed=0)
    assert d > 0 and V.perm_pvalue(d, null) < 0.05
    f = rng.standard_t(2, n)
    a, b = f + rng.standard_t(2, n), f + rng.standard_t(2, n)          # đối xứng
    d0, null0 = V.swap_null_asymmetry(a, b, k=CT.default_k(n), B=199, seed=1)
    assert V.perm_pvalue(d0, null0) > 0.01


@pytest.mark.parametrize("alt, expected", [("greater", 3 / 6), ("less", 5 / 6), ("two-sided", 5 / 6)])
def test_perm_pvalue_alternatives(alt, expected):
    null = np.array([-2.0, -1.0, 0.5, 1.0, 3.0])
    assert V.perm_pvalue(1.0, null, alt) == pytest.approx(expected)


def test_perm_pvalue_rejects_bad_alternative():
    with pytest.raises(ValueError):
        V.perm_pvalue(0.0, np.zeros(3), "khac")


def test_holm_matches_hand_computation():
    adj = V.holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])


# ----------------------------------------------------------------------------- thời gian
def test_rolling_and_period_ease_shapes():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=600)
    X = pd.DataFrame(rng.standard_t(3, (600, 3)), index=idx, columns=["VHM", "NVL", "Á"])
    roll = V.rolling_ease(X, window=250, step=50)
    assert len(roll) == 8 and set(roll.filter(like="pos_").iloc[0]) == {1, 2, 3}
    per = V.period_ease(X, {"đầu": ("2020-01-01", "2020-12-31"), "sau": ("2021-01-01", "2022-12-31")})
    assert list(per.index) == ["đầu", "sau"] and per["n"].sum() == 600
    with pytest.raises(ValueError):
        V.rolling_ease(X, window=601, step=10)


# ----------------------------------------------------------------------------- mô phỏng hiệu chỉnh
def test_fit_one_factor_recovers_loadings():
    rng = np.random.default_rng(4)
    lam = np.array([0.3, 0.5, 0.7, 0.8])
    X = pd.DataFrame(V.simulate_factor_scm(20_000, lam, df=6, rng=rng), columns=list("abcd"))
    est = V.fit_one_factor(X)
    assert np.allclose(est.values, lam, atol=0.05)


def test_simulate_factor_scm_with_edge_and_truth():
    rng = np.random.default_rng(5)
    Bc = np.zeros((3, 3)); Bc[0, 2] = 0.5
    X = V.simulate_factor_scm(5000, [0.5, 0.5, 0.5], df=5, B_coef=Bc, rng=rng)
    assert X.shape == (5000, 3)
    c = np.corrcoef(X, rowvar=False)
    assert c[0, 2] > c[0, 1] + 0.1                                       # cạnh 0 → 2 tăng tương quan
    A, obs = V.factor_scm_truth(Bc, 3)
    assert A[0, 1:].all() and A[1, 3] == 1 and obs == [1, 2, 3]
    with pytest.raises(ValueError):
        V.simulate_factor_scm(10, [0.5], df=2)


def test_order_frequency_rows_sum_to_one():
    f = V.order_frequency([["a", "b"], ["b", "a"], ["a", "b"]], ["a", "b"])
    assert f.loc["a", 1] == pytest.approx(2 / 3) and np.allclose(f.sum(axis=1), 1)


def test_loo_market_rejects_nan():
    X = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    with pytest.raises(ValueError, match="NaN"):
        V.loo_market(X)



def test_maxT_controls_fwer_and_detects_signal():
    rng = np.random.default_rng(20)
    B, p = 200, 4
    null = rng.uniform(0.45, 0.55, (B, p, p))
    obs = np.full((p, p), 0.5); obs[0, 1] = 0.9; np.fill_diagonal(obs, np.nan)
    pv = V.maxT_pvalues(obs, null)
    assert pv[0, 1] == pytest.approx(1 / (B + 1))          # vượt mọi max dưới H0
    assert np.nanmin(np.delete(pv.ravel(), [1, 0])) > 0.05 or pv[1, 0] > 0.05
    assert np.isnan(np.diag(pv)).all()
