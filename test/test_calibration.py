"""Kiểm thử module calibration: số nhân tố, mô hình nhân tố, hệ số đuôi có điều kiện, Granger trong đuôi, siêu tham số."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import calibration as C
import causal_tail as CT


def _factor_data(n=3000, K=2, p=12, seed=0):
    rng = np.random.default_rng(seed)
    L = np.zeros((p, K))
    L[:, 0] = rng.uniform(0.5, 0.7, p)
    if K > 1:                                                        # nhân tố 2: +0,5 nửa đầu, −0,5 nửa sau
        L[:, 1] = np.where(np.arange(p) < p // 2, 0.5, -0.5)
    return pd.DataFrame(C.simulate_multifactor_scm(n, L, df=5, rng=rng), columns=[f"S{i}" for i in range(p)]), L


# ----------------------------------------------------------------------------- số nhân tố
@pytest.mark.parametrize("K", [1, 2])
def test_ahn_horenstein_recovers_number_of_factors(K):
    X, _ = _factor_data(K=K, seed=K)
    res = C.n_factors_ahn_horenstein(X, kmax=5)
    assert res["K_ER"] == K and res["K_GR"] == K
    assert len(res["ER"]) == 6 and np.all(np.diff(res["eigen"]) <= 1e-12)


def test_ahn_horenstein_no_factor_and_bad_kmax():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((2000, 10))
    assert C.n_factors_ahn_horenstein(X, kmax=4)["K_ER"] == 0         # μ_0 "giả" cho phép K = 0
    with pytest.raises(ValueError):
        C.n_factors_ahn_horenstein(X, kmax=9)


# ----------------------------------------------------------------------------- mô hình nhân tố
def test_factor_model_loadings_and_residual_decorrelation():
    X, L = _factor_data(K=1, p=10, seed=4)
    fm = C.factor_model(X, K=1)
    assert np.corrcoef(fm["loadings"]["F1"], L[:, 0])[0, 1] > 0.9
    off = np.corrcoef(fm["resid"].values, rowvar=False)[np.triu_indices(10, 1)]
    raw = np.corrcoef(X.values, rowvar=False)[np.triu_indices(10, 1)]
    # PCA với p nhỏ để lại tương quan âm nhỏ (≈ −1/(p−1)) — vẫn giảm mạnh so với dữ liệu gốc
    assert np.abs(off).mean() < np.abs(raw).mean() / 3 and off.mean() < 0
    assert C.factor_model(X, K=0)["resid"].equals(X)
    with pytest.raises(ValueError):
        C.factor_model(X, K=10)


def test_kendall_correlation_matches_gaussian_correlation():
    rng = np.random.default_rng(5)
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 20_000)
    assert C.kendall_correlation(z)[0, 1] == pytest.approx(0.6, abs=0.02)


def test_simulate_multifactor_validation():
    with pytest.raises(ValueError):
        C.simulate_multifactor_scm(10, np.array([[0.8, 0.7]]), df=5)     # 1 mã × 2 nhân tố, ‖λ‖² ≥ 1
    with pytest.raises(ValueError):
        C.simulate_multifactor_scm(10, np.zeros((2, 2, 2)), df=5)
    with pytest.raises(ValueError):
        C.simulate_multifactor_scm(10, np.array([[0.5]]), df=2)


# ----------------------------------------------------------------------------- hệ số có điều kiện
def test_adjusted_coef_removes_pure_factor_dependence():
    """Mô hình một nhân tố thuần: hệ số gốc ≫ 1/2 nhưng hệ số có điều kiện thị trường về gần 1/2."""
    X, _ = _factor_data(n=4000, K=1, p=12, seed=6)
    raw = CT.tail_coef_matrix(X, kind="psi")
    adj = C.adjusted_tail_coef_matrix(X, kind="psi")["adj"]
    off = ~np.eye(12, dtype=bool)
    assert np.nanmean(raw.values[off]) > 0.6
    assert abs(np.nanmean(adj.values[off]) - 0.5) < 0.05


def test_adjusted_coef_keeps_direct_edge():
    rng = np.random.default_rng(7)
    L = np.full((10, 1), 0.5)
    Bc = np.zeros((10, 10)); Bc[0, 1] = 0.9
    X = pd.DataFrame(C.simulate_multifactor_scm(5000, L, df=3, B_coef=Bc, rng=rng), columns=list("abcdefghij"))
    adj = C.adjusted_tail_coef_matrix(X, kind="psi")["adj"]
    assert adj.loc["a", "b"] > 0.8 and adj.loc["a", "b"] > adj.loc["b", "a"] + 0.1


def test_adjusted_coef_input_validation():
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((100, 2)))
    with pytest.raises(ValueError):
        C.adjusted_tail_coef_matrix(X)                                  # < 3 mã, không có market
    with pytest.raises(ValueError):
        C.adjusted_tail_coef_matrix(pd.concat([X, X], axis=1), kind="khac")


# ----------------------------------------------------------------------------- Granger trong đuôi
def test_benjamini_hochberg_known_values():
    q = C.benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)
    assert len(C.benjamini_hochberg([])) == 0


def test_extremal_granger_detects_lagged_crash_spillover():
    rng = np.random.default_rng(8)
    n = 3000
    m = rng.standard_t(4, n)
    x = 0.4 * m + rng.standard_t(3, n)
    y = 0.4 * m + rng.standard_t(3, n)
    y[1:] += 0.8 * np.minimum(x[:-1], 0)                             # cú sập của x hôm nay → y hôm sau
    k = CT.default_k(n)
    fwd = C.extremal_granger(x, y, m, k=k, lag=1, tail="lower", B=199, seed=1)
    bwd = C.extremal_granger(y, x, m, k=k, lag=1, tail="lower", B=199, seed=2)
    assert fwd["delta"] > 0.2 and fwd["p"] < 0.05
    assert bwd["p"] > 0.05


def test_extremal_granger_validation_and_matrix():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.standard_t(4, (500, 3)), columns=["VHM", "NVL", "Á"])
    with pytest.raises(ValueError):
        C.extremal_granger(X.VHM.values, X.NVL.values, X.Á.values, k=10, tail="giua")
    with pytest.raises(ValueError):
        C.extremal_granger(X.VHM.values, X.NVL.values, X.Á.values, k=10, lag=0)
    G = C.extremal_granger_matrix(X, k=10, lag=2, B=20, n_jobs=1)
    assert len(G) == 6 and set(G.columns) >= {"cause", "effect", "gamma", "baseline", "delta", "p", "q"}
    assert G["p"].between(0, 1).all()


# ----------------------------------------------------------------------------- siêu tham số
def test_garch_spec_table_and_k_stability():
    rng = np.random.default_rng(10)
    R = pd.DataFrame(rng.standard_t(5, (800, 2)) * 0.02, columns=["VHM", "NVL"])
    tab = C.garch_spec_table(R)
    assert len(tab) == 2 * len(C.GARCH_SPECS) and tab.bic.notna().all()
    X, _ = _factor_data(n=1500, K=1, p=5, seed=11)
    ks = C.k_stability(X, [0.3, 0.4, 0.5])
    assert list(ks.index) == [0.3, 0.4, 0.5] and ks.agree_neighbors.between(0, 1).all()


# ----------------------------------------------------------------------------- so sánh phương pháp khử nhân tố
def test_sector_loo_adjust_removes_sector_factor():
    rng = np.random.default_rng(12)
    n, sec = 4000, {"a": "X", "b": "X", "c": "X", "d": "Y", "e": "Y", "f": "Z"}
    m = rng.standard_normal(n); s = rng.standard_normal(n)
    X = pd.DataFrame({c: 0.5 * m + (0.6 * s if sec[c] == "X" else 0) + rng.standard_normal(n) for c in sec})
    E = C.sector_loo_adjust(X, sec)
    before, after = abs(np.corrcoef(X.a, s)[0, 1]), abs(np.corrcoef(E.a, s)[0, 1])
    # Ngành chỉ 2 mã khác làm đại diện → sai số trong biến: chỉ loại được MỘT PHẦN nhân tố ngành
    assert before > 0.3 and after < 0.7 * before
    # Hiện tượng phụ đã biết: ngành ít mã → nhân tố LOO chứa nhiễu riêng của mã cùng ngành → tương quan âm giả
    assert np.corrcoef(E.a, E.b)[0, 1] < 0


@pytest.mark.parametrize("method", C.DECONF_METHODS)
def test_method_matrix_shapes(method):
    X, _ = _factor_data(n=800, K=1, p=6, seed=13)
    sec = {c: ("A" if i < 3 else "B") for i, c in enumerate(X.columns)}
    M = C.method_matrix(X, method, sector=sec, corr="pearson")
    assert M.shape == (6, 6) and np.isnan(np.diag(M.values)).all()


def test_method_matrix_errors():
    X, _ = _factor_data(n=300, K=1, p=4, seed=14)
    with pytest.raises(ValueError):
        C.method_matrix(X, "loo_sector")
    with pytest.raises(ValueError):
        C.method_matrix(X, "khac")


# ----------------------------------------------------------------------------- hồi quy sau code review
def test_benjamini_hochberg_ignores_nan():
    q = C.benjamini_hochberg([0.01, np.nan, 0.03])
    assert np.isnan(q[1]) and np.allclose(q[[0, 2]], [0.02, 0.03])


def test_ease_ignores_nan_coefficients():
    M = np.array([[np.nan, 0.9, 0.8], [np.nan, np.nan, 0.7], [0.6, 0.95, np.nan]])
    # điểm: nút0 = 0,6 (bỏ NaN), nút1 = 0,95, nút2 = 0,8 → gốc là nút 0 vì 0,6 nhỏ nhất, không phải vì NaN
    assert CT.ease(M)[0] == 0
    allnan = np.full((3, 3), np.nan); allnan[1, 0] = allnan[2, 0] = 0.4
    assert CT.ease(allnan)[0] == 0                  # nút 1, 2 không có thông tin → điểm 1, không được chọn gốc


def test_granger_ties_broken_randomly_but_reproducibly():
    rng = np.random.default_rng(15)
    n = 1500
    x = np.clip(rng.standard_t(3, n), -1.0, None)            # ~15% phiên trùng tại sàn
    y, m = rng.standard_t(3, n), rng.standard_t(3, n)
    a = C.extremal_granger(x, y, m, k=20, tail="lower", B=30, seed=4)
    b = C.extremal_granger(x, y, m, k=20, tail="lower", B=30, seed=4)
    c = C.extremal_granger(x, y, m, k=20, tail="lower", B=30, seed=5)
    assert a == b and a["gamma"] != c["gamma"]               # tái lập theo seed; seed khác chọn phiên trùng khác
    with pytest.raises(ValueError):
        C.extremal_granger(x, y, m, k=20, B=5, block=5000)


def test_adjusted_matrix_market_and_nan_checks():
    X, _ = _factor_data(n=300, K=1, p=4, seed=16)
    with pytest.raises(ValueError):
        C.adjusted_tail_coef_matrix(X, market=np.zeros(10))
    Xn = X.copy(); Xn.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        C.adjusted_tail_coef_matrix(Xn)
    with pytest.raises(ValueError):
        C.factor_model(X, 1, corr="Kendall")
