"""Kiểm thử xử lý biên độ giá: CAcF cho Q_t, GARCH–t có kiểm duyệt, EASE nhiều lần điền, mô phỏng dữ liệu bị chặn.

Toàn bộ dữ liệu là giả lập có seed cố định, đáp án biết trước; không đụng tới data/ thật.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

import causal_tail as CT
import price_limit as PL

# Tham số thật dùng chung cho các test CAcF: Q_t cỡ 1–3, ~10% phiên chạm ngưỡng M = 4, α_t ≈ 4.
THETA_L = dict(mu=0.5, beta0=-0.1, beta1=0.8, beta2=0.06, beta2s=0.05,
               gamma0=0.3 * 1.4 + 0.2 * 1.7, gamma1=0.7, gamma2=0.2, gamma2s=0.05)
M_L = 4.0


def _sim_cacf(seed: int, n: int = 3000, kind: str = "L", theta: dict | None = None) -> dict:
    return PL.simulate_cacf(n, theta or THETA_L, M=M_L, kind=kind, rng=np.random.default_rng(seed))


def _sim_garch(n, omega, a, b, nu, mu, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    r, s2 = np.empty(n), omega / (1 - a - b)
    for t in range(n):
        r[t] = mu + np.sqrt(s2) * eps[t]
        s2 = omega + a * (r[t] - mu) ** 2 + b * s2
    return r


def _clip_symmetric(r, mu, q):
    L = np.quantile(np.abs(r - mu), q)
    x = np.clip(r, mu - L, mu + L)
    return x, r <= mu - L, r >= mu + L


# ----------------------------------------------------------------------------- A. CAcF
def test_static_censored_frechet_recovers_parameters():
    rng = np.random.default_rng(1)
    mu, sigma, alpha, n = 0.5, 1.0, 3.0, 5000
    latent = mu + sigma * (-np.log(rng.random(n))) ** (-1 / alpha)          # CDF exp(−((q − μ)/σ)^(−α))
    M = np.quantile(latent, 0.85)
    D = (latent > M).astype(int)
    fit = PL.fit_static_censored_frechet(np.minimum(latent, M), D)
    assert fit["converged"] and np.isfinite(fit["loglik"])
    assert abs(fit["sigma"] / sigma - 1) < 0.10
    assert abs(fit["alpha"] / alpha - 1) < 0.10
    assert fit["mu"] < latent.min()


def test_simulate_cacf_is_censored_reproducible_and_matches_cdf():
    a = _sim_cacf(0)
    b = _sim_cacf(0)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert (a["Q"] <= M_L + 1e-12).all() and (a["Q"][a["D"] == 1] == M_L).all()
    assert 0.03 < a["D"].mean() < 0.3
    assert (a["Q_latent"] > THETA_L["mu"]).all()
    # PIT: U = exp(−((Q* − μ)/σ_t)^(−α_t)) phải ~ U(0,1) (kiểm tra CDF nhất quán với mô phỏng/likelihood)
    u = np.exp(-(((a["Q_latent"] - THETA_L["mu"]) / a["sigma"]) ** (-a["alpha"])))
    assert abs(u.mean() - 0.5) < 0.03 and abs(np.mean(u < 0.25) - 0.25) < 0.04


def test_fit_cacf_l_recovers_persistence_and_acf_is_more_biased_in_mu():
    fits = [(PL.fit_cacf(s["Q"], s["D"], kind="L", scale=1.0),
             PL.fit_cacf(s["Q"], s["D"], kind="L", censored=False, scale=1.0))
            for s in (_sim_cacf(0), _sim_cacf(1), _sim_cacf(2))]
    # trung bình 3 mẫu: β1, γ1 trong ±0,1 quanh giá trị thật
    assert abs(np.mean([c.params["beta1"] for c, _ in fits]) - 0.8) < 0.1
    assert abs(np.mean([c.params["gamma1"] for c, _ in fits]) - 0.7) < 0.1
    for c, a in fits:
        assert c.converged and a.converged and np.isfinite(c.loglik) and np.isfinite(a.loglik)
        assert "beta2s" in c.params and "beta2s" not in a.params           # AcF bỏ β2*, γ2*
        # bỏ qua kiểm duyệt làm μ lệch nhiều hơn (bài báo, Bảng 1)
        assert abs(a.params["mu"] - THETA_L["mu"]) > abs(c.params["mu"] - THETA_L["mu"])
        assert c.aic == pytest.approx(-2 * c.loglik + 2 * len(c.params))
        assert c.bic == pytest.approx(-2 * c.loglik + len(c.params) * np.log(c.n))
        assert (c.sigma_t > 0).all() and (c.alpha_t > 0).all() and len(c.sigma_t) == c.n


def test_cacf_loglik_at_fit_matches_and_censoring_term_matters():
    s = _sim_cacf(3, n=1500)
    fit = PL.fit_cacf(s["Q"], s["D"], kind="L", scale=1.0)
    ll = PL.cacf_loglik(fit.params, s["Q"], s["D"], kind="L", init=fit.init)
    assert ll == pytest.approx(fit.loglik, rel=1e-9)
    # coi các phiên bị chặn như quan sát đầy đủ ⇒ likelihood khác hẳn (cùng tham số)
    assert PL.cacf_loglik(fit.params, s["Q"], s["D"], kind="L", init=fit.init, censored=False) != pytest.approx(ll)


@pytest.mark.parametrize("kind", ["S", "E"])
def test_fit_cacf_other_kinds_run_and_have_expected_params(kind):
    s = _sim_cacf(4, n=800)
    fit = PL.fit_cacf(s["Q"], s["D"], kind=kind, n_starts=2)               # scale mặc định 100 (đơn vị %)
    assert fit.kind == kind and np.isfinite(fit.loglik) and np.isfinite(fit.aic)
    assert ("beta3" in fit.params and "gamma3" in fit.params) == (kind == "E")
    if kind == "E":
        assert fit.params["beta3"] > 0 and fit.params["gamma3"] > 0
    assert 0 <= fit.params["beta1"] < 1 and 0 <= fit.params["gamma1"] < 1
    assert fit.params["beta2"] > 0 and fit.params["gamma2"] > 0
    assert fit.mu < s["Q"].min() and fit.mu == pytest.approx(fit.params["mu"] / 100)
    assert fit.sigma_t.index.equals(pd.RangeIndex(800))


def test_observation_driven_functions_are_increasing_and_validate_kind():
    z = np.linspace(-3, 6, 200)
    for kind, c in (("L", None), ("S", None), ("E", 0.7)):
        g = PL.observation_driven(kind, z, c)
        assert (np.diff(g) > 0).all(), kind
    assert PL.observation_driven("S", np.array([-2.0, 2.0])).tolist() == [-4.0, 4.0]
    assert PL.observation_driven("E", np.array([0.0]), 0.5)[0] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="kind"):
        PL.observation_driven("X", z)
    with pytest.raises(ValueError, match="β3"):
        PL.observation_driven("E", z)                                          # thiếu β3
    with pytest.raises(ValueError, match="kind"):
        PL.fit_cacf(np.ones(50), np.zeros(50), kind="Q")


def test_cacf_filter_matches_direct_recursion_and_validates():
    s = _sim_cacf(5, n=300)
    sig, alp = PL.cacf_filter(THETA_L, s["Q"], s["D"], "L", init=(s["sigma"][0], s["alpha"][0]))
    np.testing.assert_allclose(sig, s["sigma"], rtol=1e-9)                    # lfilter trùng đệ quy của bộ mô phỏng
    np.testing.assert_allclose(alp, s["alpha"], rtol=1e-9)
    with pytest.raises(ValueError, match="β1"):
        PL.cacf_filter({**THETA_L, "beta1": 1.2}, s["Q"], s["D"])
    with pytest.raises(ValueError, match="thiếu"):
        PL.cacf_filter({"beta0": 0.0}, s["Q"], s["D"])


def test_censored_probability_in_unit_interval_and_monotone_in_threshold():
    s = _sim_cacf(0, n=1500)
    fit = PL.fit_cacf(s["Q"], s["D"], kind="L", scale=1.0)
    p = PL.censored_probability(fit, M_L)
    assert ((p >= 0) & (p <= 1)).all() and len(p) == fit.n
    assert (PL.censored_probability(fit, M_L + 1.0) <= p + 1e-15).all()       # ngưỡng cao hơn ⇒ ít bị chặn hơn
    assert (PL.censored_probability(fit, fit.mu - 1.0) == 1.0).all()           # M ≤ μ ⇒ chắc chắn bị chặn
    # xác suất dự báo cao hơn ở phiên bị chặn thật (so với phiên thường)
    assert p[s["D"] == 1].mean() > p[s["D"] == 0].mean()
    with pytest.raises(ValueError, match="M phải"):
        PL.censored_probability(fit, np.ones(3))


def test_evar_closed_form_monotone_nan_and_matches_quadrature():
    mu, sigma, alpha = 0.5, 1.2, 3.0
    taus = np.array([0.8, 0.9, 0.95, 0.99])
    ev = np.array([PL.evar_frechet(mu, sigma, alpha, t) for t in taus])
    assert (np.diff(ev) > 0).all()
    for t, e in zip(taus, ev):
        # CVaR = μ + σ/(1 − τ)·∫_τ^1 (−log u)^(−1/α) du  (định nghĩa), tích phân bằng quad
        num = integrate.quad(lambda u: (-np.log(u)) ** (-1 / alpha), t, 1)[0]
        assert e == pytest.approx(mu + sigma / (1 - t) * num, rel=1e-6)
    assert ev[0] > mu + sigma * (-np.log(0.8)) ** (-1 / alpha)               # EVaR ≥ VaR cùng mức
    out = PL.evar_frechet(mu, sigma, np.array([0.8, 1.0, 1.5, 3.0]), 0.9)
    assert np.isnan(out[:2]).all() and np.isfinite(out[2:]).all()            # α ≤ 1: kỳ vọng không tồn tại
    with pytest.raises(ValueError, match="τ"):
        PL.evar_frechet(mu, sigma, alpha, 1.0)


def test_censored_evar_not_above_uncensored_and_bounded_by_threshold():
    s = _sim_cacf(1, n=1500)
    fit = PL.fit_cacf(s["Q"], s["D"], kind="L", scale=1.0)
    for tau in (0.9, 0.99):
        unc, cen = PL.evar(fit, tau), PL.evar_censored(fit, tau, M_L)
        ok = unc.notna()
        assert ok.mean() > 0.5 and (cen[ok] <= unc[ok] + 1e-9).all() and (cen <= M_L + 1e-12).all()
        assert unc[~ok].empty or (fit.alpha_t[~ok] <= 1).all()          # NaN chỉ khi α_t ≤ 1
    hi, lo = PL.evar(fit, 0.99), PL.evar(fit, 0.9)
    assert (hi.dropna() > lo.dropna()).all()
    # ngưỡng rất cao ⇒ EVaR kiểm duyệt ≈ không kiểm duyệt (khi α > 1); ngưỡng dưới VaR_τ ⇒ đúng bằng ngưỡng
    big = PL.evar_censored_frechet(0.5, 1.0, 4.0, 0.9, M=1e6, n_grid=4000)[0]
    assert big == pytest.approx(PL.evar_frechet(0.5, 1.0, 4.0, 0.9), rel=2e-3)
    assert PL.evar_censored_frechet(0.5, 1.0, 4.0, 0.9, M=0.8)[0] == pytest.approx(0.8)


def test_predict_cacf_continues_the_filter_out_of_sample():
    s = _sim_cacf(2, n=1000)
    Qs = pd.Series(s["Q"], index=pd.date_range("2018-06-01", periods=1000, freq="B"))
    fit = PL.fit_cacf(Qs.iloc[:900], s["D"][:900], kind="L")                # scale = 100
    assert fit.sigma_t.index.equals(Qs.index[:900])
    pred = PL.predict_cacf(fit, Qs.iloc[900:], s["D"][900:], M_L)
    assert pred.index.equals(Qs.index[900:]) and list(pred.columns) == ["sigma", "alpha", "p_censored"]
    # tham chiếu: lọc toàn bộ chuỗi với cùng tham số/khởi đầu (đơn vị làm việc = Q·100)
    sig, alp = PL.cacf_filter(fit.params, s["Q"] * 100, s["D"].astype(float), "L", init=fit.init)
    np.testing.assert_allclose(pred["sigma"].to_numpy(), sig[900:] / 100, rtol=1e-8)
    np.testing.assert_allclose(pred["alpha"].to_numpy(), alp[900:], rtol=1e-8)
    assert ((pred["p_censored"] >= 0) & (pred["p_censored"] <= 1)).all()
    with pytest.raises(ValueError, match="cùng độ dài"):
        PL.predict_cacf(fit, Qs.iloc[900:], s["D"][:50], M_L)


def test_qt_series_censors_when_any_stock_hits_floor_and_rejects_nan_flags():
    """Sàn của A (−6,9%, làm tròn bước giá) thấp hơn mức lỗ của B (−6,95%, không chạm sàn): lỗ ẩn của A có thể vượt
    Q_t nên phiên vẫn bị kiểm duyệt tại M = Q_t."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    r = pd.DataFrame({"A": [-0.069, 0.01], "B": [-0.0695, -0.02]}, index=idx)
    fl = pd.DataFrame({"A": [True, False], "B": [False, False]}, index=idx)
    out = PL.qt_series(r, fl)
    assert out["argmax"].tolist() == ["B", "B"] and out["D"].tolist() == [1, 0]
    assert out["M"].iloc[0] == pytest.approx(0.0695) and np.isnan(out["M"].iloc[1])
    bad = fl.astype(object)
    bad.iloc[1, 0] = None
    with pytest.raises(ValueError, match="NaN"):
        PL.qt_series(r, bad)


def test_censored_garch_rejects_misaligned_flags_and_nan_flags():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    r = pd.Series(rng.standard_t(5, 120) * 0.01, index=idx)
    fl = pd.Series(False, index=idx)
    with pytest.raises(ValueError, match="cùng chỉ số"):
        PL.fit_censored_garch(r, fl.shift(1, freq="B"), np.zeros(120, bool))
    with pytest.raises(ValueError, match="NaN"):
        PL.fit_censored_garch(r.to_numpy(), np.array([None] + [False] * 119, dtype=object), np.zeros(120, bool))


def test_cacf_state_is_not_clipped_so_filter_and_simulation_agree():
    th = {**THETA_L, "gamma0": 7.0, "gamma1": 0.5}             # log α dừng ≈ 14 > log 300: giá trị dùng bị cắt
    s = PL.simulate_cacf(400, th, M_L, "L", np.random.default_rng(3))
    sig, alp = PL.cacf_filter(th, s["Q"], s["D"], "L")
    np.testing.assert_allclose(sig, s["sigma"], rtol=1e-10)
    np.testing.assert_allclose(alp, s["alpha"], rtol=1e-10)


def test_qt_series_picks_cross_sectional_max_loss_and_its_floor_flag():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    r = pd.DataFrame({"A": [-0.07, 0.01, -0.02, np.nan], "B": [-0.03, -0.05, 0.02, -0.01], "C": [0.02, -0.01, np.nan, np.nan]},
                     index=idx)
    fl = pd.DataFrame({"A": [True, False, False, False], "B": [False, False, False, False],
                       "C": [False, False, False, False]}, index=idx)
    out = PL.qt_series(r, fl)
    assert out["Q"].tolist() == pytest.approx([0.07, 0.05, 0.02, 0.01])
    assert out["D"].tolist() == [1, 0, 0, 0] and out["argmax"].tolist() == ["A", "B", "A", "B"]
    assert out["M"].iloc[0] == pytest.approx(0.07) and out["M"].iloc[1:].isna().all()
    assert out["n_floor"].tolist() == [1, 0, 0, 0] and out.index.equals(idx)
    with pytest.raises(ValueError, match="toàn NaN"):
        PL.qt_series(r.assign(A=np.nan, B=np.nan, C=np.nan), fl)
    with pytest.raises(ValueError, match="cùng kích thước"):
        PL.qt_series(r, fl.iloc[:3])


def test_cacf_input_validation_errors():
    Q, D = np.linspace(0.5, 3, 60), np.zeros(60)
    with pytest.raises(ValueError, match="cùng độ dài"):
        PL.fit_cacf(Q, D[:50])
    with pytest.raises(ValueError, match="NaN"):
        PL.fit_cacf(np.where(np.arange(60) == 3, np.nan, Q), D)
    with pytest.raises(ValueError, match="0 hoặc 1"):
        PL.fit_cacf(Q, np.full(60, 2.0))
    with pytest.raises(ValueError, match="ít nhất"):
        PL.fit_static_censored_frechet(Q[:10], D[:10])
    with pytest.raises(ValueError, match="μ < min"):
        PL.cacf_loglik({**THETA_L, "mu": Q.min() + 0.1}, Q, D)               # Q ≤ μ
    with pytest.raises(ValueError, match="scale"):
        PL.fit_cacf(Q, D, scale=0.0)
    with pytest.raises(ValueError, match="β1"):
        PL.simulate_cacf(100, {**THETA_L, "gamma1": 1.0}, M_L)


# ----------------------------------------------------------------------------- B. GARCH–t có kiểm duyệt
def test_plain_censored_garch_matches_arch():
    from arch import arch_model
    r = _sim_garch(2000, 0.05, 0.08, 0.88, 5, 0.03, seed=1)
    no = np.zeros(len(r), bool)
    mine = PL.fit_censored_garch(r, no, no, censored=False, limit_terms=False, scale=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    p = mine["params"]
    assert p["converged"] and p["delta_floor"] == 0 and p["delta_ceil"] == 0
    assert abs(p["loglik"] - ref.loglikelihood) / len(r) < 1e-3
    assert abs(p["alpha[1]"] - ref.params["alpha[1]"]) < 0.03 and abs(p["beta[1]"] - ref.params["beta[1]"]) < 0.03
    assert abs(p["nu"] - ref.params["nu"]) / ref.params["nu"] < 0.15
    # z_obs gần với phần dư chuẩn hoá của arch (cùng cấu trúc, khác điểm khởi đầu σ²)
    assert np.corrcoef(mine["z_obs"], ref.std_resid)[0, 1] > 0.99
    assert mine["cut_lo"].isna().all() and mine["cut_hi"].isna().all()


def test_censored_garch_units_scale_and_index():
    r = pd.Series(_sim_garch(1200, 5e-6, 0.08, 0.88, 6, 3e-4, seed=2) , index=pd.date_range("2019-01-01", periods=1200, freq="B"))
    no = pd.Series(False, index=r.index)
    a = PL.fit_censored_garch(r, no, no, censored=False, limit_terms=False, scale=100.0)
    b = PL.fit_censored_garch(r, no, no, censored=False, limit_terms=False, scale=None)
    assert a["sigma"].index.equals(r.index)
    # lợi suất cỡ 1e-3: tham số theo đơn vị gốc nhất quán giữa hai cách chọn scale
    assert a["params"]["omega"] == pytest.approx(b["params"]["omega"], rel=0.2)
    assert a["params"]["mu"] == pytest.approx(b["params"]["mu"], abs=3e-4)
    assert abs(a["sigma"].mean() / r.std() - 1) < 0.25


def test_censored_garch_recovers_tail_index_better_than_plain():
    err_plain, err_cens, pers_err = [], [], []
    for seed in (10, 11, 12):
        r = _sim_garch(2500, 0.05, 0.10, 0.85, 4, 0.03, seed)
        x, F, C = _clip_symmetric(r, 0.03, 0.94)                              # ~6% phiên bị chặn
        plain = PL.fit_censored_garch(x, F, C, censored=False, limit_terms=False, scale=1.0)["params"]
        cens = PL.fit_censored_garch(x, F, C, censored=True, limit_terms=True, scale=1.0)["params"]
        err_plain.append(abs(plain["nu"] - 4))
        err_cens.append(abs(cens["nu"] - 4))
        pers_err.append(abs(cens["persistence"] - 0.95))
    assert np.mean(err_cens) < np.mean(err_plain) / 2                         # ν: kiểm duyệt tốt hơn rõ rệt
    assert np.mean(err_cens) < 1.0
    assert np.mean(pers_err) < 0.05                                           # a + b phục hồi hợp lý


def test_censored_garch_standardize_api_and_alignment():
    cols = ["AAA", "BBB"]
    idx = pd.date_range("2020-01-01", periods=900, freq="B")
    R = pd.DataFrame({c: _sim_garch(900, 0.05, 0.08, 0.88, 5, 0.0, 20 + i) for i, c in enumerate(cols)}, index=idx)
    R.iloc[5, 0] = np.nan                                                      # NaN bị bỏ như TP.garch_standardize
    floor, ceil = pd.DataFrame(False, index=idx, columns=cols), pd.DataFrame(False, index=idx, columns=cols)
    for c in cols:
        L = R[c].abs().quantile(0.97)
        floor[c], ceil[c] = R[c] < -L, R[c] > L
        R[c] = R[c].clip(-L, L)
    Z, par = PL.censored_garch_standardize(R, floor, ceil, scale=1.0)
    assert list(Z.columns) == cols and list(par.index) == cols
    assert Z["AAA"].notna().sum() == 899 and np.isnan(Z["AAA"].iloc[5])
    assert {"mu", "omega", "alpha[1]", "beta[1]", "nu", "persistence", "loglik", "converged", "delta_floor"} <= set(par.columns)
    assert par["converged"].all()
    cut_lo, cut_hi = PL.censoring_cuts(Z, floor, ceil)
    assert cut_lo.notna().sum().sum() == int((floor & Z.notna()).sum().sum()) and (cut_lo.to_numpy()[cut_lo.notna().to_numpy()] < 0).all()
    assert (cut_hi.to_numpy()[cut_hi.notna().to_numpy()] > 0).all()
    with pytest.raises(ValueError, match="thiếu các cột"):
        PL.censored_garch_standardize(R, floor[["AAA"]], ceil)


def test_censored_garch_input_validation_errors():
    r = _sim_garch(200, 0.05, 0.08, 0.88, 5, 0.0, 1)
    no = np.zeros(200, bool)
    with pytest.raises(ValueError, match="floor"):
        PL.fit_censored_garch(r, no[:100], no)
    with pytest.raises(ValueError, match="NaN"):
        PL.fit_censored_garch(np.where(np.arange(200) == 7, np.nan, r), no, no)
    with pytest.raises(ValueError, match="vừa chạm sàn vừa chạm trần"):
        PL.fit_censored_garch(r, np.ones(200, bool), np.ones(200, bool))
    with pytest.raises(ValueError, match="ít nhất 50"):
        PL.fit_censored_garch(r[:20], no[:20], no[:20])
    with pytest.raises(ValueError, match="hằng số"):
        PL.fit_censored_garch(np.zeros(200), no, no)
    s = pd.Series(r, index=pd.RangeIndex(200))
    with pytest.raises(ValueError, match="cùng chỉ số"):
        PL.fit_censored_garch(s, pd.Series(no, index=pd.RangeIndex(1, 201)), pd.Series(no, index=pd.RangeIndex(1, 201)))


# ----------------------------------------------------------------------------- impute / expected
def _toy_cuts(seed=0, n=600, p=3):
    rng = np.random.default_rng(seed)
    Z = pd.DataFrame(rng.standard_normal((n, p)), columns=list("XYZ")[:p])
    lo, hi = pd.DataFrame(np.nan, index=Z.index, columns=Z.columns), pd.DataFrame(np.nan, index=Z.index, columns=Z.columns)
    for j, c in enumerate(Z.columns):
        fl, ce = rng.choice(n, 25, replace=False), rng.choice(n, 25, replace=False)
        ce = np.setdiff1d(ce, fl)
        Z.loc[fl, c], Z.loc[ce, c] = -1.8 - 0.1 * j, 1.7 + 0.1 * j
        lo.loc[fl, c], hi.loc[ce, c] = Z.loc[fl, c], Z.loc[ce, c]
    return Z, lo, hi


def test_impute_latent_respects_truncation_and_is_reproducible():
    Z, lo, hi = _toy_cuts()
    nu = pd.Series([4.0, 6.0, 9.0], index=Z.columns)
    imps = PL.impute_latent(Z, lo, hi, nu, 5, np.random.default_rng(1))
    again = PL.impute_latent(Z, lo, hi, nu, 5, np.random.default_rng(1))
    assert len(imps) == 5
    normal = lo.isna() & hi.isna()
    for A, B in zip(imps, again):
        pd.testing.assert_frame_equal(A, B)                                    # tái lập theo seed
        assert (A.values[lo.notna().values] <= lo.values[lo.notna().values] + 1e-12).all()
        assert (A.values[hi.notna().values] >= hi.values[hi.notna().values] - 1e-12).all()
        assert (A.where(normal) .equals(Z.where(normal)))                      # phiên thường giữ nguyên
        assert A.index.equals(Z.index) and np.isfinite(A.values).all()
    assert not imps[0].equals(imps[1])                                         # các bản khác nhau
    assert (imps[0].values[lo.notna().values] < lo.values[lo.notna().values]).mean() > 0.99   # điền sâu hơn ngưỡng
    with pytest.raises(ValueError, match="ν"):
        PL.impute_latent(Z, lo, hi, 1.5, 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="m phải"):
        PL.impute_latent(Z, lo, hi, 5.0, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="cùng kích thước"):
        PL.impute_latent(Z, lo.iloc[:10], hi, 5.0, 2, np.random.default_rng(0))


def test_expected_latent_matches_mean_of_imputations_and_numeric_integral():
    Z, lo, hi = _toy_cuts(seed=3, n=400, p=2)
    nu = 5.0
    E = PL.expected_latent(Z, lo, hi, nu)
    imps = PL.impute_latent(Z, lo, hi, nu, 600, np.random.default_rng(2))
    mean_imp = np.mean([i.to_numpy() for i in imps], axis=0)
    cens = (lo.notna() | hi.notna()).to_numpy()
    assert np.abs(mean_imp[cens] - E.to_numpy()[cens]).max() < 0.12            # sai số Monte Carlo của 600 bản
    assert (E.to_numpy()[lo.notna().to_numpy()] < lo.to_numpy()[lo.notna().to_numpy()]).all()
    assert (E.to_numpy()[hi.notna().to_numpy()] > hi.to_numpy()[hi.notna().to_numpy()]).all()
    # đối chiếu tích phân số cho một ngưỡng: E[ε | ε ≤ c], ε ~ t(ν) chuẩn hoá
    from scipy import stats
    s, c = np.sqrt(nu / (nu - 2)), -1.3
    num = integrate.quad(lambda x: x * s * stats.t.pdf(s * x, nu), -np.inf, c)[0] / stats.t.cdf(s * c, nu)
    one = PL.expected_latent(pd.DataFrame({"X": [c]}), pd.DataFrame({"X": [c]}), pd.DataFrame({"X": [np.nan]}), nu)
    assert one["X"].iloc[0] == pytest.approx(num, rel=1e-6)


# ----------------------------------------------------------------------------- C. EASE nhiều lần điền
def test_ease_mi_single_copy_equals_plain_ease_and_returns_valid_order():
    rng = np.random.default_rng(5)
    Z = pd.DataFrame(rng.standard_t(3, (800, 5)), columns=list("abcde"))
    Z["b"] += 0.8 * Z["a"]
    k = CT.default_k(800)
    one = PL.ease_mi([Z], k)
    ref = CT.tail_coef_matrix(Z, k=k)
    pd.testing.assert_frame_equal(one["coef"], ref)
    assert one["order"] == CT.ease(ref) and one["orders"] == [one["order"]] and one["agreement"] == 1.0
    copies = [Z, Z + 0.01 * rng.standard_normal(Z.shape), Z.copy()]
    many = PL.ease_mi(copies, k)
    assert sorted(many["order"]) == list("abcde") and len(many["orders"]) == 3
    assert 0.0 <= many["agreement"] <= 1.0 and len(many["agreement_each"]) == 3
    np.testing.assert_allclose(many["coef"].to_numpy()[~np.eye(5, dtype=bool)],
                               np.mean([CT.tail_coef_matrix(z, k=k).to_numpy() for z in
                                        copies], axis=0)[~np.eye(5, dtype=bool)], atol=1e-12)
    # deconf được áp lên từng bản (ở đây: trừ trung bình dòng) — và nhận mảng numpy
    dec = PL.ease_mi([Z.to_numpy(), Z.to_numpy()], k, deconf=lambda X: np.asarray(X) - np.asarray(X).mean(axis=1, keepdims=True))
    assert sorted(dec["order"]) == list(range(5))
    with pytest.raises(ValueError, match="rỗng"):
        PL.ease_mi([], k)


# ----------------------------------------------------------------------------- D. mô phỏng bị chặn
def test_simulate_limit_scm_hit_rate_limits_and_flags():
    rng = np.random.default_rng(7)
    sim = PL.simulate_limit_scm(3000, 6, 2.5, rng, hit_rate=0.03)
    X, R, L = sim["X"], sim["latent"], sim["limit"]
    assert X.shape == R.shape == (3000, 6) and sim["floor"].shape == sim["ceil"].shape == X.shape
    assert (np.abs(X) <= L + 1e-12).all()
    rate = (sim["floor"] | sim["ceil"]).mean(axis=0)
    assert np.abs(rate - 0.03).max() < 0.01                                    # ±1 điểm %
    assert not (sim["floor"] & sim["ceil"]).any()
    np.testing.assert_array_equal(sim["floor"], R < -L)                        # cờ khớp giá trị
    np.testing.assert_array_equal(sim["ceil"], R > L)
    assert np.allclose(X[sim["floor"]], np.broadcast_to(-L, X.shape)[sim["floor"]])
    assert np.allclose(X[sim["ceil"]], np.broadcast_to(L, X.shape)[sim["ceil"]])
    assert sim["adj"].shape[0] == 6 + 2 and sim["observed"] == list(range(2, 8))
    assert np.array_equal(X[~(sim["floor"] | sim["ceil"])], R[~(sim["floor"] | sim["ceil"])])
    # hit_rate theo từng mã
    sim2 = PL.simulate_limit_scm(3000, 4, 2.5, np.random.default_rng(8), hit_rate=[0.01, 0.02, 0.04, 0.08])
    r2 = (sim2["floor"] | sim2["ceil"]).mean(axis=0)
    assert np.all(np.diff(r2) > 0) and abs(r2[3] - 0.08) < 0.015


def test_simulate_limit_scm_spill_conserves_total_and_is_reproducible():
    kw = dict(n=1500, p=4, alpha=2.5, hit_rate=0.05)
    a = PL.simulate_limit_scm(rng=np.random.default_rng(3), spill=True, **kw)
    b = PL.simulate_limit_scm(rng=np.random.default_rng(3), spill=True, **kw)
    np.testing.assert_array_equal(a["X"], b["X"])
    # tổng lợi suất tiềm ẩn = tổng quan sát + phần dư còn dồn ở cuối kỳ
    np.testing.assert_allclose(a["latent"].sum(axis=0), a["X"].sum(axis=0) + a["carry"], atol=1e-8)
    assert (np.abs(a["X"]) <= a["limit"] + 1e-12).all()
    nos = PL.simulate_limit_scm(rng=np.random.default_rng(3), spill=False, **kw)
    assert (nos["carry"] == 0).all()
    assert (a["floor"] | a["ceil"]).mean() >= (nos["floor"] | nos["ceil"]).mean()   # dồn phiên ⇒ chạm biên nhiều hơn
    assert np.abs(a["latent"].sum(axis=0) - a["X"].sum(axis=0)).max() < np.abs(nos["latent"].sum(axis=0) - nos["X"].sum(axis=0)).max()


def test_simulate_limit_scm_input_errors():
    with pytest.raises(ValueError, match="hit_rate"):
        PL.simulate_limit_scm(500, 3, 2.0, hit_rate=0.7)
    with pytest.raises(ValueError, match="hit_rate"):
        PL.simulate_limit_scm(500, 3, 2.0, hit_rate=[0.01, 0.02])              # sai độ dài → không broadcast được
    with pytest.raises(ValueError, match="vol_persist"):
        PL.simulate_limit_scm(500, 3, 2.0, vol_persist=1.0)
    with pytest.raises(ValueError, match="n phải"):
        PL.simulate_limit_scm(5, 3, 2.0)


def test_limit_simulation_study_small_run_shape_and_reproducibility():
    kw = dict(reps=1, n=500, p=4, alphas=(2.0,), hit_rates=(0.05,), seed=3, m_imp=3, n_jobs=1)
    out = PL.limit_simulation_study(**kw)
    assert list(out.columns) == ["alpha", "hit_rate", "rep", "method", "SID"]
    assert out["method"].tolist() == list(PL.SIM_METHODS) and len(out) == 4
    assert out["SID"].between(0, 1).all()
    pd.testing.assert_frame_equal(out, PL.limit_simulation_study(**kw))


def test_projected_gradient_is_kkt_at_bounds():
    """Ở cận dưới, gradient hướng RA ngoài (g > 0) là hợp lệ; gradient hướng VÀO trong (g < 0) nghĩa là chưa tối ưu."""
    b = [(0.0, 10.0)]
    assert PL._projected_grad_max(lambda x: float(x[0]), np.array([0.0]), b) < 1e-6         # min tại biên: hội tụ
    assert PL._projected_grad_max(lambda x: float(-x[0]), np.array([0.0]), b) > 0.5         # nên đi vào trong
    assert PL._projected_grad_max(lambda x: float(-x[0]), np.array([10.0]), b) < 1e-6       # min tại cận trên
    x, f, ok = PL._minimize(lambda x: float((x[0] - 3.0) ** 2), [np.array([9.0])], b)
    assert ok and abs(x[0] - 3.0) < 1e-3


def test_fit_cacf_without_compensation_is_nested_for_lr_test():
    s = PL.simulate_cacf(1500, THETA_L, M_L, "L", np.random.default_rng(4))
    full = PL.fit_cacf(s["Q"], s["D"], kind="L", scale=1.0)
    red = PL.fit_cacf(s["Q"], s["D"], kind="L", scale=1.0, compensation=False)
    assert "beta2s" not in red.params and "gamma2s" not in red.params
    assert full.loglik >= red.loglik - 1e-6                      # mô hình lồng nhau: đầy đủ không kém hơn

