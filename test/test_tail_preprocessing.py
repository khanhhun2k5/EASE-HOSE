"""Kiểm thử tiền xử lý đuôi: GARCH, GPD, bootstrap, độ đồng thuận thứ tự."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tail_preprocessing as TP


def test_order_agreement_basic_cases():
    assert TP.order_agreement(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert TP.order_agreement(["a", "b", "c"], ["c", "b", "a"]) == 0.0
    assert TP.order_agreement(["Á"], ["Á"]) == 1.0                       # 1 phần tử: không có cặp
    with pytest.raises(ValueError):
        TP.order_agreement(["VHM", "VHM", "NVL"], ["NVL", "VHM"])     # trùng lặp
    with pytest.raises(ValueError):
        TP.order_agreement(["a", "b"], ["a", "c"])


def test_gpd_positive_shape_for_pareto_tail():
    rng = np.random.default_rng(0)
    x = rng.pareto(3.0, 20_000)                                       # ξ = 1/3
    g = TP.gpd_tail_shape(x, q=0.95)
    assert abs(g["xi"] - 1 / 3) < 0.1 and g["se"] > 0 and g["alpha"] == pytest.approx(1 / g["xi"])


def test_gpd_rejects_heavily_censored_series():
    """≥ 5% quan sát dồn ở trần: ngưỡng 95% trùng giá trị lớn nhất → lỗi rõ ràng về dữ liệu bị chặn."""
    rng = np.random.default_rng(1)
    x = np.clip(rng.standard_t(3, 2000), -2, 0.8)
    assert (x == 0.8).mean() > 0.05
    with pytest.raises(ValueError, match="bị chặn"):
        TP.gpd_tail_shape(x, q=0.95)


def test_gpd_reports_tie_share_and_nan_se_when_bounded():
    rng = np.random.default_rng(2)
    x = np.clip(rng.standard_t(3, 4000), -5, np.quantile(rng.standard_t(3, 4000), 0.98))
    g = TP.gpd_tail_shape(x, q=0.95)
    assert g["tie_share_at_max"] > 0.2
    assert g["xi"] < 0 and (np.isnan(g["se"]) or g["se"] > 0)


def test_garch_standardize_shapes_units_and_convergence():
    rng = np.random.default_rng(3)
    n = 1500
    sig2, r = np.empty(n), np.empty(n)
    sig2[0] = 1e-4
    for t in range(n):
        if t:
            sig2[t] = 1e-6 + 0.1 * r[t - 1] ** 2 + 0.85 * sig2[t - 1]
        r[t] = np.sqrt(sig2[t]) * rng.standard_t(6) / np.sqrt(1.5)
    df = pd.DataFrame({"VHM": r, "HPG": rng.standard_normal(n) * 0.02})
    z, par = TP.garch_standardize(df)
    assert z.shape == (n, 2) and par["converged"].all()
    assert abs(z["VHM"].std() - 1) < 0.15
    assert par.loc["VHM", "omega"] < 1e-4                              # đơn vị lợi suất gốc, không phải %²
    assert 0.8 < par.loc["VHM", "persistence"] < 1.0


def test_bootstrap_coef_shape_and_reproducible():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.standard_t(3, (400, 3)), columns=list("abc"))
    b1 = TP.bootstrap_coef(X, k=10, B=5, seed=7)
    b2 = TP.bootstrap_coef(X, k=10, B=5, seed=7)
    assert b1.shape == (5, 3, 3) and np.allclose(b1, b2, equal_nan=True)
