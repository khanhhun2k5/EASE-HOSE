"""Kiểm thử module nonlinear_factor: PCA, autoencoder (tuyến tính / ReLU / hệ số tải phi tuyến), cross-fitting,
R², β tăng/giảm, mô phỏng nhân tố bất đối xứng."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nonlinear_factor as NF

pytest.importorskip("torch")

FAST = dict(epochs=150, patience=20, batch=256)


def _asym(n=2000, p=20, up=0.4, down=0.4, df=6.0, seed=0):
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.05, 0.05, p)                             # cùng độ lệch cho β⁺, β⁻ (up = down ⇒ đối xứng)
    bu, bd = up + jitter, down + jitter
    X = NF.simulate_asymmetric_factor_scm(n, bu, bd, df, rng=rng)
    return pd.DataFrame(X, columns=[f"S{i}" for i in range(p)]), bu, bd


# ----------------------------------------------------------------------------- cấu hình
@pytest.mark.parametrize("kw", [dict(arch="mlp"), dict(K=0), dict(hidden=0), dict(val_frac=1.0),
                                dict(l1=-1.0), dict(lr=0.0), dict(n_ensemble=0)])
def test_config_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        NF.AEConfig(**kw)


# ----------------------------------------------------------------------------- PCA
def test_pca_recovers_one_factor_and_is_projection():
    X, _, _ = _asym(up=0.7, down=0.7)
    m = NF.fit_pca(X, 1)
    R2 = NF.r2_total(X, m.predict(X))
    assert 0.4 < R2 < 0.6                                            # phần chung ≈ 0,49 phương sai
    S = (X.to_numpy() - m.mu) / m.sd
    P = m._predict(S)
    assert np.allclose(m._predict(P), P, atol=1e-10)                 # chiếu hai lần = chiếu một lần
    assert m.factors(X).shape == (len(X), 1)


def test_pca_and_model_input_validation():
    X, _, _ = _asym(p=5)
    with pytest.raises(ValueError):
        NF.fit_pca(X, 5)
    with pytest.raises(ValueError):
        NF.fit_pca(X.assign(S0=1.0), 1)                              # cột hằng
    m = NF.fit_pca(X, 1)
    with pytest.raises(ValueError):
        m.predict(X.iloc[:, :3])


# ----------------------------------------------------------------------------- autoencoder
def test_linear_autoencoder_matches_pca():
    """Gu, Kelly & Xiu (2021), Mệnh đề 1: AE tuyến tính một lớp ẩn ≡ PCA (cùng không gian nhân tố)."""
    X, _, _ = _asym(up=0.6, down=0.6, seed=1)
    pca = NF.fit_pca(X, 1).predict(X)
    ae = NF.fit_autoencoder(X, NF.AEConfig(K=1, arch="linear", l1=0.0, **FAST)).predict(X)
    assert abs(NF.r2_total(X, ae) - NF.r2_total(X, pca)) < 0.005
    assert np.corrcoef(ae.ravel(), pca.ravel())[0, 1] > 0.99


def test_homogeneous_asymmetry_is_absorbed_by_linear_factor():
    """Mọi mã cùng hình dạng g(F) = β⁺F⁺ + β⁻F⁻ (cùng tỷ lệ β⁻/β⁺): nhân tố tuyến tính f ∝ g(F) đã khớp đúng,
    nên AE phi tuyến KHÔNG hơn PCA đáng kể — phi tuyến chỉ có ích khi độ bất đối xứng khác nhau giữa các mã."""
    X, _, _ = _asym(n=3000, up=0.15, down=0.85, seed=2)
    pca = NF.r2_total(X, NF.fit_pca(X, 1).predict(X))
    nl = NF.r2_total(X, NF.fit_autoencoder(X, NF.AEConfig(K=1, arch="nonlinear", **FAST)).predict(X))
    assert abs(nl - pca) < 0.01


def test_nonlinear_autoencoder_learns_heterogeneous_asymmetry():
    """Nửa số mã β⁻ ≫ β⁺, nửa kia ngược lại: một β tuyến tính khớp kém ở cả hai đuôi; AE hệ số tải phi tuyến
    (một nhân tố) khớp tốt hơn rõ rệt."""
    rng = np.random.default_rng(2)
    p = 20
    bu = np.where(np.arange(p) < p // 2, 0.15, 0.8)
    X = pd.DataFrame(NF.simulate_asymmetric_factor_scm(3000, bu, bu[::-1], 6.0, rng=rng))
    pca = NF.fit_report(X, pd.DataFrame(NF.fit_pca(X, 1).predict(X)))
    m = NF.fit_autoencoder(X, NF.AEConfig(K=1, arch="nonlinear", lr=1e-2, **FAST))
    nl = NF.fit_report(X, pd.DataFrame(m.predict(X)))
    # Mức trần với một nhân tố tuyến tính f: hồi quy hai bước x ~ f⁺ + f⁻ (≈ +0,033 R² tổng ở dữ liệu này)
    assert nl["R2"] > pca["R2"] + 0.025
    assert nl["R2_duoi_duoi"] > pca["R2_duoi_duoi"] + 0.08 and nl["R2_duoi_tren"] > pca["R2_duoi_tren"] + 0.05
    assert len(m.history) == 1 and len(m.history[0]) >= 1


def test_nonneg_relu_autoencoder_respects_constraints():
    X, _, _ = _asym(p=8, seed=3)
    cfg = NF.AEConfig(K=2, arch="relu", nonneg=True, lam_orth=1e-2, lam_decor=1e-2, **FAST)
    m = NF.fit_autoencoder(X, cfg)
    assert np.isfinite(m.predict(X)).all() and m.factors(X).min() >= 0     # nhân tố sau ReLU không âm


def test_nonneg_projection_is_applied():
    import torch

    X, _, _ = _asym(p=8, seed=4)
    S = (X.to_numpy() - X.to_numpy().mean(0)) / X.to_numpy().std(0)
    cfg = NF.AEConfig(K=2, arch="relu", nonneg=True, **FAST)
    net, _ = NF._train_one(S, cfg, NF._pca_loadings(S, 2), seed=0)
    W = net.dec.weight.detach()
    assert (net.enc.weight.detach() >= 0).all() and (W >= 0).all()
    assert torch.allclose(W.norm(dim=0), torch.ones(2), atol=1e-4)


def test_ensemble_averages_and_is_reproducible():
    X, _, _ = _asym(p=6, seed=5)
    cfg = NF.AEConfig(K=1, arch="nonlinear", n_ensemble=2, seed=7, **FAST)
    a, b = NF.fit_autoencoder(X, cfg), NF.fit_autoencoder(X, cfg)
    assert len(a.history) == 2
    assert np.allclose(a.predict(X), b.predict(X))                   # cùng seed → cùng kết quả


def test_divergent_training_raises_clear_error_and_restores_threads():
    import torch

    X, _, _ = _asym(p=5, seed=15)
    S = X.to_numpy().copy()
    S[:, 0] = np.inf                                                 # loss kiểm định không hữu hạn ngay epoch đầu
    with pytest.raises(FloatingPointError, match="lr"):
        NF._train_one(S, NF.AEConfig(K=1, **FAST), np.ones((5, 1)) / np.sqrt(5), seed=0)
    before = torch.get_num_threads()
    NF.fit_autoencoder(X, NF.AEConfig(K=1, epochs=3))
    assert torch.get_num_threads() == before                         # không để lại set_num_threads(1)


def test_autoencoder_rejects_nan():
    X, _, _ = _asym(p=5)
    X.iloc[3, 1] = np.nan
    with pytest.raises(ValueError):
        NF.fit_autoencoder(X, NF.AEConfig(K=1, **FAST))


# ----------------------------------------------------------------------------- cross-fitting
def test_time_folds_are_contiguous_and_balanced():
    f = NF.time_folds(103, 5)
    assert f[0] == 0 and f[-1] == 4 and np.all(np.diff(f) >= 0)
    sizes = np.bincount(f)
    assert sizes.max() - sizes.min() <= 1
    with pytest.raises(ValueError):
        NF.time_folds(10, 1)


def test_crossfit_is_out_of_sample():
    """Dự báo khối f phải đúng bằng mô hình huấn luyện KHÔNG có khối f (PCA: tất định)."""
    X, _, _ = _asym(n=500, p=6, seed=6)
    cf = NF.crossfit(X, NF.AEConfig(K=1, arch="pca"), n_folds=4)
    fold = cf["fold"].to_numpy()
    for f in range(4):
        m = NF.fit_pca(X[fold != f], 1)
        assert np.allclose(cf["fitted"][fold == f], m.predict(X[fold == f]))
    assert np.allclose(cf["resid"] + cf["fitted"], X)
    assert cf["factors"].shape == (500, 1)


def test_crossfit_parallel_equals_serial_and_rejects_nan():
    X, _, _ = _asym(n=400, p=5, seed=8)
    cfg = NF.AEConfig(K=1, arch="nonlinear", epochs=20, patience=5)
    a = NF.crossfit(X, cfg, n_folds=3, n_jobs=1)["fitted"]
    b = NF.crossfit(X, cfg, n_folds=3, n_jobs=2)["fitted"]
    assert np.allclose(a, b, atol=1e-6)
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        NF.crossfit(X, cfg)


# ----------------------------------------------------------------------------- đánh giá
def test_r2_total_known_values():
    X = np.array([[1.0, 2.0], [3.0, -1.0]])
    assert NF.r2_total(X, X) == 1.0
    assert NF.r2_total(X, np.zeros_like(X)) == 0.0
    assert np.allclose(NF.r2_total(X, np.array([[1.0, 0.0], [3.0, 0.0]]), axis=0), [1.0, 0.0])


def test_market_tail_mask():
    X = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    lo, up = NF.market_tail_mask(X, 0.1, "lower"), NF.market_tail_mask(X, 0.1, "upper")
    assert lo.sum() == 10 and lo[:10].all() and up.sum() == 10 and up[-10:].all()
    for bad in (dict(q=0.0), dict(q=0.5), dict(tail="both")):
        with pytest.raises(ValueError):
            NF.market_tail_mask(X, **({"q": 0.1, "tail": "lower"} | bad))


def test_up_down_betas_recover_asymmetry():
    X, bu, bd = _asym(n=20000, p=12, up=0.3, down=0.7, seed=9)
    b = NF.up_down_betas(X)
    # β trên thị trường LOO (trung bình 11 mã) lớn hơn β trên nhân tố thật một hệ số chung; tỷ lệ và dấu giữ nguyên
    assert (b["beta_down"] > b["beta_up"]).all() and (b["t_D"] > 3).all()
    Xs, _, _ = _asym(n=20000, p=12, up=0.5, down=0.5, seed=10)
    assert NF.up_down_betas(Xs)["t_D"].abs().max() < 4                # đối xứng: không có khác biệt lớn


def test_up_down_betas_with_given_market():
    rng = np.random.default_rng(11)
    m = pd.Series(rng.standard_normal(5000))
    x = 0.2 * np.maximum(m, 0) + 0.9 * np.minimum(m, 0) + 0.1 * rng.standard_normal(5000)
    b = NF.up_down_betas(pd.DataFrame({"x": x, "y": rng.standard_normal(5000)}), market=m)
    assert b.loc["x", "beta_up"] == pytest.approx(0.2, abs=0.02)
    assert b.loc["x", "beta_down"] == pytest.approx(0.9, abs=0.02)


# ----------------------------------------------------------------------------- mô phỏng
def test_simulation_has_unit_variance_and_reduces_to_linear():
    X, _, _ = _asym(n=200000, p=4, up=0.3, down=0.8, df=5, seed=12)
    assert np.allclose(X.var(), 1.0, atol=0.03)
    Xl, bu, _ = _asym(n=200000, p=4, up=0.6, down=0.6, df=5, seed=13)
    C = np.corrcoef(Xl.to_numpy(), rowvar=False)
    assert C[0, 1] == pytest.approx(bu[0] * bu[1], abs=0.02)        # β⁺ = β⁻: một nhân tố tuyến tính


def test_simulation_edges_and_validation():
    rng = np.random.default_rng(14)
    B = np.zeros((3, 3))
    B[0, 1] = 0.8
    X = NF.simulate_asymmetric_factor_scm(50000, [0.0] * 3, [0.0] * 3, 6, B_coef=B, rng=rng)
    C = np.corrcoef(X, rowvar=False)
    assert C[0, 1] > 0.5 and abs(C[0, 2]) < 0.03
    with pytest.raises(ValueError):
        NF.simulate_asymmetric_factor_scm(10, [1.0], [1.2], 6)       # phần chung ≥ 1
    with pytest.raises(ValueError):
        NF.simulate_asymmetric_factor_scm(10, [0.5], [0.5, 0.5], 6)
    with pytest.raises(ValueError):
        NF.simulate_asymmetric_factor_scm(10, [0.5], [0.5], 2)
