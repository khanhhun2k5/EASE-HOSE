"""Hiệu chỉnh mô hình (giai đoạn Calibration): khử nhân tố chung, nhân quả đuôi theo thời gian, chọn siêu tham số.

Bối cảnh: Validation cho thấy một mô hình một nhân tố KHÔNG có nhân quả giữa các mã tái hiện được thứ tự EASE.
Module này cài các công cụ sau (nguồn trích dẫn ghi ở từng mục):

1. **Số nhân tố** — Ahn & Horenstein (2013, Econometrica, doi:10.3982/ECTA8968): ER(k) = μ_k/μ_{k+1},
   GR(k) = ln(1 + μ_k/V_k)/ln(1 + μ_{k+1}/V_{k+1}); giá trị riêng "giả" μ_0 = V(0)/ln m.
2. **Mô hình nhân tố vững** — tải nhân tố từ ma trận tương quan Kendall đã biến đổi sin(πτ/2)
   (Fan, Liu, Wang 2018, JRSSB, doi:10.1111/rssb.12265), điểm nhân tố bằng hồi quy GLS-đơn giản.
3. **Hệ số đuôi có điều kiện thị trường** — chỉ dùng các phiên thị trường (không chứa cặp đang xét) KHÔNG cực trị,
   so với mức nền trên cùng tập phiên (ý tưởng "adjusted CTC", Pasche và cs. 2023, doi:10.1007/s10687-022-00456-4;
   tập điều kiện kiểu Bodik & Pasche 2024, arXiv:2407.09632).
4. **Granger trong đuôi** — Γ_{X→Y} = E[max_{h≤p} F̃_Y(Y_{t+h}) | X_t cực trị, Y_t và thị trường không cực trị]
   so với mức nền; kiểm định bằng moving-block bootstrap; hiệu chỉnh Benjamini–Hochberg (Bodik & Pasche 2024).
5. **Siêu tham số** — chọn dạng GARCH theo BIC; chọn số mũ ν của k = ⌊n^ν⌋ theo độ ổn định của thứ tự EASE.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

import causal_tail as CT


# ============================================================================ 1. số nhân tố
def n_factors_ahn_horenstein(X: pd.DataFrame | np.ndarray, kmax: int = 6) -> dict:
    """Ước lượng số nhân tố bằng ER và GR của Ahn & Horenstein (2013).

    Dữ liệu được chuẩn hoá từng cột; μ_k là giá trị riêng thứ k (giảm dần) của XXᵀ/(NT) (≡ XᵀX/(NT));
    V_k = Σ_{j>k} μ_j. Thêm μ_0 = V_0 / ln(min(N, T)) để cho phép kết luận K = 0.
    Trả dict: eigen (μ_1..μ_m), ER, GR (chỉ số k = 0..kmax), K_ER, K_GR.
    """
    A = np.asarray(X, float)
    T, N = A.shape
    if kmax < 1 or kmax >= min(N, T) - 1:
        raise ValueError("kmax phải trong [1, min(N, T) − 2].")
    A = (A - A.mean(0)) / A.std(0, ddof=0)
    mu = np.sort(np.linalg.eigvalsh(A.T @ A / (N * T)))[::-1]
    V = np.array([mu[k:].sum() for k in range(len(mu) + 1)])          # V[k] = Σ_{j>k} μ_j (chỉ số từ 1)
    mu_full = np.r_[V[0] / np.log(min(N, T)), mu]                       # mu_full[k] = μ_k, k = 0..m
    ER = np.array([mu_full[k] / mu_full[k + 1] for k in range(kmax + 1)])
    GR = np.array([np.log(1 + mu_full[k] / V[k]) / np.log(1 + mu_full[k + 1] / V[k + 1])
                   for k in range(kmax + 1)])
    return {"eigen": mu, "ER": ER, "GR": GR, "K_ER": int(np.argmax(ER)), "K_GR": int(np.argmax(GR))}


# ============================================================================ 2. mô hình nhân tố vững
def kendall_correlation(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Ma trận tương quan sin(π τ / 2) từ Kendall τ — ước lượng vững của tương quan elliptic (đuôi dày)."""
    A = np.asarray(X, float)
    p = A.shape[1]
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            tau = stats.kendalltau(A[:, i], A[:, j]).statistic
            R[i, j] = R[j, i] = np.sin(np.pi * tau / 2)
    return R


def factor_model(X: pd.DataFrame, K: int, corr: str = "kendall") -> dict:
    """Mô hình K nhân tố trên dữ liệu chuẩn hoá: tải Λ (p × K) = K vectơ riêng đầu × √giá trị riêng của ma trận
    tương quan (Kendall hoặc Pearson); điểm nhân tố F = X_s Λ (ΛᵀΛ)⁻¹; phần dư E = X_s − F Λᵀ (theo đơn vị X).

    K = 0 trả phần dư = X. Trả dict: loadings (DataFrame), factors (DataFrame n × K), resid (DataFrame), R.
    """
    if K < 0 or K >= X.shape[1]:
        raise ValueError("K phải trong [0, p − 1].")
    if corr not in ("kendall", "pearson"):
        raise ValueError("corr phải là 'kendall' hoặc 'pearson'.")
    mu, sd = X.mean(), X.std(ddof=0)
    Xs = (X - mu) / sd
    R = kendall_correlation(Xs) if corr == "kendall" else np.corrcoef(Xs.to_numpy(), rowvar=False)
    if K == 0:
        return {"loadings": pd.DataFrame(index=X.columns), "factors": pd.DataFrame(index=X.index),
                "resid": X.copy(), "R": R}
    w, v = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1][:K]
    L = v[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    L *= np.where(L.sum(axis=0) < 0, -1, 1)                            # dấu: tổng tải dương
    F = Xs.to_numpy() @ L @ np.linalg.inv(L.T @ L)
    E = (Xs.to_numpy() - F @ L.T) * sd.to_numpy()
    cols = [f"F{i + 1}" for i in range(K)]
    return {"loadings": pd.DataFrame(L, index=X.columns, columns=cols),
            "factors": pd.DataFrame(F, index=X.index, columns=cols),
            "resid": pd.DataFrame(E, index=X.index, columns=X.columns), "R": R}


def simulate_multifactor_scm(n: int, loadings: np.ndarray, df: float, B_coef: np.ndarray | None = None,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    """X = F Λᵀ + (√(1 − ‖λ_i‖²) ⊙ ε)(I − B)⁻¹, F (n × K), ε ~ Student-t(df) chuẩn hoá phương sai 1, F ẩn.

    ``B_coef[i, j]``: hệ số cạnh i → j giữa các mã (None: không có nhân quả). ‖λ_i‖ phải < 1.
    """
    rng = rng if rng is not None else np.random.default_rng()
    L = np.asarray(loadings, float)
    if L.ndim == 1:                                   # vectơ (p,) = một nhân tố
        L = L[:, None]
    if L.ndim != 2:
        raise ValueError("loadings phải là vectơ (p,) hoặc ma trận p × K.")
    p, K = L.shape
    comm = (L ** 2).sum(axis=1)
    if (comm >= 1).any():
        raise ValueError("Mỗi mã phải có tổng bình phương tải < 1.")
    if df <= 2:
        raise ValueError("df phải > 2.")
    scale = np.sqrt((df - 2) / df)
    F = rng.standard_t(df, (n, K)) * scale
    U = rng.standard_t(df, (n, p)) * scale * np.sqrt(1 - comm)
    if B_coef is not None:
        U = U @ np.linalg.inv(np.eye(p) - np.asarray(B_coef, float))
    return F @ L.T + U


# ============================================================================ 3. hệ số đuôi có điều kiện thị trường
def _pair_market(total: np.ndarray, A: np.ndarray, j: int, l: int) -> np.ndarray:
    p = A.shape[1]
    return (total - A[:, j] - A[:, l]) / (p - 2)


def adjusted_tail_coef_matrix(X: pd.DataFrame | np.ndarray, k: int | None = None, kind: str = "psi",
                              market_quantile: float = 0.9, market: np.ndarray | None = None,
                              seed: int = 0) -> dict:
    """Hệ số đuôi có điều kiện "thị trường không cực trị".

    Với cặp (j → l): thị trường m_t = trung bình đều các mã KHÁC j và l (hoặc ``market`` cho trước).
    Tập phiên hợp lệ E = {t : σ(F̂_m(m_t)) ≤ q} với σ(u) = |2u − 1| (kind="psi", loại cả cú tăng lẫn cú sập
    của thị trường) hoặc F̂_m(m_t) ≤ q (gamma_upper) / F̂_m(m_t) ≥ 1 − q (gamma_lower). Trong E, lấy k phiên
    cực trị của X_j và tính thống kê như hệ số gốc (thang hạng F̂ trên toàn mẫu), so với mức nền
    b_{jl} = trung bình thống kê đó trên toàn E. Chuẩn hoá:
        Ψ̃_{jl} = 1/2 + 1/2 · (Ψ_{jl} − b_{jl}) / (1 − b_{jl})   (độc lập có điều kiện → 1/2; kéo theo hoàn toàn → 1).

    Trả dict: adj (p × p, đã chuẩn hoá), raw (chưa chuẩn hoá), base (mức nền), n_eligible (số phiên hợp lệ).
    """
    if kind not in CT.COEF_KINDS:
        raise ValueError(f"kind phải thuộc {CT.COEF_KINDS}")
    if not 0 < market_quantile <= 1:
        raise ValueError("market_quantile phải trong (0, 1].")
    labels = list(X.columns) if isinstance(X, pd.DataFrame) else None
    A = np.asarray(X, float)
    n, p = A.shape
    if market is None and p < 3:
        raise ValueError("Cần ≥ 3 mã để dựng thị trường không chứa cặp đang xét (hoặc truyền market).")
    k = CT.default_k(n) if k is None else k
    if np.isnan(A).any():
        raise ValueError("Dữ liệu có NaN — hãy bỏ các dòng thiếu trước.")
    if market is not None:
        if isinstance(market, pd.Series) and isinstance(X, pd.DataFrame):
            market = market.reindex(X.index)
        market = np.asarray(market, float)
        if market.shape != (n,) or np.isnan(market).any():
            raise ValueError("market phải là chuỗi độ dài n, khớp chỉ mục với X và không có NaN.")
    F = (stats.rankdata(A, axis=0, method="average") - 0.5) / n
    T = {"psi": np.abs(2 * F - 1), "gamma_upper": F, "gamma_lower": 1 - F}[kind]
    total = A.sum(axis=1)
    raw, base, adj = (np.full((p, p), np.nan) for _ in range(3))
    n_elig = np.zeros((p, p), dtype=int)
    for j in range(p):
        rng = np.random.default_rng(seed + j)
        tb_up, tb_lo = rng.random(n), rng.random(n)
        x = A[:, j] if kind != "gamma_lower" else -A[:, j]
        for l in range(p):
            if l == j:
                continue
            m = market if market is not None else _pair_market(total, A, j, l)
            Fm = (stats.rankdata(m, method="average") - 0.5) / n
            ok = {"psi": np.abs(2 * Fm - 1) <= market_quantile, "gamma_upper": Fm <= market_quantile,
                  "gamma_lower": Fm >= 1 - market_quantile}[kind]
            idx = np.flatnonzero(ok)
            kk = min(k, len(idx) - 1)
            if kk < 1:
                continue
            up = idx[np.lexsort((tb_up[idx], x[idx]))[-kk:]]
            if kind == "psi":
                lo = idx[np.lexsort((tb_lo[idx], x[idx]))[:kk]]
                r = 0.5 * T[up, l].mean() + 0.5 * T[lo, l].mean()
            else:
                r = T[up, l].mean()
            b = T[idx, l].mean()
            raw[j, l], base[j, l], n_elig[j, l] = r, b, len(idx)
            adj[j, l] = 0.5 + 0.5 * (r - b) / (1 - b) if b < 1 else np.nan
    wrap = (lambda M: pd.DataFrame(M, index=labels, columns=labels)) if labels else (lambda M: M)
    return {"adj": wrap(adj), "raw": wrap(raw), "base": wrap(base), "n_eligible": wrap(n_elig)}


# ============================================================================ 4. Granger trong đuôi
def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """q-value Benjamini–Hochberg (kiểm soát FDR)."""
    p = np.asarray(pvalues, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)                                   # NaN (không kiểm định được) giữ NaN, không tính vào m
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    q = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    qq = np.empty(m)
    qq[order] = np.minimum(q, 1.0)
    out[ok] = qq
    return out


def _granger_parts(x: np.ndarray, y: np.ndarray, m: np.ndarray, lag: int, tail: str, y_quantile: float,
                   market_quantile: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chuẩn bị (x_t, w_t, eligible_t) cho t = 0..n−lag−1, với w_t = max_{h=1..lag} F̃_Y(y_{t+h})."""
    if tail not in ("lower", "upper"):
        raise ValueError("tail phải là 'lower' hoặc 'upper'.")
    s = -1.0 if tail == "lower" else 1.0
    x, y, m = s * np.asarray(x, float), s * np.asarray(y, float), s * np.asarray(m, float)
    n = len(x)
    if lag < 1 or lag >= n:
        raise ValueError("lag phải trong [1, n − 1].")
    Fy = (stats.rankdata(y, method="average") - 0.5) / n
    Fm = (stats.rankdata(m, method="average") - 0.5) / n
    Ft = np.where(Fy >= 0.5, Fy, 0.0)                     # F̃: phân phối thực nghiệm, đặt 0 dưới trung vị
    nn = n - lag
    W = np.column_stack([Ft[h:h + nn] for h in range(1, lag + 1)]).max(axis=1)
    elig = (Fy[:nn] <= y_quantile) & (Fm[:nn] <= market_quantile)
    return x[:nn], W, elig


def _granger_stat(xs: np.ndarray, W: np.ndarray, elig: np.ndarray, k: int, tie: np.ndarray) -> tuple[float, float]:
    idx = np.flatnonzero(elig)
    kk = min(k, len(idx) - 1)
    if kk < 1:
        return np.nan, np.nan
    # k phiên lớn nhất; trùng giá trị (biên độ ±7%) phá thế bằng khoá ngẫu nhiên có seed như các hệ số khác
    top = idx[np.lexsort((tie[idx], xs[idx]))[-kk:]]
    return float(W[top].mean()), float(W[idx].mean())


def extremal_granger(x: np.ndarray, y: np.ndarray, market: np.ndarray, k: int, lag: int = 1, tail: str = "lower",
                     y_quantile: float = 0.8, market_quantile: float = 0.9, B: int = 300,
                     block: int | None = None, seed: int = 0) -> dict:
    """Granger trong đuôi X → Y (theo Bodik & Pasche 2024, bản đơn giản hoá ghi rõ dưới đây).

    Γ̂ = trung bình max_{h=1..lag} F̃_Y(y_{t+h}) trên k phiên t có x_t cực trị (đuôi ``tail``) trong số các phiên
    "sạch": y_t không cực trị (F̂_Y(y_t) ≤ ``y_quantile``) và thị trường không cực trị cùng chiều
    (F̂_M(m_t) ≤ ``market_quantile``). Mức nền b = trung bình cùng đại lượng trên mọi phiên sạch.
    Δ = Γ̂ − b > 0: cú sốc cực trị của X hôm nay làm Y dễ cực trị trong ``lag`` phiên tới, vượt mức thông thường.

    Kiểm định H0: Δ ≤ 0 bằng moving-block bootstrap (khối dài ``block``, mặc định ⌈√n⌉) trên các bộ
    (x_t, w_t, sạch_t), p = P*(Δ* − Δ̂ ≥ Δ̂). Khác bản gốc: điều kiện chỉ xét phiên t (không xét cả p phiên quá khứ).
    Trả dict: gamma, baseline, delta, p, extreme_rule (Γ̂ > (1 + b)/2).
    """
    xs, W, elig = _granger_parts(x, y, market, lag, tail, y_quantile, market_quantile)
    nn = len(xs)
    rng = np.random.default_rng(seed)
    tie = rng.random(nn)
    g, b = _granger_stat(xs, W, elig, k, tie)
    d = g - b
    L = block or int(np.ceil(np.sqrt(nn)))
    if not 1 <= L <= nn:
        raise ValueError(f"block phải trong [1, {nn}], nhận {L}.")
    n_blocks = int(np.ceil(nn / L))
    boot = np.empty(B)
    for i in range(B):
        starts = rng.integers(0, nn - L + 1, n_blocks)
        ix = (starts[:, None] + np.arange(L)).ravel()[:nn]
        gb, bb = _granger_stat(xs[ix], W[ix], elig[ix], k, tie[ix])
        boot[i] = gb - bb
    boot = boot[~np.isnan(boot)]
    p = float((1 + np.sum(boot - d >= d)) / (len(boot) + 1)) if len(boot) and not np.isnan(d) else np.nan
    return {"gamma": g, "baseline": b, "delta": d, "p": p, "extreme_rule": bool(g > (1 + b) / 2)}


def extremal_granger_matrix(X: pd.DataFrame, k: int | None = None, lag: int = 1, tail: str = "lower",
                            B: int = 300, seed: int = 0, n_jobs: int = -1, **kw) -> pd.DataFrame:
    """Granger trong đuôi cho mọi cặp có hướng (j → l); thị trường của cặp = trung bình các mã khác j, l.
    Chạy song song theo nguyên nhân j (joblib). Trả DataFrame dài: cause, effect, gamma, baseline, delta, p, q (BH)."""
    from joblib import Parallel, delayed

    A = X.to_numpy(float)
    n, p = A.shape
    if p < 3:
        raise ValueError("Cần ≥ 3 mã.")
    k = CT.default_k(n) if k is None else k
    total = A.sum(axis=1)
    names = list(X.columns)

    def for_cause(j):
        out = []
        for l in range(p):
            if l != j:
                r = extremal_granger(A[:, j], A[:, l], _pair_market(total, A, j, l), k=k, lag=lag, tail=tail,
                                     B=B, seed=seed + 1000 * j + l, **kw)
                out.append({"cause": names[j], "effect": names[l], **r})
        return out

    rows = Parallel(n_jobs=n_jobs)(delayed(for_cause)(j) for j in range(p))
    df = pd.DataFrame([r for rr in rows for r in rr])
    df["q"] = benjamini_hochberg(df["p"].fillna(1.0))
    return df


# ============================================================================ 5. siêu tham số
GARCH_SPECS = {
    "GARCH(1,1)-t": dict(vol="GARCH", p=1, o=0, q=1, dist="t"),
    "GJR-GARCH(1,1)-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH(1,1)-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
    "GARCH(1,1)-skewt": dict(vol="GARCH", p=1, o=0, q=1, dist="skewt"),
}


def garch_spec_table(returns: pd.DataFrame, specs: dict | None = None, scale: float = 100.0) -> pd.DataFrame:
    """BIC của từng dạng GARCH cho từng mã (thấp hơn = tốt hơn) và cờ hội tụ. Trả DataFrame dài."""
    from arch import arch_model

    specs = specs or GARCH_SPECS
    rows = []
    for col in returns.columns:
        r = returns[col].dropna() * scale
        for name, spec in specs.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fit = arch_model(r, mean="Constant", **spec).fit(disp="off")
            rows.append({"ma": col, "dang": name, "bic": fit.bic, "loglik": fit.loglikelihood,
                         "converged": fit.convergence_flag == 0})
    return pd.DataFrame(rows)


def k_stability(X: pd.DataFrame, nus: Sequence[float], kind: str = "psi") -> pd.DataFrame:
    """Thứ tự EASE theo ν (k = ⌊n^ν⌋) và độ đồng thuận cặp với hai ν kề bên (tiêu chí "vùng ổn định":
    chọn ν có độ đồng thuận lân cận cao nhất). Trả DataFrame index ν: k, order, agree_neighbors."""
    from tail_preprocessing import order_agreement

    nus = list(nus)
    orders = [CT.ease(CT.tail_coef_matrix(X, k=CT.default_k(len(X), nu), kind=kind)) for nu in nus]
    agree = []
    for i, o in enumerate(orders):
        nb = [orders[j] for j in (i - 1, i + 1) if 0 <= j < len(orders)]
        agree.append(float(np.mean([order_agreement(o, q) for q in nb])) if nb else np.nan)
    return pd.DataFrame({"k": [CT.default_k(len(X), nu) for nu in nus], "order": orders,
                         "agree_neighbors": agree}, index=pd.Index(nus, name="nu"))


# ============================================================================ 6. so sánh phương pháp khử nhân tố
DECONF_METHODS = ("raw", "loo", "loo_sector", "pca", "conditional")
DECONF_LABELS = {"raw": "Gốc (không khử)", "loo": "Loại thị trường LOO", "loo_sector": "Loại thị trường + ngành LOO",
                 "pca": "Loại K nhân tố PCA", "conditional": "Có điều kiện thị trường"}


def sector_loo_adjust(X: pd.DataFrame, sector: dict) -> pd.DataFrame:
    """Phần dư OLS của từng mã i trên [thị trường LOO (mọi mã khác i), ngành LOO (các mã cùng ngành khác i)].
    Mã duy nhất trong ngành chỉ được loại thị trường.

    Lưu ý: ngành ít mã thì nhân tố ngành LOO của i chứa nhiễu riêng của từng mã cùng ngành, nên phần dư của các mã
    cùng ngành có **tương quan âm giả** (khoảng −1/(m−1) với m mã/ngành khi nhân tố ngành yếu). Đọc kết quả
    trong nhóm ngành nhỏ một cách thận trọng. Ngoài ra nhân tố LOO là biến đại diện có nhiễu (sai số trong biến), nên
    với ngành 2–4 mã chỉ loại được MỘT PHẦN nhân tố ngành thật."""
    A = X.to_numpy(float)
    n, p = A.shape
    total = A.sum(axis=1)
    names = list(X.columns)
    out = np.empty_like(A)
    for i, s in enumerate(names):
        peers = [j for j, t in enumerate(names) if j != i and sector[t] == sector[s]]
        cols = [(total - A[:, i]) / (p - 1)]
        if peers:
            cols.append(A[:, peers].mean(axis=1))
        D = np.column_stack([np.ones(n), *cols])
        coef, *_ = np.linalg.lstsq(D, A[:, i], rcond=None)
        out[:, i] = A[:, i] - D @ coef
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def method_matrix(X: pd.DataFrame, method: str, k: int | None = None, K: int = 1, sector: dict | None = None,
                  corr: str = "kendall") -> pd.DataFrame:
    """Ma trận Ψ̂ (p × p) sau khi khử nhân tố chung bằng ``method`` (xem DECONF_LABELS)."""
    if method == "raw":
        return CT.tail_coef_matrix(X, k=k)
    if method == "loo":
        from validation import loo_market_adjust
        return CT.tail_coef_matrix(loo_market_adjust(X)[0], k=k)
    if method == "loo_sector":
        if sector is None:
            raise ValueError("Cần sector để loại nhân tố ngành.")
        return CT.tail_coef_matrix(sector_loo_adjust(X, sector), k=k)
    if method == "pca":
        return CT.tail_coef_matrix(factor_model(X, K=K, corr=corr)["resid"], k=k)
    if method == "conditional":
        return adjusted_tail_coef_matrix(X, k=k)["adj"]
    raise ValueError(f"method phải thuộc {DECONF_METHODS}")
