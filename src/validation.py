"""Kiểm chứng (validation) kết quả EASE ở giai đoạn Processing.

Bốn hướng:
1. **Nhân tố thị trường**: loại nhân tố thị trường quan sát được (hồi quy OLS) rồi chạy lại EASE trên phần dư.
2. **Kiểm định hoán vị**: (a) giả thuyết độc lập — xáo trộn thời gian từng cột; (b) giả thuyết không có chiều —
   hoán đổi ngẫu nhiên hai biến trong từng phiên trên thang hạng (phân phối đồng thời đối xứng).
3. **Ổn định theo thời gian**: EASE trên cửa sổ trượt và theo giai đoạn.
4. **Mô phỏng hiệu chỉnh**: sinh dữ liệu từ mô hình một nhân tố ẩn (H0: không có nhân quả giữa các mã) và từ
   mô hình nhân tố + DAG (H1), với tham số khớp từ dữ liệu thật, để xem kết quả thật có tái hiện được không.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

import causal_tail as CT


# ============================================================================ 1. nhân tố thị trường
def market_adjust(X: pd.DataFrame, factor: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Phần dư e_i = X_i − a_i − b_i·F (OLS từng cột) và hệ số tải b_i. Các dòng thiếu bị bỏ."""
    df = pd.concat([X, factor.rename("_F")], axis=1, join="inner").dropna()
    f = df.pop("_F").to_numpy()
    resid, betas = {}, {}
    for c in df.columns:
        b, a = np.polyfit(f, df[c].to_numpy(), 1)
        resid[c] = df[c].to_numpy() - a - b * f
        betas[c] = b
    return pd.DataFrame(resid, index=df.index), pd.Series(betas, name="beta")


def loo_market(X: pd.DataFrame) -> pd.DataFrame:
    """Nhân tố thị trường leave-one-out: cột i = trung bình đều của mọi cột khác i (cùng phiên).

    Dùng thay nhân tố "ngoài mẫu" khi các mã lớn đều nằm trong mẫu: không chứa chính mã i nên tránh tương quan
    cơ học giữa mã và nhân tố của nó. Cần ít nhất 2 cột.
    """
    if X.shape[1] < 2:
        raise ValueError("Cần ít nhất 2 mã để dựng nhân tố leave-one-out.")
    if X.isna().to_numpy().any():
        raise ValueError("Dữ liệu có NaN — hãy bỏ các phiên thiếu (dropna) trước khi dựng nhân tố LOO.")
    total = X.sum(axis=1)
    return X.apply(lambda col: (total - col) / (X.shape[1] - 1))


def loo_market_adjust(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Phần dư e_i = X_i − a_i − b_i·MKT_LOO,i (OLS từng mã với nhân tố leave-one-out riêng của nó)."""
    M = loo_market(X)
    resid, betas = {}, {}
    for c in X.columns:
        b, a = np.polyfit(M[c].to_numpy(), X[c].to_numpy(), 1)
        resid[c] = X[c].to_numpy() - a - b * M[c].to_numpy()
        betas[c] = b
    return pd.DataFrame(resid, index=X.index), pd.Series(betas, name="beta_loo")


# ============================================================================ 2. kiểm định hoán vị
def permutation_null_independence(X: pd.DataFrame | np.ndarray, k: int, kind: str = "psi", B: int = 500,
                                  seed: int = 0) -> np.ndarray:
    """Phân phối của ma trận hệ số khi các cột độc lập: xáo trộn thứ tự thời gian của từng cột riêng rẽ
    (giữ nguyên phân phối biên, phá mọi phụ thuộc đồng thời). Trả mảng B × p × p."""
    A = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    n, p = A.shape
    out = np.empty((B, p, p))
    for b in range(B):
        P = np.column_stack([A[rng.permutation(n), j] for j in range(p)])
        out[b] = CT.tail_coef_matrix(P, k=k, kind=kind, seed=seed * 7919 + b * p)
    return out


def swap_null_asymmetry(x: np.ndarray, y: np.ndarray, k: int, kind: str = "psi", B: int = 1000,
                        seed: int = 0) -> tuple[float, np.ndarray]:
    """Kiểm định chiều của cặp (x, y): thống kê D = coef(x→y) − coef(y→x).

    H0: phân phối đồng thời của (F_x(X), F_y(Y)) đối xứng (hoán đổi được) — khi đó không có chiều ưu tiên và
    D ≈ 0. Dưới H0, hoán đổi ngẫu nhiên hai thành phần trong từng phiên (trên thang hạng, để hai biên giống
    nhau) không đổi phân phối. Trả (D quan sát, mảng B giá trị D dưới H0).
    """
    u = stats.rankdata(np.asarray(x, float)) / (len(x) + 1)
    v = stats.rankdata(np.asarray(y, float)) / (len(y) + 1)
    rng = np.random.default_rng(seed)

    def stat(a, b, s):
        # Ma trận 2 × 2 vector hoá: hàng 0 dùng seed s, hàng 1 dùng s + 1 (giống hai lời gọi theo cặp).
        M = CT.tail_coef_matrix(np.column_stack([a, b]), k=k, kind=kind, seed=s)
        return M[0, 1] - M[1, 0]

    d_obs = stat(u, v, seed)
    null = np.empty(B)
    for i in range(B):
        flip = rng.random(len(u)) < 0.5
        a, b = np.where(flip, v, u), np.where(flip, u, v)
        null[i] = stat(a, b, seed + 2 * i + 2)
    return float(d_obs), null


def swap_test_pairs(X: pd.DataFrame, pairs: Sequence[tuple[str, str]], k: int, kind: str = "psi",
                    B: int = 999, seed: int = 0, n_jobs: int = -1) -> pd.DataFrame:
    """Kiểm định hoán đổi cho nhiều cặp, song song bằng joblib; mỗi cặp có dải seed riêng không chồng nhau nên kết quả
    không phụ thuộc số tiến trình. Trả DataFrame: cap, D, q025, q975, p (hai phía), p_holm, chieu."""
    from joblib import Parallel, delayed

    def one(i, a, b):
        # mỗi cặp dùng các seed seed + i·(2B + 10) … + 2B + 3: các khoảng KHÔNG chồng nhau giữa các cặp
        d, null = swap_null_asymmetry(X[a].to_numpy(), X[b].to_numpy(), k=k, kind=kind, B=B,
                                      seed=seed + i * (2 * B + 10))
        return {"cap": f"{a}–{b}", "D": d, "q025": float(np.quantile(null, 0.025)),
                "q975": float(np.quantile(null, 0.975)), "p": perm_pvalue(d, null),
                "chieu": f"{a}→{b}" if d > 0 else f"{b}→{a}"}

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(i, a, b) for i, (a, b) in enumerate(pairs))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"]) if len(out) else []
    from calibration import benjamini_hochberg
    out["q_bh"] = benjamini_hochberg(out["p"]) if len(out) else []
    return out


def maxT_pvalues(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """p-value hiệu chỉnh single-step max-T (Westfall & Young, 1993) cho kiểm định một phía "lớn hơn".

    ``observed``: ma trận p × p (đường chéo bỏ qua); ``null``: B × p × p từ CÙNG các hoán vị cho mọi cặp.
    p̃_{jl} = (1 + #{b : max_{cặp} null_b ≥ obs_{jl}}) / (B + 1). Kiểm soát FWER mà không cần B rất lớn như Holm
    (Holm với 380 cặp cần p < 0,05/380 ≈ 1,3·10⁻⁴, không đạt được khi B = 500)."""
    obs = np.asarray(observed, float)
    p = obs.shape[0]
    off = ~np.eye(p, dtype=bool)
    mx = np.nanmax(np.asarray(null, float)[:, off], axis=1)
    out = np.full((p, p), np.nan)
    out[off] = (1 + (mx[None, :] >= obs[off][:, None]).sum(axis=1)) / (len(mx) + 1)
    return out


def perm_pvalue(observed: float, null: np.ndarray, alternative: str = "two-sided") -> float:
    """p-value hoán vị có hiệu chỉnh +1 (Phipson & Smyth, 2010): (1 + #{null cực đoan bằng/hơn}) / (B + 1)."""
    null = np.asarray(null, float)
    if alternative == "greater":
        extreme = np.sum(null >= observed)
    elif alternative == "less":
        extreme = np.sum(null <= observed)
    elif alternative == "two-sided":
        extreme = np.sum(np.abs(null) >= abs(observed))
    else:
        raise ValueError("alternative phải là 'greater', 'less' hoặc 'two-sided'.")
    return float((1 + extreme) / (len(null) + 1))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni cho nhiều kiểm định (kiểm soát FWER)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ============================================================================ 3. ổn định theo thời gian
def rolling_ease(X: pd.DataFrame, window: int, step: int, nu: float = 0.4, kind: str = "psi") -> pd.DataFrame:
    """EASE trên các cửa sổ trượt dài ``window`` phiên, bước ``step``. Mỗi dòng: ngày cuối cửa sổ, k dùng,
    vị trí của từng biến (1 = gốc) và trung bình Ψ̂ ngoài đường chéo."""
    if window > len(X):
        raise ValueError("window dài hơn dữ liệu.")
    rows = []
    for end in range(window, len(X) + 1, step):
        W = X.iloc[end - window:end]
        k = CT.default_k(window, nu)
        M = CT.tail_coef_matrix(W, k=k, kind=kind)
        order = CT.ease(M)
        off = ~np.eye(M.shape[0], dtype=bool)
        rows.append({"end": W.index[-1], "k": k, "coef_mean": float(M.values[off].mean()),
                     **{f"pos_{c}": order.index(c) + 1 for c in X.columns}})
    return pd.DataFrame(rows).set_index("end")


def period_ease(X: pd.DataFrame, periods: dict[str, tuple[str, str]], nu: float = 0.4,
                kind: str = "psi") -> pd.DataFrame:
    """EASE theo từng giai đoạn {tên: (từ, đến)} (bao gồm hai đầu)."""
    rows = []
    for name, (a, b) in periods.items():
        W = X.loc[a:b]
        k = CT.default_k(len(W), nu)
        M = CT.tail_coef_matrix(W, k=k, kind=kind)
        off = ~np.eye(M.shape[0], dtype=bool)
        rows.append({"giai_doan": name, "n": len(W), "k": k, "thu_tu": " → ".join(CT.ease(M)),
                     "coef_mean": float(M.values[off].mean())})
    return pd.DataFrame(rows).set_index("giai_doan")


# ============================================================================ 4. mô phỏng hiệu chỉnh
def fit_one_factor(X: pd.DataFrame) -> pd.Series:
    """Hệ số tải λ_i của mô hình một nhân tố trên tương quan hạng (điểm chuẩn Φ⁻¹(rank/(n+1))):
    corr(i, j) ≈ λ_i λ_j. Ước lượng bằng thành phần chính đầu tiên của ma trận tương quan, hiệu chỉnh lặp
    phần đường chéo (principal axis factoring, 50 vòng). Dấu chọn sao cho tổng λ dương."""
    Zs = stats.norm.ppf(stats.rankdata(X.to_numpy(float), axis=0) / (len(X) + 1))
    R = np.corrcoef(Zs, rowvar=False)
    w, v = np.linalg.eigh(R)
    lam = np.clip(v[:, -1] * np.sqrt(w[-1]), -0.99, 0.99)
    for _ in range(50):
        Rr = R.copy()
        np.fill_diagonal(Rr, lam ** 2)
        w, v = np.linalg.eigh(Rr)
        lam = np.clip(v[:, -1] * np.sqrt(max(w[-1], 0)), -0.99, 0.99)
    if lam.sum() < 0:
        lam = -lam
    return pd.Series(lam, index=X.columns, name="lambda")


def simulate_factor_scm(n: int, loadings: Sequence[float], df: float, B_coef: np.ndarray | None = None,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """X = λ·F + (I − Bᵀ)⁻¹·(√(1 − λ²)·ε), với F, ε ~ Student-t(df) chuẩn hoá phương sai 1.

    ``B_coef[i, j]``: hệ số của cạnh i → j giữa các mã (None: không có nhân quả — mô hình nhân tố thuần, H0).
    Thành phần riêng √(1 − λ²)·ε giữ tương quan với nhân tố đúng bằng λ khi không có cạnh.
    """
    rng = rng if rng is not None else np.random.default_rng()
    lam = np.asarray(loadings, float)
    p = len(lam)
    if df <= 2:
        raise ValueError("df phải > 2 để chuẩn hoá phương sai.")
    scale = np.sqrt((df - 2) / df)
    F = rng.standard_t(df, n) * scale
    eps = rng.standard_t(df, (n, p)) * scale * np.sqrt(1 - lam ** 2)
    U = eps
    if B_coef is not None:
        Bc = np.asarray(B_coef, float)
        U = eps @ np.linalg.inv(np.eye(p) - Bc)       # hàng: u = ε + u·B  ⇒  u = ε (I − B)⁻¹
    return F[:, None] * lam + U


def factor_scm_truth(B_coef: np.ndarray | None, p: int) -> tuple[np.ndarray, list[int]]:
    """DAG đầy đủ (nút 0 = nhân tố ẩn F → mọi mã) và chỉ số nút quan sát, dùng cho SID."""
    A = np.zeros((p + 1, p + 1), dtype=int)
    A[0, 1:] = 1
    if B_coef is not None:
        A[1:, 1:] = (np.abs(np.asarray(B_coef)) > 0).astype(int)
    return A, list(range(1, p + 1))


def order_frequency(orders: Sequence[Sequence], names: Sequence) -> pd.DataFrame:
    """Bảng tần suất mỗi tên ở mỗi vị trí (1 = gốc)."""
    out = pd.DataFrame(0.0, index=list(names), columns=range(1, len(names) + 1))
    for o in orders:
        for pos, s in enumerate(o, 1):
            out.loc[s, pos] += 1 / len(orders)
    return out
