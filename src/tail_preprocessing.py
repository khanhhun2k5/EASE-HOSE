"""Tiền xử lý cho phân tích đuôi: lọc GARCH, ước lượng tham số hình dạng GPD, bootstrap ma trận hệ số.

Lý do (từ EDA): lợi suất ngày có hiệu ứng ARCH mạnh (phát hiện 5) nên các quan sát cực trị dồn theo cụm,
vi phạm giả định quan sát độc lập của Định lý 2 (Gnecco và cs., 2021). Chuẩn hoá theo biến động có điều
kiện z_t = (r_t − μ̂)/σ̂_t (GARCH(1,1), nhiễu Student-t) cho chuỗi gần độc lập cùng phân phối hơn.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

import causal_tail as CT


def garch_standardize(returns: pd.DataFrame, scale: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Khớp GARCH(1,1)–t cho từng cột (lợi suất log); trả (phần dư chuẩn hoá z, bảng tham số)."""
    from arch import arch_model

    z, params = {}, {}
    for col in returns.columns:
        r = returns[col].dropna() * scale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # chỉ ẩn cảnh báo số học, giữ cảnh báo hội tụ
            fit = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        if fit.convergence_flag != 0:
            warnings.warn(f"GARCH cho {col} không hội tụ (flag = {fit.convergence_flag}); kiểm tra lại phần dư.",
                          stacklevel=2)
        z[col] = fit.std_resid
        pr = fit.params
        # mu, omega đổi về đơn vị lợi suất gốc (omega là phương sai); alpha, beta, nu không có đơn vị.
        params[col] = {"mu": pr["mu"] / scale, "omega": pr["omega"] / scale ** 2, "alpha[1]": pr["alpha[1]"],
                       "beta[1]": pr["beta[1]"], "nu": pr["nu"],
                       "persistence": pr["alpha[1]"] + pr["beta[1]"], "loglik": fit.loglikelihood,
                       "converged": fit.convergence_flag == 0}
    return pd.DataFrame(z), pd.DataFrame(params).T


def gpd_tail_shape(x: np.ndarray, q: float = 0.95, upper: bool = True) -> dict:
    """Khớp GPD (MLE, vị trí 0) cho phần vượt ngưỡng phân vị q; trả ξ̂, SE ≈ (1 + ξ̂)/√n_u (Coles, 2001)
    và α̂ = 1/ξ̂ khi ξ̂ > 0 (SE = NaN khi ξ̂ ≤ −0,5). ``upper=False``: đuôi dưới (áp dụng cho −x)."""
    x = np.asarray(x, float)
    x = x if upper else -x
    u = np.quantile(x, q)
    if u >= x.max():
        raise ValueError(f"Ngưỡng phân vị {q} trùng giá trị lớn nhất: ≥ {1 - q:.0%} số quan sát dồn ở biên "
                         "(dữ liệu bị chặn, ví dụ biên độ giá) — GPD không áp dụng được.")
    exc = x[x > u] - u
    if len(exc) < 10:
        raise ValueError("Quá ít quan sát vượt ngưỡng để khớp GPD.")
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    # Công thức tiệm cận chỉ đúng khi ξ > −0,5 (MLE chính quy); ngoài vùng đó trả NaN.
    se = (1 + xi) / np.sqrt(len(exc)) if xi > -0.5 else np.nan
    # Tỷ lệ phần vượt bằng đúng giá trị lớn nhất: cao ⇒ dữ liệu bị chặn, ξ̂ chệch âm (đọc kèm khi diễn giải).
    tie_share = float(np.mean(exc == exc.max()))
    return {"xi": xi, "se": se, "beta": beta, "n_exc": len(exc), "tie_share_at_max": tie_share,
            "alpha": 1 / xi if xi > 0 else np.inf}


def bootstrap_coef(X: pd.DataFrame, k: int, kind: str = "psi", B: int = 200, seed: int = 0) -> np.ndarray:
    """Bootstrap i.i.d. theo dòng (giữ phụ thuộc đồng thời giữa các mã); trả mảng B × p × p."""
    rng = np.random.default_rng(seed)
    A = X.to_numpy(float)
    n = len(A)
    out = np.empty((B, A.shape[1], A.shape[1]))
    for b in range(B):
        # seed riêng mỗi lần lặp, cách nhau p để các luồng phá-thế-trùng (seed + j) của các lần lặp không chồng nhau
        out[b] = CT.tail_coef_matrix(A[rng.integers(0, n, n)], k=k, kind=kind, seed=seed * 7919 + b * A.shape[1])
    return out


def order_agreement(order_a, order_b) -> float:
    """Tỷ lệ cặp (i, j) được hai thứ tự xếp cùng chiều (= 1 − khoảng cách Kendall chuẩn hoá)."""
    pos_a = {v: i for i, v in enumerate(order_a)}
    pos_b = {v: i for i, v in enumerate(order_b)}
    items = list(pos_a)
    if len(order_a) != len(pos_a) or len(order_b) != len(pos_b) or set(items) != set(pos_b):
        raise ValueError("Hai thứ tự phải là hoán vị của cùng một tập phần tử (không trùng lặp).")
    if len(items) < 2:
        return 1.0
    pairs = [(u, v) for i, u in enumerate(items) for v in items[i + 1:]]
    return float(np.mean([(pos_a[u] < pos_a[v]) == (pos_b[u] < pos_b[v]) for u, v in pairs]))
