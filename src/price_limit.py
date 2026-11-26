"""Xử lý biên độ giá ±7% (HOSE) theo mô hình có kiểm duyệt của Xu, Shu & Chen (2024),
"Tail Risk Dynamics under Price-Limited Constraint: A Censored Autoregressive Conditional Fréchet Model",
Entropy 26:555, doi:10.3390/e26070555.

Bối cảnh: phiên chạm sàn/trần khiến lợi suất quan sát bị kiểm duyệt (đúng bằng ngưỡng), trong khi lợi suất
tiềm ẩn (latent) có thể lớn hơn. Bài báo xử lý bài toán này cho cực đại chéo Q_t = max_i(−r_it); ở đây áp dụng
(A) nguyên mô hình cho Q_t và (B) nguyên tắc likelihood kiểm duyệt cho từng mã, rồi (C) dùng nhiều lần điền
(multiple imputation) để chạy EASE trên phần dư đã bù kiểm duyệt, (D) mô phỏng dữ liệu bị chặn để kiểm chứng.

A. CAcF cho Q_t (mục 2.1–2.4 của bài báo)
    Q*_t = μ + σ_t·E_t^(−1/α_t), E_t ~ Exp(1) (bản PDF ghi Y^(−1/α) với Y Fréchet đơn vị — không nhất quán với (5));
    Q_t = D_t·M + (1 − D_t)·Q*_t, D_t = 1{Q*_t > M}   (1)–(2)
    log σ_t = β0 + β1·log σ_{t−1} + (β2 + β2*·D_{t−1})·G1(Q_{t−1})
    log α_t = γ0 + γ1·log α_{t−1} − (γ2 + γ2*·D_{t−1})·G2(Q_{t−1})
    G: "L" (tuyến tính) G(z) = z; "S" (bình phương) G(z) = sgn(z)·z²; "E" (mũ) G1 = −exp(−β3·z), G2 = −exp(−γ3·z).
    Hàm CDF nhất quán với likelihood (5): P(Q* ≤ q) = exp(−((q − μ)/σ_t)^(−α_t)), q > μ.
    Log-likelihood kiểm duyệt (5), M_t là ngưỡng của phiên t (ở đây M_t = Q_t khi D_t = 1):
        D_t·log(1 − exp(−σ_t^α (M_t − μ)^(−α))) + (1 − D_t)·[log α + α·log σ − (α + 1)·log(Q_t − μ) − σ_t^α (Q_t − μ)^(−α)]
    Giá trị khởi đầu (σ1, α1) lấy từ Fréchet tĩnh có kiểm duyệt (Remark 6). Xác suất bị chặn một bước
    P̂_t = 1 − exp(−σ_t^α (M − μ)^(−α)) (mục 2.4); EVaR¹ không kiểm duyệt
    μ + σ_t/(1 − τ)·γ(1 − 1/α_t, −log τ), γ là hàm gamma bất toàn dưới KHÔNG chuẩn hoá.
    AcF (Zhao và cs.) = CAcF với M = ∞: bỏ qua kiểm duyệt, coi mọi D = 0.

B. GARCH(1,1)–t có kiểm duyệt cho từng mã — PHẦN MỞ RỘNG của đồ án, không phải mô hình trong bài báo: lấy cơ chế
   kiểm duyệt (3), nguyên tắc likelihood kiểm duyệt (5) và số hạng bù chạm biên (Remark 3) áp cho từng chuỗi lợi suất
    r*_t = μ + σ_t·ε_t, ε ~ Student-t chuẩn hoá (phương sai 1); r_t = clip(r*_t, −c, c)
    σ²_t = ω + a·(r_{t−1} − μ)² + b·σ²_{t−1} + δ⁻·F_{t−1} + δ⁺·C_{t−1}   (F/C: cờ chạm sàn/trần)
    Phiên thường dùng log f; phiên sàn dùng log F_ε((c_t − μ)/σ_t); phiên trần dùng log(1 − F_ε((c_t − μ)/σ_t)).

Ghi chú cài đặt: numba chưa chạy được với numpy 2.5 trong môi trường này, nên các đệ quy (σ_t, α_t của CAcF và σ²_t
của GARCH) đều tuyến tính theo đầu vào đã biết nên dùng ``scipy.signal.lfilter`` (vectorized, không cần vòng lặp Python).
"""

from __future__ import annotations

import itertools
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special
from scipy.signal import lfilter

import causal_tail as CT
import tail_preprocessing as TP

PENALTY = 1e10          # giá trị hàm mục tiêu hữu hạn thay cho NaN/inf khi tham số rơi vào vùng suy biến
KINDS = ("L", "S", "E")
SIM_METHODS = ("Không chặn (GARCH)", "GARCH thường", "GARCH kiểm duyệt", "Kiểm duyệt + điền nhiều lần")


# ============================================================================ tiện ích chung
def _logistic(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, float)))


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _minimize(fun: Callable, starts: Sequence[np.ndarray], bounds: Sequence[tuple], maxiter: int = 400,
              rounds: int = 15, grad_tol: float = 0.05) -> tuple:
    """L-BFGS-B từ nhiều điểm khởi đầu, chọn nghiệm tốt nhất, rồi luân phiên Nelder–Mead ↔ L-BFGS-B (tối đa ``rounds``
    vòng) tới khi tổng −log-likelihood giảm < 1e-6.

    Trả (x, f, converged). ``converged`` = gradient chiếu tại x nhỏ: max |∂f/∂x_i| < ``grad_tol`` trên các toạ độ không
    chạm biên (x là tham số đã biến đổi log/logit, nên 0,05 ⇔ đổi tham số 0,01 làm f đổi < 5·10⁻⁴). Không dựa vào cờ
    ``success``: L-BFGS-B với đạo hàm số hay báo "ABNORMAL_TERMINATION" ở cực trị, còn Nelder–Mead có thể dừng vì
    maxiter khi chưa tới cực trị."""
    bounds = list(bounds)
    best = None
    for x0 in starts:
        res = optimize.minimize(fun, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return np.asarray(starts[0], float), PENALTY, False
    x, f = np.asarray(best.x, float), float(best.fun)
    for _ in range(rounds):                      # luân phiên Nelder–Mead ↔ L-BFGS-B tới khi không còn cải thiện
        f_before = f
        nm = optimize.minimize(fun, x, method="Nelder-Mead", bounds=bounds,
                               options={"maxiter": 400 * len(x), "xatol": 1e-7, "fatol": 1e-10, "adaptive": True})
        if nm.fun < f:
            x, f = np.asarray(nm.x, float), float(nm.fun)
        lb = optimize.minimize(fun, x, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        if lb.fun < f:
            x, f = np.asarray(lb.x, float), float(lb.fun)
        if f_before - f < 1e-6:
            break
    converged = bool(f < PENALTY and _projected_grad_max(fun, x, bounds) < grad_tol)
    return x, f, converged


def _projected_grad_max(fun: Callable, x: np.ndarray, bounds: Sequence[tuple], eps: float = 1e-6) -> float:
    """Độ lớn gradient CHIẾU (điều kiện KKT với ràng buộc hộp), sai phân số — tiêu chí hội tụ trực tiếp, không phụ thuộc
    vào cờ ``success`` của từng bộ tối ưu. Toạ độ trong miền: |g_i|. Toạ độ ở cận dưới: chỉ tính phần g_i < 0 (giảm f bằng
    cách đi VÀO miền); ở cận trên: chỉ tính g_i > 0. Ở biên dùng sai phân một phía hướng vào trong."""
    g = np.zeros(len(x))
    for i in range(len(x)):
        lo, hi = bounds[i]
        e = np.zeros(len(x))
        e[i] = eps
        if x[i] - lo < 1e-4:                            # cận dưới: đạo hàm theo hướng vào trong (+)
            d = (fun(x + e) - fun(x)) / eps
            g[i] = min(d, 0.0)
        elif hi - x[i] < 1e-4:                          # cận trên: đạo hàm theo hướng vào trong (−)
            d = (fun(x) - fun(x - e)) / eps
            g[i] = max(d, 0.0)
        else:
            g[i] = (fun(x + e) - fun(x - e)) / (2 * eps)
    return float(np.max(np.abs(g))) if len(g) else 0.0


def _reject_nan_flags(v, name: str) -> None:
    """Cờ chạm biên không được có NaN/None ở bất kỳ kiểu dữ liệu nào (astype(bool) sẽ biến NaN thành True)."""
    if pd.isna(np.asarray(v, dtype=object)).any():
        raise ValueError(f"{name} có NaN/None — điền False (không chạm biên) trước khi gọi.")


def _as_vector(x, name: str) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 1:
        raise ValueError(f"{name} phải là mảng 1 chiều.")
    return a


# ============================================================================ A. CAcF cho Q_t
def qt_series(returns: pd.DataFrame, floor: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Cực đại chéo của lợi suất âm Q_t = max_i(−r_it) và cờ kiểm duyệt.

    ``returns``: lợi suất log n × p; ``floor``: cờ chạm sàn cùng kích thước (không NaN). Cột trả về: ``Q``, ``D``
    (1 khi CÓ ÍT NHẤT MỘT mã chạm sàn: lỗ ẩn của mã đó có thể vượt Q_t nên chỉ biết Q*_t ≥ Q_t — kể cả khi mã đạt cực đại
    quan sát không chạm sàn, ví dụ sàn bị làm tròn bước giá thấp hơn mức lỗ của mã khác), ``M`` (= Q_t, cận dưới của Q*_t
    khi D = 1; NaN khi D = 0). Coi phiên đó là kiểm duyệt tại Q_t là xấp xỉ bảo thủ: bỏ thông tin "giá trị đúng tại mã
    không chạm sàn" nhưng không bao giờ coi một Q_t bị chặn là quan sát đầy đủ.
    ``argmax`` (tên cột của mã đạt cực đại), ``n_floor`` (số mã chạm sàn trong phiên — thông tin thêm).
    Chấp nhận phiên có một phần NaN (bỏ qua mã thiếu); phiên toàn NaN thì báo lỗi.
    """
    R = returns.to_numpy(float) if isinstance(returns, pd.DataFrame) else np.asarray(returns, float)
    Fl = floor.to_numpy() if isinstance(floor, pd.DataFrame) else np.asarray(floor)
    if R.ndim != 2 or R.shape != Fl.shape:
        raise ValueError(f"returns và floor phải cùng kích thước n × p, nhận {R.shape} và {Fl.shape}.")
    if isinstance(returns, pd.DataFrame) and isinstance(floor, pd.DataFrame) and not (
            returns.index.equals(floor.index) and returns.columns.equals(floor.columns)):
        raise ValueError("returns và floor phải cùng chỉ số dòng/cột.")
    if R.shape[0] == 0 or np.isnan(R).all(axis=1).any():
        raise ValueError("Có phiên toàn NaN (hoặc không có dữ liệu): bỏ các dòng này trước khi tính Q_t.")
    _reject_nan_flags(Fl, "floor")
    Fl = np.where(np.isnan(R), False, Fl.astype(bool))
    neg = np.where(np.isnan(R), -np.inf, -R)
    j = neg.argmax(axis=1)
    rows = np.arange(R.shape[0])
    Q = neg[rows, j]
    D = Fl.any(axis=1).astype(int)
    cols = returns.columns if isinstance(returns, pd.DataFrame) else pd.RangeIndex(R.shape[1])
    idx = returns.index if isinstance(returns, pd.DataFrame) else pd.RangeIndex(R.shape[0])
    return pd.DataFrame({"Q": Q, "D": D, "M": np.where(D == 1, Q, np.nan), "argmax": np.asarray(cols)[j],
                         "n_floor": Fl.sum(axis=1)}, index=idx)


def observation_driven(kind: str, z, c: float | None = None) -> np.ndarray:
    """Hàm dẫn động bởi quan sát G(z): "L" → z; "S" → sgn(z)·z²; "E" → −exp(−c·z) (c = β3 hoặc γ3 > 0)."""
    z = np.asarray(z, float)
    if kind == "L":
        return z
    if kind == "S":
        return np.sign(z) * z * z
    if kind == "E":
        if c is None or not c > 0:
            raise ValueError("Dạng 'E' cần tham số c = β3 (hoặc γ3) > 0.")
        return -np.exp(np.minimum(-c * z, 50.0))
    raise ValueError(f"kind phải thuộc {KINDS}, nhận {kind!r}.")


def _check_qdm(Q, D, M=None, min_n: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = _as_vector(Q, "Q")
    D = _as_vector(D, "D")
    if len(Q) != len(D):
        raise ValueError(f"Q và D phải cùng độ dài, nhận {len(Q)} và {len(D)}.")
    if len(Q) < min_n:
        raise ValueError(f"Cần ít nhất {min_n} quan sát, nhận {len(Q)}.")
    if not np.isfinite(Q).all() or not np.isfinite(D).all():
        raise ValueError("Q hoặc D có NaN/inf — bỏ các phiên thiếu trước.")
    if not np.isin(D, (0, 1)).all():
        raise ValueError("D chỉ nhận giá trị 0 hoặc 1.")
    if M is None:
        M = Q.copy()
    else:
        M = _as_vector(M, "M") if np.ndim(M) else np.full(len(Q), float(M))
        if len(M) != len(Q):
            raise ValueError("M phải cùng độ dài với Q (hoặc là một số).")
        M = np.where(D == 1, M, Q)                    # M chỉ dùng ở phiên bị kiểm duyệt
        if not np.isfinite(M).all():
            raise ValueError("M có NaN/inf ở phiên bị kiểm duyệt.")
    return Q, D, M


def _frechet_terms(mu, sigma, alpha, Q, D, M) -> np.ndarray:
    """Đóng góp của từng phiên vào log-likelihood (5); tính trong không gian log để ổn định số học.

    Yêu cầu (Q − μ) > 0 và (M − μ) > 0 (người gọi bảo đảm)."""
    x = np.where(D > 0.5, M, Q) - mu
    lx = np.log(x)
    ls = np.log(sigma)
    e = alpha * (ls - lx)                              # α·log(σ/(Q − μ))
    w = np.exp(np.minimum(e, 700.0))                   # σ^α (Q − μ)^(−α)
    cens = np.log(np.maximum(-np.expm1(-w), 1e-300))   # log(1 − e^(−w)) = log(−expm1(−w))
    unc = np.log(alpha) + e - lx - w                   # log α + α log σ − (α + 1) log(Q − μ) − w
    return np.where(D > 0.5, cens, unc)


def fit_static_censored_frechet(Q, D, M=None) -> dict:
    """Fréchet tĩnh (μ, σ, α) có kiểm duyệt bằng MLE của (5) với σ, α hằng — giá trị khởi đầu cho CAcF (Remark 6).

    Trả dict: mu, sigma, alpha, loglik (tổng, không chia n), converged."""
    Q, D, M = _check_qdm(Q, D, M)
    qmin, sd = float(Q.min()), float(Q.std()) or 1.0

    def unpack(x):
        return qmin - np.exp(x[0]), np.exp(x[1]), np.exp(x[2])

    def nll(x):
        mu, sigma, alpha = unpack(x)
        v = _frechet_terms(mu, sigma, alpha, Q, D, M)
        s = -v.sum()
        return float(s) if np.isfinite(s) else PENALTY

    starts = []
    for a0 in (1.5, 3.0, 6.0):
        med = float(np.median(Q - (qmin - 0.3 * sd)))
        starts.append(np.array([np.log(0.3 * sd), np.log(max(med, 1e-6) * np.log(2) ** (1 / a0)), np.log(a0)]))
    bounds = [(np.log(1e-4 * sd), np.log(30 * sd)), (np.log(1e-6 * sd), np.log(1e3 * sd)), (np.log(0.1), np.log(200.0))]
    x, f, ok = _minimize(nll, starts, bounds)
    mu, sigma, alpha = unpack(x)
    return {"mu": float(mu), "sigma": float(sigma), "alpha": float(alpha), "loglik": float(-f), "converged": ok}


def cacf_filter(theta: dict, Q, D, kind: str = "L", init: tuple[float, float] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Đệ quy (σ_t, α_t) của (2) bằng lfilter (AR(1) trên log). σ_t, α_t chỉ phụ thuộc Q_{t−1}, D_{t−1}.

    ``theta``: dict có beta0, beta1, beta2, [beta2s], [beta3], gamma0, gamma1, gamma2, [gamma2s], [gamma3]
    (beta2s/gamma2s vắng = 0). ``init`` = (σ1, α1); mặc định là trung bình dừng exp(β0/(1−β1)), exp(γ0/(1−γ1)).
    Trả hai mảng độ dài n (đơn vị của Q)."""
    Q = _as_vector(Q, "Q")
    D = _as_vector(D, "D")
    if len(Q) != len(D):
        raise ValueError("Q và D phải cùng độ dài.")
    if kind not in KINDS:
        raise ValueError(f"kind phải thuộc {KINDS}, nhận {kind!r}.")
    for key in ("beta0", "beta1", "beta2", "gamma0", "gamma1", "gamma2"):
        if key not in theta:
            raise ValueError(f"theta thiếu tham số {key!r}.")
    if kind == "E" and ("beta3" not in theta or "gamma3" not in theta):
        raise ValueError("Dạng 'E' cần beta3 và gamma3 trong theta.")
    b1, g1 = float(theta["beta1"]), float(theta["gamma1"])
    if not (0 <= b1 < 1 and 0 <= g1 < 1):
        raise ValueError("Cần 0 ≤ β1, γ1 < 1 (ràng buộc dừng).")
    if init is None:
        ls1, la1 = theta["beta0"] / (1 - b1), theta["gamma0"] / (1 - g1)
    else:
        ls1, la1 = np.log(init[0]), np.log(init[1])
    return _clip_state(*_cacf_log_filter(theta, Q, D, kind, ls1, la1))


def _clip_state(log_sigma, log_alpha):
    """(σ, α) từ trạng thái log. Chỉ cắt GIÁ TRỊ DÙNG (tránh tràn số), không cắt trạng thái đệ quy — cùng một quy tắc ở
    cacf_filter, simulate_cacf và predict_cacf."""
    return np.exp(np.clip(log_sigma, -60, 60)), np.exp(np.clip(log_alpha, np.log(0.05), np.log(300.0)))


def _cacf_log_filter(theta: dict, Q, D, kind: str, ls1: float, la1: float) -> tuple[np.ndarray, np.ndarray]:
    """Trạng thái (log σ_t, log α_t) KHÔNG cắt của (2), bắt đầu từ (ls1, la1)."""
    b1, g1 = float(theta["beta1"]), float(theta["gamma1"])
    u1 = (theta["beta2"] + theta.get("beta2s", 0.0) * D) * observation_driven(kind, Q, theta.get("beta3"))
    u2 = (theta["gamma2"] + theta.get("gamma2s", 0.0) * D) * observation_driven(kind, Q, theta.get("gamma3"))
    x1 = np.empty(len(Q))
    x2 = np.empty(len(Q))
    x1[0], x2[0] = ls1, la1                                   # trạng thái ban đầu (y_0 = x_0)
    x1[1:] = theta["beta0"] + u1[:-1]
    x2[1:] = theta["gamma0"] - u2[:-1]
    return lfilter([1.0], [1.0, -b1], x1), lfilter([1.0], [1.0, -g1], x2)


def cacf_loglik(theta: dict, Q, D, M=None, kind: str = "L", init: tuple[float, float] | None = None,
                censored: bool = True) -> float:
    """Tổng log-likelihood (5) tại ``theta`` (cần có ``mu``). ``censored=False``: coi mọi D = 0 (AcF).
    Báo ValueError nếu μ ≥ min(Q)."""
    Q, D, M = _check_qdm(Q, D, M, min_n=2)
    mu = float(theta["mu"])
    if not mu < Q.min():
        raise ValueError(f"Cần μ < min(Q) = {Q.min():.6g}, nhận μ = {mu:.6g} (Q*_t > μ với mọi t).")
    D_eff = D if censored else np.zeros_like(D)
    sigma, alpha = cacf_filter(theta, Q, D_eff, kind, init)
    return float(_frechet_terms(mu, sigma, alpha, Q, D_eff, M).sum())


def _cacf_names(kind: str, censored: bool, compensation: bool = True) -> list[str]:
    comp = censored and compensation
    nb = ["beta0", "beta1", "beta2"] + (["beta2s"] if comp else []) + (["beta3"] if kind == "E" else [])
    ng = ["gamma0", "gamma1", "gamma2"] + (["gamma2s"] if comp else []) + (["gamma3"] if kind == "E" else [])
    return ["mu"] + nb + ng


_UNIT_PARAMS = {"beta1", "gamma1"}                          # (0, 1): biến đổi logistic
_POSITIVE_PARAMS = {"beta2", "gamma2", "beta3", "gamma3"}   # > 0: biến đổi exp


@dataclass
class CacfFit:
    """Kết quả khớp CAcF/AcF. ``params`` ở ĐƠN VỊ LÀM VIỆC (Q × scale); ``sigma_t``, ``alpha_t`` là dự báo một bước
    trong mẫu; ``sigma_t`` quy về đơn vị gốc của Q (chia ``scale``)."""
    params: dict
    loglik: float
    aic: float
    bic: float
    sigma_t: pd.Series
    alpha_t: pd.Series
    kind: str
    censored: bool
    converged: bool
    scale: float = 1.0
    init: tuple[float, float] = (np.nan, np.nan)             # (σ1, α1) ở đơn vị làm việc
    state: dict = field(default_factory=dict)                # trạng thái cuối mẫu cho predict_cacf
    n: int = 0

    @property
    def mu(self) -> float:
        """Tham số vị trí μ theo đơn vị gốc của Q."""
        return self.params["mu"] / self.scale


def fit_cacf(Q, D, M=None, kind: str = "L", censored: bool = True, n_starts: int = 3, seed: int = 0,
             scale: float = 100.0, compensation: bool = True) -> CacfFit:
    """Khớp CAcF-L/S/E (``censored=True``) hoặc AcF (``censored=False``: bỏ kiểm duyệt, bỏ β2*, γ2*) bằng MLE (5).

    ``compensation=False`` (chỉ có nghĩa khi censored=True): vẫn dùng likelihood kiểm duyệt nhưng ràng buộc β2* = γ2* = 0 —
    mô hình rút gọn cho kiểm định tỷ số hợp lý LR = 2(ℓ_đầy đủ − ℓ_rút gọn) ~ χ²(2) của số hạng bù ngưỡng (Remark 3).

    Q, D, M theo đơn vị gốc (Q ≥ 0 cỡ vài %); bên trong nhân ``scale`` (mặc định 100 → đơn vị %). Tham số hoá: μ = min(Q) −
    exp(·) (đảm bảo Q_t > μ), logit cho β1, γ1 ∈ (0, 1), log cho các tham số dương. Tối ưu L-BFGS-B nhiều điểm
    khởi đầu + Nelder–Mead tinh chỉnh; hàm mục tiêu trả phạt hữu hạn khi suy biến. (σ1, α1) lấy từ Fréchet tĩnh (Remark 6)."""
    if kind not in KINDS:
        raise ValueError(f"kind phải thuộc {KINDS}, nhận {kind!r}.")
    if not scale > 0:
        raise ValueError("scale phải dương.")
    index = Q.index if isinstance(Q, pd.Series) else None
    for nm_, v_ in (("D", D), ("M", M)):
        if isinstance(v_, pd.Series) and (index is None or not v_.index.equals(index)):
            raise ValueError(f"Q và {nm_} phải cùng chỉ số (hoặc truyền dạng mảng).")
    Q, D, M = _check_qdm(Q, D, M)
    Qs, Ms = Q * scale, M * scale
    D_eff = D if censored else np.zeros_like(D)
    qmin, sd = float(Qs.min()), float(Qs.std()) or 1.0
    static = fit_static_censored_frechet(Qs, D_eff, Ms)
    init = (static["sigma"], static["alpha"])
    names = _cacf_names(kind, censored, compensation)
    mean_g = {"L": float(Qs.mean()), "S": float(np.mean(np.sign(Qs) * Qs ** 2)), "E": -0.5}

    def to_theta(phi) -> dict:
        th = {}
        for nm, x in zip(names, phi):
            if nm == "mu":
                th[nm] = qmin - np.exp(x)
            elif nm in _UNIT_PARAMS:
                th[nm] = float(_logistic(x))
            elif nm in _POSITIVE_PARAMS:
                th[nm] = float(np.exp(x))
            else:
                th[nm] = float(x)
        return th

    def nll(phi):
        th = to_theta(phi)
        s, a = cacf_filter(th, Qs, D_eff, kind, init)
        v = _frechet_terms(th["mu"], s, a, Qs, D_eff, Ms)
        tot = -v.sum()
        return float(tot) if np.isfinite(tot) else PENALTY

    presets = [(0.8, 0.7, 0.05, 0.05), (0.5, 0.5, 0.15, 0.15), (0.9, 0.9, 0.02, 0.02), (0.3, 0.6, 0.1, 0.02)]
    rng = np.random.default_rng(seed)
    m0 = np.log(max(qmin - static["mu"], 1e-6 * sd))
    starts = []
    for i in range(max(1, n_starts)):
        b1, g1, b2, g2 = presets[i % len(presets)]
        phi = []
        for nm in names:
            if nm == "mu":
                phi.append(m0)
            elif nm == "beta0":
                phi.append((1 - b1) * np.log(static["sigma"]) - b2 * mean_g[kind])
            elif nm == "gamma0":
                phi.append((1 - g1) * np.log(static["alpha"]) + g2 * mean_g[kind])
            elif nm == "beta1":
                phi.append(_logit(b1))
            elif nm == "gamma1":
                phi.append(_logit(g1))
            elif nm == "beta2":
                phi.append(np.log(b2))
            elif nm == "gamma2":
                phi.append(np.log(g2))
            elif nm in ("beta3", "gamma3"):
                phi.append(np.log(0.3))
            else:                                           # beta2s, gamma2s
                phi.append(0.0)
        phi = np.array(phi, float)
        starts.append(phi if i == 0 else phi + rng.normal(0, 0.05, len(phi)))
    bounds = []
    for nm in names:
        if nm == "mu":
            bounds.append((np.log(1e-3 * sd), np.log(30 * sd)))
        elif nm in _UNIT_PARAMS:
            bounds.append((-6.0, 6.0))
        elif nm in ("beta3", "gamma3"):
            bounds.append((-6.0, 2.0))
        elif nm in _POSITIVE_PARAMS:
            bounds.append((-8.0, 3.0))
        elif nm in ("beta2s", "gamma2s"):
            bounds.append((-5.0, 5.0))
        else:
            bounds.append((-30.0, 30.0))
    phi, f, ok = _minimize(nll, starts, bounds)
    theta = to_theta(phi)
    ls_t, la_t = _cacf_log_filter(theta, Qs, D_eff, kind, np.log(init[0]), np.log(init[1]))
    sig, alp = _clip_state(ls_t, la_t)
    k = len(names)
    n = len(Q)
    ll = float(-f)
    if not ok:
        warnings.warn("CAcF/AcF không hội tụ chắc chắn; kiểm tra lại tham số và điểm khởi đầu.", stacklevel=2)
    idx = index if index is not None else pd.RangeIndex(n)
    state = {"Q": float(Qs[-1]), "D": float(D_eff[-1]), "sigma": float(sig[-1]), "alpha": float(alp[-1]),
             "log_sigma": float(ls_t[-1]), "log_alpha": float(la_t[-1])}      # trạng thái log không cắt
    return CacfFit(params=theta, loglik=ll, aic=-2 * ll + 2 * k, bic=-2 * ll + k * np.log(n),
                   sigma_t=pd.Series(sig / scale, index=idx, name="sigma"),
                   alpha_t=pd.Series(alp, index=idx, name="alpha"), kind=kind, censored=censored,
                   converged=ok, scale=float(scale), init=init, state=state, n=n)


def _censored_probability_arrays(mu, sigma, alpha, M) -> np.ndarray:
    sigma, alpha = np.asarray(sigma, float), np.asarray(alpha, float)
    M = np.asarray(M, float)
    gap = M - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.exp(np.minimum(alpha * (np.log(sigma) - np.log(np.where(gap > 0, gap, 1.0))), 700.0))
    return np.where(gap > 0, -np.expm1(-w), 1.0)          # M ≤ μ ⇒ Q* > M chắc chắn


def censored_probability(fit: CacfFit, M) -> pd.Series:
    """P̂_t = P(Q*_t > M | F_{t−1}) = 1 − exp(−σ_t^α (M − μ)^(−α)): xác suất một bước tới bị kiểm duyệt (mục 2.4).

    ``M`` theo đơn vị gốc của Q: số (ví dụ trung vị ngưỡng) hoặc mảng cùng độ dài mẫu."""
    M = np.asarray(M, float)
    if M.ndim and M.shape != (fit.n,):
        raise ValueError(f"M phải là số hoặc mảng độ dài {fit.n}.")
    p = _censored_probability_arrays(fit.params["mu"], fit.sigma_t.to_numpy() * fit.scale, fit.alpha_t.to_numpy(),
                                     M * fit.scale)
    return pd.Series(p, index=fit.sigma_t.index, name="p_censored")


def evar_frechet(mu, sigma, alpha, tau: float) -> np.ndarray:
    """EVaR¹ (= CVaR) không kiểm duyệt của Q* ~ μ + σ·Fréchet(α): μ + σ/(1 − τ)·γ(1 − 1/α, −log τ), γ KHÔNG chuẩn hoá
    (= gammainc·Γ). NaN khi α ≤ 1 (kỳ vọng không tồn tại)."""
    if not 0 < tau < 1:
        raise ValueError("τ phải thuộc (0, 1).")
    alpha = np.asarray(alpha, float)
    a = 1.0 - 1.0 / alpha
    ok = alpha > 1.0
    a_safe = np.where(ok, a, 0.5)
    lower = special.gammainc(a_safe, -np.log(tau)) * special.gamma(a_safe)
    out = mu + np.asarray(sigma, float) / (1.0 - tau) * lower
    return np.where(ok, out, np.nan)


def evar_censored_frechet(mu, sigma, alpha, tau: float, M, n_grid: int = 2000) -> np.ndarray:
    """EVaR¹ của Q = min(Q*, M): (1/(1 − τ))·∫_τ^1 min(VaR_u, M) du, VaR_u = μ + σ(−log u)^(−1/α) (tích phân điểm giữa
    trên lưới ``n_grid``; luôn hữu hạn kể cả α ≤ 1 vì bị chặn bởi M)."""
    if not 0 < tau < 1:
        raise ValueError("τ phải thuộc (0, 1).")
    u = tau + (1 - tau) * (np.arange(n_grid) + 0.5) / n_grid
    h = (-np.log(u))[None, :]                                 # (1, G)
    sigma, alpha, M = (np.atleast_1d(np.asarray(v, float)) for v in np.broadcast_arrays(sigma, alpha, M))
    # σ·(−log u)^(−1/α): dùng exp(−log h/α) để tránh luỹ thừa biên
    var_u = mu + sigma[:, None] * np.exp(-np.log(h) / alpha[:, None])
    return np.minimum(var_u, M[:, None]).mean(axis=1)


def evar(fit: CacfFit, tau: float) -> pd.Series:
    """EVaR¹ không kiểm duyệt theo từng phiên của mẫu (đơn vị gốc của Q); NaN khi α_t ≤ 1."""
    out = evar_frechet(fit.mu, fit.sigma_t.to_numpy(), fit.alpha_t.to_numpy(), tau)
    return pd.Series(out, index=fit.sigma_t.index, name=f"evar_{tau:g}")


def evar_censored(fit: CacfFit, tau: float, M, n_grid: int = 2000) -> pd.Series:
    """EVaR¹ của Q kiểm duyệt tại ngưỡng ``M`` (số hoặc mảng), đơn vị gốc của Q."""
    M = np.broadcast_to(np.asarray(M, float), (fit.n,))
    out = evar_censored_frechet(fit.mu, fit.sigma_t.to_numpy(), fit.alpha_t.to_numpy(), tau, M, n_grid)
    return pd.Series(out, index=fit.sigma_t.index, name=f"evar_cens_{tau:g}")


def predict_cacf(fit: CacfFit, Q_new, D_new, M_new) -> pd.DataFrame:
    """Dự báo một bước NGOÀI MẪU với tham số cố định: lọc tiếp từ trạng thái cuối mẫu.

    ``Q_new``, ``D_new``: giá trị thực hiện ở các phiên mới (đơn vị gốc); (σ_t, α_t) của phiên mới thứ j chỉ dùng thông
    tin đến phiên j − 1. ``M_new``: ngưỡng của từng phiên mới (số hoặc mảng). Trả DataFrame: sigma, alpha, p_censored."""
    idx = Q_new.index if isinstance(Q_new, pd.Series) else None
    Q_new = _as_vector(Q_new, "Q_new")
    D_new = _as_vector(D_new, "D_new")
    if len(Q_new) != len(D_new) or len(Q_new) == 0:
        raise ValueError("Q_new và D_new phải cùng độ dài và không rỗng.")
    if not (np.isfinite(Q_new).all() and np.isfinite(D_new).all() and np.isin(D_new, (0, 1)).all()):
        raise ValueError("Q_new/D_new có NaN/inf hoặc D ngoài {0, 1}.")
    M_new = np.broadcast_to(np.asarray(M_new, float), Q_new.shape)
    st = fit.state
    D_ext = np.concatenate([[st["D"]], D_new if fit.censored else np.zeros_like(D_new)])
    Q_ext = np.concatenate([[st["Q"]], Q_new * fit.scale])
    ls, la = _cacf_log_filter(fit.params, Q_ext, D_ext, fit.kind,
                              st.get("log_sigma", np.log(st["sigma"])), st.get("log_alpha", np.log(st["alpha"])))
    s, a = _clip_state(ls, la)
    s, a = s[1:len(Q_new) + 1], a[1:len(Q_new) + 1]            # s[0] = trạng thái cuối mẫu (đã biết)
    p = _censored_probability_arrays(fit.params["mu"], s, a, M_new * fit.scale)
    return pd.DataFrame({"sigma": s / fit.scale, "alpha": a, "p_censored": p},
                        index=idx if idx is not None else pd.RangeIndex(len(Q_new)))


def simulate_cacf(n: int, params: dict, M, kind: str = "L", rng: np.random.Generator | None = None,
                  init: tuple[float, float] | None = None) -> dict:
    """Mô phỏng CAcF (2) với tham số ``params`` (cần ``mu`` và các tham số của cacf_filter), ngưỡng ``M`` (số hoặc mảng).

    Trả dict: Q (đã kiểm duyệt), D, Q_latent (Q*), sigma, alpha. Mọi đại lượng cùng đơn vị với ``params``/``M``."""
    if n < 2:
        raise ValueError("n phải ≥ 2.")
    if kind not in KINDS:
        raise ValueError(f"kind phải thuộc {KINDS}, nhận {kind!r}.")
    rng = rng if rng is not None else np.random.default_rng()
    M = np.broadcast_to(np.asarray(M, float), (n,))
    b1, g1 = params["beta1"], params["gamma1"]
    if not (0 <= b1 < 1 and 0 <= g1 < 1):
        raise ValueError("Cần 0 ≤ β1, γ1 < 1.")
    ls = params["beta0"] / (1 - b1) if init is None else np.log(init[0])
    la = params["gamma0"] / (1 - g1) if init is None else np.log(init[1])
    e = -np.log(rng.random(n))                 # E ~ Exp(1); Fréchet(α) đơn vị: Y = E^(−1/α) (CDF exp(−y^(−α)))
    sigma, alpha = np.empty(n), np.empty(n)
    Q, D, Qlat = np.empty(n), np.zeros(n, int), np.empty(n)
    for t in range(n):
        sigma[t], alpha[t] = (float(v) for v in _clip_state(ls, la))
        Qlat[t] = params["mu"] + sigma[t] * e[t] ** (-1.0 / alpha[t])
        D[t] = int(Qlat[t] > M[t])
        Q[t] = M[t] if D[t] else Qlat[t]
        g1v = float(observation_driven(kind, Q[t], params.get("beta3")))
        g2v = float(observation_driven(kind, Q[t], params.get("gamma3")))
        ls = params["beta0"] + b1 * ls + (params["beta2"] + params.get("beta2s", 0.0) * D[t]) * g1v
        la = params["gamma0"] + g1 * la - (params["gamma2"] + params.get("gamma2s", 0.0) * D[t]) * g2v
    return {"Q": Q, "D": D, "Q_latent": Qlat, "sigma": sigma, "alpha": alpha}


# ============================================================================ B. GARCH(1,1)–t có kiểm duyệt
def _garch_variance(r, mu, omega, a, b, dF, dC, F, C, s2_0, s2_floor) -> np.ndarray:
    """σ²_t = ω + a·(r_{t−1} − μ)² + b·σ²_{t−1} + δ⁻F_{t−1} + δ⁺C_{t−1} bằng lfilter; σ²_1 = s2_0; sàn dưới s2_floor."""
    u = np.empty(len(r))
    u[0] = s2_0
    e = r[:-1] - mu
    u[1:] = omega + a * e * e + dF * F[:-1] + dC * C[:-1]
    return np.maximum(lfilter([1.0], [1.0, -b], u), s2_floor)


def _as_flags(x, n: int, name: str) -> np.ndarray:
    v = np.asarray(x)
    if v.ndim != 1 or len(v) != n:
        raise ValueError(f"{name} phải là mảng 1 chiều độ dài {n}.")
    if v.dtype != bool:
        _reject_nan_flags(v, name)
        v = v.astype(bool)
    return v


def fit_censored_garch(r, floor, ceil, censored: bool = True, limit_terms: bool = True,
                       scale: float | None = 100.0, n_starts: int = 2) -> dict:
    """Khớp GARCH(1,1)–t có kiểm duyệt cho MỘT mã.

    ``r``: lợi suất log (Series hoặc mảng; không NaN); ``floor``/``ceil``: cờ chạm sàn/trần; ở phiên bị chặn r bằng ngưỡng
    quan sát c_t. ``censored=True``: phiên sàn dùng log F_ε((c − μ)/σ), phiên trần dùng log(1 − F_ε(·)); ``False``: coi mọi phiên
    là quan sát đầy đủ. ``limit_terms=True``: thêm δ⁻F_{t−1} + δ⁺C_{t−1} vào σ²_t; ``False``: δ = 0.
    ``censored=False, limit_terms=False`` tương đương GARCH(1,1)–t thông thường (khớp ``arch``).

    Tính trên r·``scale`` như ``tail_preprocessing.garch_standardize`` (``scale=None``: tự chọn 1/độ lệch chuẩn); trả về
    μ, ω, δ theo đơn vị gốc, ``loglik`` theo đơn vị đã nhân scale (so sánh được với arch).
    Trả dict: ``params`` (mu, omega, alpha[1], beta[1], delta_floor, delta_ceil, nu, persistence, loglik, converged),
    ``sigma`` (độ lệch chuẩn có điều kiện, đơn vị gốc), ``z_obs`` = (r − μ)/σ, ``cut_lo``/``cut_hi`` = z_obs ở phiên
    sàn/trần (NaN ở phiên khác)."""
    index = r.index if isinstance(r, pd.Series) else None
    r = _as_vector(r, "r")
    n = len(r)
    if n < 50:
        raise ValueError(f"Cần ít nhất 50 quan sát để khớp GARCH, nhận {n}.")
    if not np.isfinite(r).all():
        raise ValueError("r có NaN/inf — bỏ các phiên thiếu trước.")
    for nm_, fl_ in (("floor", floor), ("ceil", ceil)):
        if isinstance(fl_, pd.Series) and (index is None or not fl_.index.equals(index)):
            raise ValueError(f"r và {nm_} phải cùng chỉ số (truyền r dạng Series cùng index, hoặc cả hai dạng mảng).")
    Fb, Cb = _as_flags(floor, n, "floor"), _as_flags(ceil, n, "ceil")
    if (Fb & Cb).any():
        raise ValueError("Có phiên vừa chạm sàn vừa chạm trần.")
    if scale is None:
        sd0 = float(r.std())
        if not sd0 > 0:
            raise ValueError("r hằng số — không khớp được GARCH.")
        scale = 1.0 / sd0
    if not scale > 0:
        raise ValueError("scale phải dương.")
    x = r * scale
    var0 = float(x.var())
    if not var0 > 0:
        raise ValueError("r hằng số — không khớp được GARCH.")
    F, C = Fb.astype(float), Cb.astype(float)
    idxF, idxC = np.flatnonzero(Fb), np.flatnonzero(Cb)
    s2_floor = 1e-12 * var0
    use_cens = censored and (idxF.size + idxC.size > 0)

    def unpack(phi):
        mu = phi[0]
        omega = np.exp(phi[1])
        pers = float(_logistic(phi[2]))
        share = float(_logistic(phi[3]))
        a, b = pers * share, pers * (1 - share)
        # δ = exp(φ) − ω ≥ −ω ⇒ ω + δ ≥ 0: đầu vào đệ quy luôn ≥ 0 nên σ²_t > 0 mà không cần cắt trong vòng lặp
        dF, dC = (np.exp(phi[4]) - omega, np.exp(phi[5]) - omega) if limit_terms else (0.0, 0.0)
        nu = 2.05 + np.exp(phi[-1])
        return mu, omega, a, b, dF, dC, nu

    def nll(phi):
        mu, omega, a, b, dF, dC, nu = unpack(phi)
        s2 = _garch_variance(x, mu, omega, a, b, dF, dC, F, C, var0, s2_floor)
        z = (x - mu) / np.sqrt(s2)
        c0 = special.gammaln(0.5 * (nu + 1)) - special.gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2))
        ll = c0 - 0.5 * np.log(s2) - 0.5 * (nu + 1) * np.log1p(z * z / (nu - 2))
        if use_cens:
            s = np.sqrt(nu / (nu - 2))
            if idxF.size:
                ll[idxF] = np.log(np.maximum(special.stdtr(nu, s * z[idxF]), 1e-300))
            if idxC.size:
                ll[idxC] = np.log(np.maximum(special.stdtr(nu, -s * z[idxC]), 1e-300))
        tot = -ll.sum()
        return float(tot) if np.isfinite(tot) else PENALTY

    presets = [(0.10, 0.85, 6.0), (0.05, 0.90, 12.0), (0.15, 0.70, 4.0)]
    starts = []
    for a0, b0, nu0 in presets[:max(1, n_starts)]:
        pers = a0 + b0
        phi = [float(x.mean()), np.log(var0 * (1 - pers)), _logit(pers), _logit(a0 / pers)]
        if limit_terms:
            phi += [phi[1], phi[1]]                     # δ = 0 lúc khởi đầu
        phi.append(np.log(nu0 - 2.05))
        starts.append(np.array(phi, float))
    bounds = [(float(x.min()), float(x.max())), (np.log(1e-8 * var0), np.log(10 * var0)), (-8.0, 9.0), (-8.0, 8.0)]
    if limit_terms:
        bounds += [(np.log(1e-10 * var0), np.log(20 * var0))] * 2
    bounds.append((-5.0, np.log(500.0)))
    phi, f, ok = _minimize(nll, starts, bounds)
    mu, omega, a, b, dF, dC, nu = unpack(phi)
    s2 = _garch_variance(x, mu, omega, a, b, dF, dC, F, C, var0, s2_floor)
    sig = np.sqrt(s2)
    z = (x - mu) / sig
    ix = index if index is not None else pd.RangeIndex(n)
    return {
        "params": {"mu": mu / scale, "omega": omega / scale ** 2, "alpha[1]": a, "beta[1]": b,
                   "delta_floor": dF / scale ** 2, "delta_ceil": dC / scale ** 2, "nu": nu, "persistence": a + b,
                   "loglik": float(-f), "converged": ok},
        "sigma": pd.Series(sig / scale, index=ix, name="sigma"),
        "z_obs": pd.Series(z, index=ix, name="z_obs"),
        "cut_lo": pd.Series(np.where(Fb, z, np.nan), index=ix, name="cut_lo"),
        "cut_hi": pd.Series(np.where(Cb, z, np.nan), index=ix, name="cut_hi"),
    }


def censored_garch_standardize(returns: pd.DataFrame, floor: pd.DataFrame, ceil: pd.DataFrame, **kw
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cùng API ``tail_preprocessing.garch_standardize`` nhưng khớp GARCH–t có kiểm duyệt từng cột.

    Trả (Z_obs: phần dư chuẩn hoá quan sát — ở phiên bị chặn bằng ngưỡng chuẩn hoá —, bảng tham số). ``**kw`` chuyển cho
    ``fit_censored_garch`` (censored, limit_terms, scale, n_starts). Cảnh báo khi không hội tụ."""
    for name, fl in (("floor", floor), ("ceil", ceil)):
        missing = [c for c in returns.columns if c not in fl.columns]
        if missing:
            raise ValueError(f"{name} thiếu các cột {missing[:5]}…")
    z, params = {}, {}
    for col in returns.columns:
        r = returns[col].dropna()
        fl = floor[col].reindex(r.index).fillna(False).astype(bool)
        ce = ceil[col].reindex(r.index).fillna(False).astype(bool)
        res = fit_censored_garch(r, fl, ce, **kw)
        if not res["params"]["converged"]:
            warnings.warn(f"GARCH kiểm duyệt cho {col} không hội tụ; kiểm tra lại phần dư.", stacklevel=2)
        z[col] = res["z_obs"]
        params[col] = res["params"]
    return pd.DataFrame(z), pd.DataFrame(params).T


def censoring_cuts(Z_obs: pd.DataFrame, floor: pd.DataFrame, ceil: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cut_lo, cut_hi) = Z_obs ở phiên chạm sàn/trần, NaN ở phiên khác — đầu vào cho ``impute_latent``."""
    fl = floor.reindex(index=Z_obs.index, columns=Z_obs.columns).fillna(False).astype(bool)
    ce = ceil.reindex(index=Z_obs.index, columns=Z_obs.columns).fillna(False).astype(bool)
    return Z_obs.where(fl), Z_obs.where(ce)


def _nu_vector(nu, columns) -> np.ndarray:
    v = nu.reindex(columns).to_numpy(float) if isinstance(nu, pd.Series) else np.broadcast_to(
        np.asarray(nu, float), (len(columns),)).copy()
    if not (v > 2).all():
        raise ValueError("Bậc tự do ν phải > 2 (phương sai hữu hạn).")
    return v


def _check_cuts(Z_obs, cut_lo, cut_hi) -> None:
    if not isinstance(Z_obs, pd.DataFrame):
        raise ValueError("Z_obs phải là DataFrame.")
    for name, c in (("cut_lo", cut_lo), ("cut_hi", cut_hi)):
        if c.shape != Z_obs.shape:
            raise ValueError(f"{name} phải cùng kích thước với Z_obs ({Z_obs.shape}), nhận {c.shape}.")
    if (cut_lo.notna().to_numpy() & cut_hi.notna().to_numpy()).any():
        raise ValueError("Có phiên vừa bị chặn dưới vừa bị chặn trên.")


def _lower_tail_draw(c: np.ndarray, nu: float, rng: np.random.Generator) -> np.ndarray:
    """Rút ε ~ t chuẩn hoá có điều kiện ε ≤ c (nghịch đảo CDF): ε = T⁻¹(U·F_T(s·c))/s, s = √(ν/(ν − 2))."""
    s = np.sqrt(nu / (nu - 2))
    top = special.stdtr(nu, s * c)
    u = np.maximum(rng.random(len(c)) * top, 1e-300)
    return special.stdtrit(nu, u) / s


def impute_latent(Z_obs: pd.DataFrame, cut_lo: pd.DataFrame, cut_hi: pd.DataFrame, nu, m: int,
                  rng: np.random.Generator) -> list[pd.DataFrame]:
    """``m`` bản điền phần dư tiềm ẩn: phiên sàn (cut_lo không NaN) thay bằng rút từ t chuẩn hoá cắt cụt (−∞, cut_lo], phiên
    trần từ [cut_hi, ∞); phiên thường giữ nguyên. ``nu``: Series theo cột (hoặc số). Tái lập theo ``rng``."""
    if m < 1:
        raise ValueError("m phải ≥ 1.")
    _check_cuts(Z_obs, cut_lo, cut_hi)
    nus = _nu_vector(nu, Z_obs.columns)
    A = Z_obs.to_numpy(float)
    lo, hi = cut_lo.to_numpy(float), cut_hi.to_numpy(float)
    out = []
    for _ in range(m):
        B = A.copy()
        for j in range(A.shape[1]):
            il, ih = np.flatnonzero(~np.isnan(lo[:, j])), np.flatnonzero(~np.isnan(hi[:, j]))
            if il.size:
                B[il, j] = _lower_tail_draw(lo[il, j], nus[j], rng)
            if ih.size:
                B[ih, j] = -_lower_tail_draw(-hi[ih, j], nus[j], rng)       # đối xứng: ε ≥ c ⇔ −ε ≤ −c
        out.append(pd.DataFrame(B, index=Z_obs.index, columns=Z_obs.columns))
    return out


def _t_lower_mean(c, nu):
    """E[ε | ε ≤ c] cho ε ~ t chuẩn hoá: E[T | T ≤ t0] = −(ν + t0²)/(ν − 1)·f_T(t0)/F_T(t0), t0 = s·c, rồi chia s."""
    s = np.sqrt(nu / (nu - 2))
    t0 = s * c
    logf = special.gammaln(0.5 * (nu + 1)) - special.gammaln(0.5 * nu) - 0.5 * np.log(np.pi * nu) \
        - 0.5 * (nu + 1) * np.log1p(t0 * t0 / nu)
    ratio = np.exp(logf - np.log(np.maximum(special.stdtr(nu, t0), 1e-300)))
    return -(nu + t0 * t0) / (nu - 1) * ratio / s


def expected_latent(Z_obs: pd.DataFrame, cut_lo: pd.DataFrame, cut_hi: pd.DataFrame, nu) -> pd.DataFrame:
    """Bản điểm: thay phiên sàn bằng E[ε | ε ≤ cut_lo], phiên trần bằng E[ε | ε ≥ cut_hi] (công thức đóng của t); khác giữ nguyên."""
    _check_cuts(Z_obs, cut_lo, cut_hi)
    nus = _nu_vector(nu, Z_obs.columns)
    if (nus <= 2).any():
        raise ValueError("ν phải > 2.")
    B = Z_obs.to_numpy(float).copy()
    lo, hi = cut_lo.to_numpy(float), cut_hi.to_numpy(float)
    for j in range(B.shape[1]):
        il, ih = np.flatnonzero(~np.isnan(lo[:, j])), np.flatnonzero(~np.isnan(hi[:, j]))
        if il.size:
            B[il, j] = _t_lower_mean(lo[il, j], nus[j])
        if ih.size:
            B[ih, j] = -_t_lower_mean(-hi[ih, j], nus[j])
    return pd.DataFrame(B, index=Z_obs.index, columns=Z_obs.columns)


# ============================================================================ C. EASE với nhiều lần điền
def ease_mi(Z_list: Sequence, k: int, kind: str = "psi", deconf: Callable | None = None, seed: int = 0) -> dict:
    """EASE trên nhiều bản điền (Rubin: gộp theo trung bình). Mỗi bản Z_j (n × p) → (tuỳ chọn) ``deconf(Z_j)`` → Ψ̂_j; hệ số gộp
    = trung bình các Ψ̂_j; thứ tự = ``CT.ease`` trên hệ số gộp.

    Trả dict: ``coef`` (Ψ̂ trung bình), ``order``, ``orders`` (thứ tự từng bản), ``agreement`` (trung bình
    ``TP.order_agreement`` giữa thứ tự từng bản và thứ tự gộp), ``agreement_each`` (từng bản)."""
    if len(Z_list) == 0:
        raise ValueError("Z_list rỗng.")
    coefs, orders, labels = [], [], None
    for i, Z in enumerate(Z_list):
        X = deconf(Z) if deconf is not None else Z
        if i == 0 and isinstance(X, pd.DataFrame):
            labels = list(X.columns)                       # nhãn lấy từ đúng lần xử lý bản đầu tiên
        coef = CT.tail_coef_matrix(X, k=k, kind=kind, seed=seed)
        coefs.append(np.asarray(coef, float))
        orders.append(CT.ease(coef))
    shapes = {c.shape for c in coefs}
    if len(shapes) != 1:
        raise ValueError("Các bản điền cho ma trận hệ số khác kích thước.")
    mean = np.mean(coefs, axis=0)
    coef_df = pd.DataFrame(mean, index=labels, columns=labels) if labels else mean
    order = CT.ease(coef_df)
    each = [TP.order_agreement(o, order) for o in orders]
    return {"coef": coef_df, "order": order, "orders": orders, "agreement": float(np.mean(each)),
            "agreement_each": each}


# ============================================================================ D. Mô phỏng dữ liệu bị chặn biên độ
def simulate_limit_scm(n: int, p: int, alpha: float, rng: np.random.Generator | None = None,
                       hit_rate: float | Sequence[float] = 0.025, n_hidden: int = 2, vol_persist: float = 0.97,
                       vol_sd: float = 0.25, spill: bool = False) -> dict:
    """SCM tuyến tính nhiễu t(α) với nhân tố ẩn (``CT.simulate_scm(..., "hidden")``), nhân biến động chung exp(h_t)
    (h_t AR(1) với hệ số ``vol_persist``, độ lệch chuẩn dừng ``vol_sd``), rồi chặn theo ngưỡng tuyệt đối L_j.

    L_j = phân vị (1 − hit_rate_j) của |R_j| (``hit_rate`` số hoặc mảng p phần tử ∈ (0, 0,5)); X = clip(R, −L, L);
    ``spill=True``: phần vượt ngưỡng dồn sang phiên sau (R_eff_t = R_t + phần dư_{t−1}).
    Trả dict: X (đã chặn), floor, ceil (cờ bool n × p), adj, observed, latent (R, chưa chặn), limit (L, p phần tử),
    carry (phần dư dồn cuối kỳ, p phần tử; 0 nếu không ``spill``)."""
    rng = rng if rng is not None else np.random.default_rng()
    try:
        hr = np.broadcast_to(np.asarray(hit_rate, float), (p,)).copy()
    except ValueError:
        raise ValueError(f"hit_rate phải là một số hoặc mảng {p} phần tử.") from None
    if not ((hr > 0) & (hr < 0.5)).all():
        raise ValueError("hit_rate phải thuộc (0; 0,5).")
    if not 0 <= vol_persist < 1 or vol_sd < 0:
        raise ValueError("Cần 0 ≤ vol_persist < 1 và vol_sd ≥ 0.")
    if n < 20:
        raise ValueError("n phải ≥ 20.")
    base = CT.simulate_scm(n, p, alpha, "hidden", rng, n_hidden=n_hidden)
    h = np.empty(n)
    h[0] = rng.normal(0, vol_sd)
    inno = rng.normal(0, vol_sd * np.sqrt(1 - vol_persist ** 2), n)
    for t in range(1, n):
        h[t] = vol_persist * h[t - 1] + inno[t]
    R = base["X"] * np.exp(h)[:, None]
    L = np.array([np.quantile(np.abs(R[:, j]), 1 - hr[j]) for j in range(p)])
    if spill:
        X = np.empty_like(R)
        eff = np.empty_like(R)
        carry = np.zeros(p)
        for t in range(n):
            eff[t] = R[t] + carry
            X[t] = np.clip(eff[t], -L, L)
            carry = eff[t] - X[t]
        floor, ceil = eff < -L, eff > L
    else:
        X = np.clip(R, -L, L)
        floor, ceil = R < -L, R > L
        carry = np.zeros(p)
    return {"X": X, "floor": floor, "ceil": ceil, "adj": base["adj"], "observed": base["observed"], "latent": R,
            "limit": L, "carry": carry}


def _garch_z(X: np.ndarray, floor: np.ndarray, ceil: np.ndarray, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(range(X.shape[1]))
    df = lambda a: pd.DataFrame(a, columns=cols)               # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return censored_garch_standardize(df(X), df(floor), df(ceil), scale=None, **kw)


def _one_limit_rep(alpha, hit_rate, rep, child_seed, n, p, m_imp, limit_terms) -> list[dict]:
    rng = np.random.default_rng(child_seed)
    sim = simulate_limit_scm(n, p, alpha, rng, hit_rate=hit_rate)
    k = CT.default_k(n)
    truth = CT.observed_ancestral_dag(sim["adj"], sim["observed"])
    no_flag = np.zeros_like(sim["floor"])

    def sid_of(order) -> float:
        return CT.sid(truth, CT.order_to_adjacency(order))

    rows = {}
    Z, _ = _garch_z(sim["latent"], no_flag, no_flag, censored=False, limit_terms=False)
    rows[SIM_METHODS[0]] = sid_of(CT.ease(CT.tail_coef_matrix(Z, k=k)))
    Z, _ = _garch_z(sim["X"], no_flag, no_flag, censored=False, limit_terms=False)
    rows[SIM_METHODS[1]] = sid_of(CT.ease(CT.tail_coef_matrix(Z, k=k)))
    Z, par = _garch_z(sim["X"], sim["floor"], sim["ceil"], censored=True, limit_terms=limit_terms)
    rows[SIM_METHODS[2]] = sid_of(CT.ease(CT.tail_coef_matrix(Z, k=k)))
    cut_lo, cut_hi = censoring_cuts(Z, pd.DataFrame(sim["floor"], columns=Z.columns),
                                    pd.DataFrame(sim["ceil"], columns=Z.columns))
    Zs = impute_latent(Z, cut_lo, cut_hi, par["nu"].astype(float), m_imp, rng)
    rows[SIM_METHODS[3]] = sid_of(ease_mi(Zs, k)["order"])
    return [{"alpha": alpha, "hit_rate": float(np.mean(hit_rate)), "rep": rep, "method": m, "SID": s}
            for m, s in rows.items()]


def limit_simulation_study(reps: int = 20, n: int = 2000, p: int = 10, alphas=(1.5, 3.0), hit_rates=(0.01, 0.025, 0.05),
                           seed: int = 2026, m_imp: int = 10, n_jobs: int = -1, limit_terms: bool = False,
                           progress: bool = False) -> pd.DataFrame:
    """Lặp mô phỏng dữ liệu bị chặn theo lưới (α, hit_rate) × ``reps``; trả DataFrame dạng dài (alpha, hit_rate, rep,
    method, SID) với 4 phương pháp ``SIM_METHODS``: EASE (k = ⌊n^0,4⌋) trên phần dư (1) GARCH–t của chuỗi KHÔNG chặn
    (đối chứng lý tưởng), (2) GARCH–t thường trên chuỗi bị chặn, (3) GARCH–t kiểm duyệt (điểm), (4) kiểm duyệt + ``m_imp``
    lần điền (ease_mi). SID = ``CT.sid`` so với DAG tổ tiên trên biến quan sát. ``limit_terms``: thêm δ⁻, δ⁺ ở (3)–(4).

    Song song bằng joblib theo từng lần lặp, seed con sinh từ ``SeedSequence(seed)`` ⇒ tái lập, không phụ thuộc ``n_jobs``."""
    from joblib import Parallel, delayed

    grid = [(a, h, r) for a, h in itertools.product(alphas, hit_rates) for r in range(reps)]
    children = np.random.SeedSequence(seed).spawn(len(grid))
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=10 if progress else 0)(
        delayed(_one_limit_rep)(a, h, r, child, n, p, m_imp, limit_terms) for (a, h, r), child in zip(grid, children))
    if progress:
        print(f"{len(grid)} bộ dữ liệu × {len(SIM_METHODS)} phương pháp: {time.time() - t0:.0f}s")
    return pd.DataFrame([row for rows in out for row in rows])
