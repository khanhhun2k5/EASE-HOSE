"""Khử nhân tố chung PHI TUYẾN bằng autoencoder trước khi chạy EASE.

Bối cảnh: Calibration cho thấy phải khử nhân tố thị trường trước EASE, nhưng các cách khử đều tuyến tính
(β không đổi). EDA lại cho thấy phụ thuộc đuôi dưới vượt mức copula Gauss: khi thị trường sập, các mã cùng giảm
mạnh hơn mức một β cố định dự báo. Khi đó phần dư tuyến tính e_i = x_i − β̂_i f còn (β_i⁻ − β̂_i)·f⁻ trong đuôi
dưới — một thành phần CHUNG còn sót, đúng loại nhiễu làm EASE sinh thứ tự giả.

Mô hình (đặt tên theo tài liệu):

* ``linear``  — autoencoder tuyến tính một lớp ẩn K nơ-ron: x̂ = W(Hᵀx + b_E) + b_D. Với sai số bình phương, nghiệm
  trùng không gian PCA (Gu, Kelly & Xiu 2021, Mệnh đề 1, doi:10.1016/j.jeconom.2020.07.009; Bourlard & Kamp 1988).
* ``relu``    — autoencoder của Spilak & Härdle (2022, arXiv:2204.02757, pt. 1): x̂ = W·ReLU(Hᵀx + b_E) + b_D,
  tuỳ chọn ràng buộc H, W ≥ 0 (``nonneg``), phạt L1, phạt trực giao mềm ‖WᵀW − I‖² và phạt hiệp phương sai
  ngoài đường chéo của nhân tố (pt. 4–6). Lưu ý của chính bài báo (mục 4.1): vùng ReLU tắt (đuôi trái) KHÔNG
  được giải thích — tức cú sập chung không bị khử; mô hình này được giữ để đối chiếu.
* ``nonlinear`` — đề xuất của đồ án: nhân tố vẫn là danh mục tuyến tính f = Hᵀx + b_E (mạng nhân tố một lớp của
  Gu, Kelly & Xiu 2021, pt. 13–15 — nhân tố diễn giải được như danh mục), còn phần giải mã có hệ số tải phụ thuộc
  trạng thái: x̂ = W f + V·ReLU(A f + c) + b_D. Phần V·ReLU(·) là hàm tuyến tính từng khúc của f, nên mô hình
  chứa β⁺ ≠ β⁻ (hệ số tải khác nhau khi thị trường tăng/giảm, Ang, Chen & Xing 2006) và chứa ``linear`` (V = 0).

Ước lượng: Adam, lô nhỏ, dừng sớm trên tập kiểm định, phạt L1, trung bình tập hợp nhiều seed (cả hai bài báo).
Khởi tạo H, W từ PCA (``linear``/``nonlinear``) — điểm xuất phát là nghiệm tuyến tính, phần phi tuyến V bắt đầu
từ 0. Spilak & Härdle khởi tạo từ NMF lồi; ở đây dùng PCA chiếu lên miền ≥ 0 cho ``relu`` khi ``nonneg``.

Phần dư dùng cho EASE được tính **ngoài mẫu bằng cross-fitting** (chia thời gian thành ``n_folds`` khối liên tiếp,
huấn luyện trên các khối còn lại, dự báo khối giữ lại): autoencoder đủ linh hoạt có thể "học thuộc" cả cú sốc riêng
của từng mã nếu đánh giá trong mẫu.

Đánh giá: R² tổng (Gu, Kelly & Xiu 2021, pt. 20) và R² trong đuôi (chỉ các phiên thị trường cực trị).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

ARCHS = ("pca", "linear", "relu", "nonlinear")
ARCH_LABELS = {"pca": "PCA", "linear": "AE tuyến tính", "relu": "AE ReLU (Spilak–Härdle)",
               "nonlinear": "AE hệ số tải phi tuyến"}


# ============================================================================ cấu hình
@dataclass(frozen=True)
class AEConfig:
    """Siêu tham số. ``K``: số nhân tố; ``hidden``: số nơ-ron ReLU của phần giải mã phi tuyến (``nonlinear``).

    l1, lam_orth, lam_decor: λ₁, λ₂, λ₃ của Spilak & Härdle (pt. 4–6). ``val_frac``: tỷ lệ phiên huấn luyện
    (chọn ngẫu nhiên) dùng để dừng sớm. ``n_ensemble``: số seed lấy trung bình dự báo.
    """

    K: int = 1
    arch: str = "nonlinear"
    hidden: int = 8
    l1: float = 1e-4
    lam_orth: float = 0.0
    lam_decor: float = 0.0
    nonneg: bool = False
    lr: float = 5e-3
    epochs: int = 300
    batch: int = 256
    patience: int = 25
    val_frac: float = 0.15
    n_ensemble: int = 1
    seed: int = 0

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch phải thuộc {ARCHS}")
        if self.K < 1:
            raise ValueError("K phải ≥ 1.")
        if self.hidden < 1 or self.epochs < 1 or self.batch < 1 or self.n_ensemble < 1:
            raise ValueError("hidden, epochs, batch, n_ensemble phải ≥ 1.")
        if not 0 < self.val_frac < 1:
            raise ValueError("val_frac phải trong (0, 1).")
        if min(self.l1, self.lam_orth, self.lam_decor) < 0 or self.lr <= 0:
            raise ValueError("Hệ số phạt phải ≥ 0 và lr > 0.")


# ============================================================================ PCA (đối chứng, không cần torch)
def _pca_loadings(Xs: np.ndarray, K: int) -> np.ndarray:
    """K vectơ riêng đầu của ma trận tương quan (cột đã chuẩn hoá), dấu: tổng phần tử dương."""
    R = np.corrcoef(Xs, rowvar=False)
    w, v = np.linalg.eigh(R)
    L = v[:, np.argsort(w)[::-1][:K]]
    return L * np.where(L.sum(axis=0) < 0, -1, 1)


class FactorModel:
    """Mô hình đã khớp: ``predict(X)`` trả x̂ (đơn vị gốc), ``factors(X)`` trả f (n × K)."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray, predict_std, factor_std, history: list | None = None):
        self.mu, self.sd = mu, sd
        self._predict, self._factor = predict_std, factor_std
        self.history = history or []

    def _std(self, X) -> np.ndarray:
        A = np.asarray(X, float)
        if A.ndim != 2 or A.shape[1] != len(self.mu):
            raise ValueError(f"X phải có {len(self.mu)} cột.")
        return (A - self.mu) / self.sd

    def predict(self, X) -> np.ndarray:
        return self._predict(self._std(X)) * self.sd + self.mu

    def factors(self, X) -> np.ndarray:
        return self._factor(self._std(X))


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0, ddof=0)
    if (sd <= 0).any() or np.isnan(sd).any():
        raise ValueError("Có cột hằng hoặc NaN — không chuẩn hoá được.")
    return mu, sd


def fit_pca(X, K: int) -> FactorModel:
    """PCA K nhân tố trên tương quan Pearson: f = x_s L, x̂_s = f Lᵀ (L trực chuẩn ⇒ phép chiếu)."""
    A = np.asarray(X, float)
    if not 1 <= K < A.shape[1]:
        raise ValueError("K phải trong [1, p − 1].")
    mu, sd = _standardize_fit(A)
    L = _pca_loadings((A - mu) / sd, K)
    return FactorModel(mu, sd, lambda S: S @ L @ L.T, lambda S: S @ L)


# ============================================================================ autoencoder (torch)
def _build_net(p: int, cfg: AEConfig, L0: np.ndarray, rng_seed: int):
    import torch
    from torch import nn

    torch.manual_seed(rng_seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(p, cfg.K)                          # f = Hᵀx + b_E (danh mục tuyến tính)
            self.dec = nn.Linear(cfg.K, p)                          # W f + b_D
            if cfg.arch == "nonlinear":
                self.gate = nn.Linear(cfg.K, cfg.hidden)            # A f + c
                self.V = nn.Linear(cfg.hidden, p, bias=False)       # V·ReLU(A f + c)
            with torch.no_grad():                                   # khởi tạo từ PCA (điểm xuất phát tuyến tính)
                L = torch.tensor(L0, dtype=torch.float32)
                if cfg.arch == "relu" and cfg.nonneg:
                    L = L.clamp(min=0)
                    L = L / L.norm(dim=0, keepdim=True).clamp(min=1e-8)
                self.enc.weight.copy_(L.T)
                self.dec.weight.copy_(L)
                self.enc.bias.zero_()
                self.dec.bias.zero_()
                if cfg.arch == "relu":                              # ReLU bật ở hầu hết phiên lúc đầu
                    self.enc.bias.fill_(1.0)
                    self.dec.bias.copy_(-(L.sum(dim=1)))
                if cfg.arch == "nonlinear":
                    self.V.weight.zero_()                           # bắt đầu đúng bằng mô hình tuyến tính

        def factor(self, x):
            f = self.enc(x)
            return torch.relu(f) if cfg.arch == "relu" else f

        def forward(self, x):
            f = self.factor(x)
            out = self.dec(f)
            if cfg.arch == "nonlinear":
                out = out + self.V(torch.relu(self.gate(f)))
            return out, f

    return Net()


def _penalty(net, f, cfg: AEConfig):
    import torch

    pen = cfg.l1 * (net.enc.weight.abs().sum() + net.dec.weight.abs().sum())
    if cfg.arch == "nonlinear":
        pen = pen + cfg.l1 * net.V.weight.abs().sum()
    if cfg.lam_orth > 0:
        W = net.dec.weight
        pen = pen + cfg.lam_orth * ((W.T @ W - torch.eye(W.shape[1])) ** 2).sum()
    if cfg.lam_decor > 0 and f.shape[1] > 1 and f.shape[0] > 2:
        fc = (f - f.mean(0)) / f.std(0).clamp(min=1e-6)
        S = fc.T @ fc / (len(fc) - 1)
        pen = pen + cfg.lam_decor * ((S - torch.diag(torch.diag(S))) ** 2).sum()
    return pen


def _train_one(S: np.ndarray, cfg: AEConfig, L0: np.ndarray, seed: int):
    import torch

    rng = np.random.default_rng(seed)
    n = len(S)
    perm = rng.permutation(n)
    n_val = max(1, int(round(cfg.val_frac * n)))
    tr, va = perm[n_val:], perm[:n_val]
    Xt = torch.tensor(S, dtype=torch.float32)
    net = _build_net(S.shape[1], cfg, L0, seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(cfg.epochs):
        net.train()
        order = tr[rng.permutation(len(tr))]
        for b in range(0, len(order), cfg.batch):
            xb = Xt[order[b:b + cfg.batch]]
            out, f = net(xb)
            loss = ((out - xb) ** 2).mean() + _penalty(net, f, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg.nonneg:                                          # chiếu lên miền ≥ 0, cột W chuẩn đơn vị
                with torch.no_grad():
                    net.enc.weight.clamp_(min=0)
                    net.dec.weight.clamp_(min=0)
                    net.dec.weight.div_(net.dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8))
        net.eval()
        with torch.no_grad():
            vl = float(((net(Xt[va])[0] - Xt[va]) ** 2).mean())
        hist.append(vl)
        if not np.isfinite(vl):
            if best_state is None:
                raise FloatingPointError(f"Loss kiểm định không hữu hạn ngay từ epoch {ep} — giảm lr (đang {cfg.lr}) "
                                         "hoặc kiểm tra dữ liệu.")
            break                                                   # phân kỳ giữa chừng: giữ trạng thái tốt nhất
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, hist


def fit_autoencoder(X, cfg: AEConfig) -> FactorModel:
    """Khớp autoencoder theo ``cfg`` (chuẩn hoá cột bằng trung bình/độ lệch chuẩn của chính X).

    ``cfg.arch == "pca"`` chuyển sang ``fit_pca``. Với ``n_ensemble > 1``: dự báo và nhân tố là trung bình
    của các mạng huấn luyện từ seed khác nhau (nhân tố chỉ trung bình được khi các mạng cùng khởi tạo PCA —
    thoả mãn ở đây vì dấu/thứ tự nhân tố cố định theo PCA).
    """
    import torch

    if cfg.arch == "pca":
        return fit_pca(X, cfg.K)
    A = np.asarray(X, float)
    if not 1 <= cfg.K < A.shape[1]:
        raise ValueError("K phải trong [1, p − 1].")
    if np.isnan(A).any():
        raise ValueError("X có NaN.")
    mu, sd = _standardize_fit(A)
    S = (A - mu) / sd
    L0 = _pca_loadings(S, cfg.K)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)                                        # chạy song song bằng joblib ở tầng ngoài
    nets, hists = [], []
    try:
        for e in range(cfg.n_ensemble):
            net, h = _train_one(S, cfg, L0, seed=cfg.seed * 1000 + e)
            nets.append(net)
            hists.append(h)
    finally:
        torch.set_num_threads(n_threads)                            # không để lại thiết lập cho phiên gọi

    def _run(Sx, which):
        with torch.no_grad():
            t = torch.tensor(np.asarray(Sx, float), dtype=torch.float32)
            outs = [net(t)[which].numpy().astype(float) for net in nets]
        return np.mean(outs, axis=0)

    return FactorModel(mu, sd, lambda Sx: _run(Sx, 0), lambda Sx: _run(Sx, 1), hists)


# ============================================================================ cross-fitting
def time_folds(n: int, n_folds: int) -> np.ndarray:
    """Nhãn khối (0..n_folds−1) cho n phiên: các khối thời gian liên tiếp, kích thước gần bằng nhau."""
    if not 2 <= n_folds <= n:
        raise ValueError("n_folds phải trong [2, n].")
    return np.minimum(np.arange(n) * n_folds // n, n_folds - 1)


def crossfit(X: pd.DataFrame, cfg: AEConfig, n_folds: int = 5, n_jobs: int = 1) -> dict:
    """Dự báo ngoài mẫu cho mọi phiên: huấn luyện trên các khối khác, dự báo khối giữ lại.

    Trả dict: fitted (DataFrame x̂), resid (X − x̂), factors (DataFrame n × K, mỗi khối từ mô hình riêng — dấu cố
    định theo PCA nhưng thang có thể khác giữa các khối), fold (nhãn khối).
    """
    if X.isna().any().any():
        raise ValueError("X có NaN — hãy bỏ các dòng thiếu trước.")
    A = X.to_numpy(float)
    fold = time_folds(len(A), n_folds)

    def one(f):
        m = fit_autoencoder(A[fold != f], replace(cfg, seed=cfg.seed + 101 * f))
        te = A[fold == f]
        return f, m.predict(te), m.factors(te)

    if n_jobs == 1:
        parts = [one(f) for f in range(n_folds)]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs)(delayed(one)(f) for f in range(n_folds))
    fitted = np.empty_like(A)
    F = np.empty((len(A), cfg.K))
    for f, pr, fa in parts:
        fitted[fold == f], F[fold == f] = pr, fa
    fitted_df = pd.DataFrame(fitted, index=X.index, columns=X.columns)
    return {"fitted": fitted_df, "resid": X - fitted_df,
            "factors": pd.DataFrame(F, index=X.index, columns=[f"F{i + 1}" for i in range(cfg.K)]),
            "fold": pd.Series(fold, index=X.index, name="fold")}


# ============================================================================ đánh giá
def r2_total(X, fitted, axis: int | None = None):
    """R² tổng = 1 − Σ(x − x̂)² / Σx² (Gu, Kelly & Xiu 2021, pt. 20; mẫu số không trừ trung bình).
    ``axis=0``: một giá trị cho mỗi cột (mỗi mã); None: gộp toàn bảng."""
    A, B = np.asarray(X, float), np.asarray(fitted, float)
    return 1 - ((A - B) ** 2).sum(axis=axis) / (A ** 2).sum(axis=axis)


def market_tail_mask(X: pd.DataFrame, q: float = 0.05, tail: str = "lower") -> np.ndarray:
    """Phiên mà trung bình đều của các cột nằm trong đuôi ``tail`` mức q (phân vị thực nghiệm)."""
    if not 0 < q < 0.5:
        raise ValueError("q phải trong (0; 0,5).")
    m = np.asarray(X, float).mean(axis=1)
    if tail == "lower":
        return m <= np.quantile(m, q)
    if tail == "upper":
        return m >= np.quantile(m, 1 - q)
    raise ValueError("tail phải là 'lower' hoặc 'upper'.")


def fit_report(X: pd.DataFrame, fitted: pd.DataFrame, q: float = 0.05) -> dict:
    """R² tổng toàn mẫu và trên các phiên thị trường cực trị (đuôi dưới/trên mức q)."""
    lo, up = market_tail_mask(X, q, "lower"), market_tail_mask(X, q, "upper")
    return {"R2": float(r2_total(X, fitted)), "R2_duoi_duoi": float(r2_total(X[lo], fitted[lo])),
            "R2_duoi_tren": float(r2_total(X[up], fitted[up]))}


def up_down_betas(X: pd.DataFrame, market: pd.Series | None = None) -> pd.DataFrame:
    """Hồi quy x_i = a + β⁺·m⁺ + β⁻·m⁻ + e với m⁺ = max(m, 0), m⁻ = min(m, 0) (Ang, Chen & Xing 2006).

    m mặc định là trung bình đều các cột KHÁC i (LOO, không chứa i). Sai số chuẩn HC0 (White) cho
    D = β⁻ − β⁺; t = D / SE(D). Trả DataFrame: beta_up, beta_down, D, t_D.
    """
    A = X.to_numpy(float)
    n, p = A.shape
    total = A.sum(axis=1)
    rows = {}
    for i, s in enumerate(X.columns):
        m = market.reindex(X.index).to_numpy(float) if market is not None else (total - A[:, i]) / (p - 1)
        D = np.column_stack([np.ones(n), np.maximum(m, 0), np.minimum(m, 0)])
        XtX_inv = np.linalg.inv(D.T @ D)
        b = XtX_inv @ D.T @ A[:, i]
        e = A[:, i] - D @ b
        cov = XtX_inv @ (D.T * e ** 2) @ D @ XtX_inv
        c = np.array([0.0, -1.0, 1.0])
        d = float(c @ b)
        rows[s] = {"beta_up": b[1], "beta_down": b[2], "D": d, "t_D": d / np.sqrt(c @ cov @ c)}
    return pd.DataFrame(rows).T


# ============================================================================ dữ liệu
def extended_log_returns(long: pd.DataFrame, symbols, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Lợi suất log ngày của ``symbols`` (dữ liệu dạng dài symbol, time, close) trên đúng các phiên ``dates``.

    Giá được đưa về lịch ``dates`` (cộng phiên liền trước để tính lợi suất phiên đầu) và điền tiến: phiên mã
    không giao dịch (tạm ngừng) có lợi suất 0, cú nhảy khi giao dịch lại dồn vào phiên đầu tiên có giá.
    Báo lỗi nếu một mã chưa có giá ở phiên đầu.
    """
    d = long.assign(time=pd.to_datetime(long["time"]))
    d = d[d["symbol"].isin(list(symbols))]
    close = d.pivot_table(index="time", columns="symbol", values="close").sort_index()
    missing = [s for s in symbols if s not in close.columns]
    if missing:
        raise ValueError(f"Không có dữ liệu cho: {missing}")
    dates = pd.DatetimeIndex(dates)
    prev = close.index[close.index < dates[0]]
    if len(prev) == 0:
        raise ValueError("Cần ít nhất một phiên trước phiên đầu của dates.")
    cal = dates.insert(0, prev[-1])
    c = close.reindex(close.index.union(cal)).ffill().reindex(cal)[list(symbols)]
    if c.iloc[0].isna().any():
        raise ValueError(f"Chưa có giá ở phiên đầu: {list(c.columns[c.iloc[0].isna()])}")
    return np.log(c / c.shift(1)).iloc[1:]


# ============================================================================ mô phỏng
def simulate_asymmetric_factor_scm(n: int, beta_up, beta_down, df: float, B_coef: np.ndarray | None = None,
                                   rng: np.random.Generator | None = None) -> np.ndarray:
    """x_i = β_i⁺·max(F, 0) + β_i⁻·min(F, 0) + u_i, u = (σ ⊙ ε)(I − B)⁻¹, F, ε ~ Student-t(df) phương sai 1.

    σ_i chọn để Var(x_i) = 1 khi không có cạnh: Var(phần chung) = ½(β⁺² + β⁻²) − c²(β⁺ − β⁻)², c = E[max(F, 0)].
    β⁺ = β⁻ trả về đúng mô hình một nhân tố tuyến tính của ``validation.simulate_factor_scm``.
    ``B_coef[i, j]``: hệ số cạnh i → j (None: H0 không có nhân quả).
    """
    rng = rng if rng is not None else np.random.default_rng()
    bu, bd = np.asarray(beta_up, float), np.asarray(beta_down, float)
    if bu.shape != bd.shape or bu.ndim != 1:
        raise ValueError("beta_up và beta_down phải là vectơ cùng độ dài.")
    if df <= 2:
        raise ValueError("df phải > 2.")
    p = len(bu)
    scale = np.sqrt((df - 2) / df)
    from scipy import special
    # E[max(T, 0)] của Student-t(df) chuẩn hoá: √df Γ((df+1)/2) / (√π (df−1) Γ(df/2)) × scale
    c = scale * np.sqrt(df) * np.exp(special.gammaln((df + 1) / 2) - special.gammaln(df / 2)) / (np.sqrt(np.pi) * (df - 1))
    var_common = 0.5 * (bu ** 2 + bd ** 2) - c ** 2 * (bu - bd) ** 2
    if (var_common >= 1).any():
        raise ValueError("Phần chung có phương sai ≥ 1 ở ít nhất một mã — giảm β.")
    F = rng.standard_t(df, n) * scale
    U = rng.standard_t(df, (n, p)) * scale * np.sqrt(1 - var_common)
    if B_coef is not None:
        U = U @ np.linalg.inv(np.eye(p) - np.asarray(B_coef, float))
    return np.maximum(F, 0)[:, None] * bu + np.minimum(F, 0)[:, None] * bd + U
