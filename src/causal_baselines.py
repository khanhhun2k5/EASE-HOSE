"""Các phương pháp đối chứng (baseline) cho EASE, đúng như mục 5.1.1 của Gnecco và cs. (2021):

| Phương pháp | Đầu ra | Thư viện |
|---|---|---|
| EASE (Ψ̂, k = ⌊n^0,4⌋) | thứ tự nhân quả | ``causal_tail`` (tự cài đặt) |
| ICA-LiNGAM (Shimizu và cs., 2006) | thứ tự nhân quả | ``lingam.ICALiNGAM`` |
| Pairwise LiNGAM (Hyvärinen & Smith, 2013) | thứ tự nhân quả | ``lingam.DirectLiNGAM(measure="pwling")`` |
| Rank PC (Harris & Drton, 2013) | CPDAG | ``causallearn`` PC, kiểm định Fisher-z trên điểm chuẩn theo hạng |
| Ngẫu nhiên | thứ tự nhân quả | hoán vị ngẫu nhiên |

Rank PC gốc dùng tương quan Spearman; ở đây dùng biến đổi điểm chuẩn theo hạng Φ⁻¹(rank/(n+1))
(phương pháp nonparanormal) rồi kiểm định Fisher-z — cùng tinh thần "chỉ dùng hạng", khác biệt nhỏ
ở hiệu chỉnh tương quan. Mức ý nghĩa 5·10⁻⁴ như bài báo.

Mọi hàm nhận ma trận n × p và trả về dict {"order": list | None, "adj": ndarray}; ``adj`` là DAG đầy đủ
theo thứ tự (với phương pháp trả thứ tự) hoặc CPDAG (với Rank PC, cạnh vô hướng = 1 ở cả hai chiều).
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
import io
import warnings

import numpy as np
from scipy import stats

import causal_tail as CT

METHODS = ("EASE", "Pairwise LiNGAM", "ICA-LiNGAM", "Rank PC", "Ngẫu nhiên")


def _from_order(order) -> dict:
    order = [int(i) for i in order]
    return {"order": order, "adj": CT.order_to_adjacency(order)}


def run_ease(X: np.ndarray, k: int | None = None, kind: str = "psi", seed: int = 0) -> dict:
    return _from_order(CT.ease(CT.tail_coef_matrix(np.asarray(X, float), k=k, kind=kind, seed=seed)))


def run_ica_lingam(X: np.ndarray, seed: int = 0) -> dict:
    """ICA-LiNGAM trên các cột đã XÁO TRỘN ngẫu nhiên (theo ``seed``) rồi ánh xạ thứ tự về chỉ số gốc.

    Lý do: khi tín hiệu phi Gauss yếu (p = 20 mã), thứ tự ICA-LiNGAM bám theo thứ tự cột đầu vào (đã kiểm tra:
    đồng thuận 0,81–0,90 với thứ tự cột bị xáo trộn). Xáo trộn cột khiến sai lệch này thành nhiễu ngẫu nhiên thay vì
    thiên lệch có hệ thống; nên lấy thứ tự đồng thuận qua nhiều seed (``ica_lingam_consensus``).
    """
    from lingam import ICALiNGAM
    A = np.asarray(X, float)
    perm = np.random.default_rng(seed).permutation(A.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # FastICA có thể cảnh báo chưa hội tụ với nhiễu đuôi rất dày
        model = ICALiNGAM(random_state=seed, max_iter=2000).fit(A[:, perm])
    return _from_order([perm[i] for i in model.causal_order_])


def ica_lingam_consensus(X: np.ndarray, seeds: Sequence[int] = range(10)) -> tuple[list[int], np.ndarray]:
    """Thứ tự đồng thuận của ICA-LiNGAM qua nhiều seed: sắp theo vị trí trung bình (Borda). Trả (thứ tự, mảng
    seeds × p vị trí) để đánh giá độ ổn định."""
    A = np.asarray(X, float)
    pos = np.empty((len(seeds), A.shape[1]))
    for r, s in enumerate(seeds):
        o = run_ica_lingam(A, seed=s)["order"]
        pos[r, o] = np.arange(A.shape[1])
    return [int(i) for i in np.argsort(pos.mean(axis=0), kind="stable")], pos


def run_pairwise_lingam(X: np.ndarray) -> dict:
    from lingam import DirectLiNGAM
    model = DirectLiNGAM(measure="pwling").fit(np.asarray(X, float))
    return _from_order(model.causal_order_)


def normal_scores(X: np.ndarray) -> np.ndarray:
    """Φ⁻¹(rank/(n+1)) từng cột — đưa biên về chuẩn, chỉ giữ thông tin hạng."""
    X = np.asarray(X, float)
    return stats.norm.ppf(stats.rankdata(X, axis=0) / (X.shape[0] + 1))


def run_rank_pc(X: np.ndarray, alpha: float = 5e-4) -> dict:
    from causallearn.search.ConstraintBased.PC import pc
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc(normal_scores(X), alpha, "fisherz", show_progress=False, verbose=False)
    g = cg.G.graph               # causal-learn: g[j,i]=1 & g[i,j]=-1 ⇔ i → j; -1/-1 vô hướng; 1/1 hai chiều
    p = g.shape[0]
    C = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if g[j, i] == 1 and g[i, j] == -1:
                C[i, j] = 1
            elif g[i, j] != 0 and g[i, j] == g[j, i]:  # vô hướng hoặc hai chiều → coi là vô hướng
                C[i, j] = C[j, i] = 1
    return {"order": None, "adj": C}


def run_random(p: int, rng: np.random.Generator) -> dict:
    return _from_order(rng.permutation(p))


def run_all(X: np.ndarray, rng: np.random.Generator, k: int | None = None,
            methods: tuple[str, ...] = METHODS) -> dict[str, dict]:
    """Chạy các phương pháp trên cùng dữ liệu; trả {tên: kết quả}."""
    X = np.asarray(X, float)
    out = {}
    for m in methods:
        if m == "EASE":
            out[m] = run_ease(X, k=k)
        elif m == "ICA-LiNGAM":
            out[m] = run_ica_lingam(X, seed=int(rng.integers(1 << 30)))
        elif m == "Pairwise LiNGAM":
            out[m] = run_pairwise_lingam(X)
        elif m == "Rank PC":
            out[m] = run_rank_pc(X)
        elif m == "Ngẫu nhiên":
            out[m] = run_random(X.shape[1], rng)
        else:
            raise ValueError(f"Không biết phương pháp {m!r}; chọn trong {METHODS}")
    return out


def evaluate(results: dict[str, dict], true_adj: np.ndarray, observed=None) -> dict[str, dict]:
    """SID chuẩn hoá (cả 5 phương pháp) và tỷ lệ xếp ngược tổ tiên–hậu duệ (chỉ phương pháp trả thứ tự).

    Khi có biến ẩn (``observed`` là tập con thật sự), SID được tính so với DAG tổ tiên trên các biến quan
    sát (``causal_tail.observed_ancestral_dag``) — cách "điều chỉnh nhẹ SID" như mục 5.1.1 của bài báo.
    """
    n_nodes = np.asarray(true_adj).shape[0]
    if observed is not None and len(observed) < n_nodes:
        true_adj, observed = CT.observed_ancestral_dag(true_adj, observed), None
    scores = {}
    for m, res in results.items():
        if res["order"] is None:
            s = CT.sid_cpdag(true_adj, res["adj"], observed)
            err = np.nan
        else:
            s = CT.sid(true_adj, res["adj"], observed)
            err = CT.ancestral_order_error(true_adj, res["order"], observed)
        scores[m] = {"SID": s, "sai_thu_tu": err}
    return scores


def _one_simulation(n, p, a, st, r, child_seed, methods, k_nu):
    rng = np.random.default_rng(child_seed)
    sim = CT.simulate_scm(n, p, a, st, rng)
    k = CT.default_k(n, k_nu)
    res = run_all(sim["X"], rng, k=k, methods=methods)
    return [{"n": n, "p": p, "alpha": a, "setting": st, "rep": r, "method": m, **sc}
            for m, sc in evaluate(res, sim["adj"], sim["observed"]).items()]


def simulation_study(ns=(500, 2000), ps=(5, 10), alphas=(1.5, 3.0), settings=CT.SETTINGS, reps: int = 20,
                     seed: int = 2026, methods: tuple[str, ...] = METHODS, progress: bool = False,
                     n_jobs: int = -1, k_nu: float = 0.4):
    """Lặp mô phỏng theo lưới (n, p, α, kịch bản) × ``reps``; trả DataFrame dạng dài, mỗi dòng một
    (lần lặp, phương pháp) với SID và tỷ lệ xếp ngược. k = ⌊n^ν⌋ (mặc định ν = 0,4 như bài báo).

    Chạy song song bằng joblib (``n_jobs`` tiến trình; -1 = mọi lõi). Mỗi lần lặp có seed con riêng sinh từ
    ``SeedSequence(seed)`` nên kết quả tái lập được và KHÔNG phụ thuộc số tiến trình hay thứ tự chạy.
    """
    import itertools
    import time

    import pandas as pd
    from joblib import Parallel, delayed

    grid = [(n, p, a, st, r) for n, p, a, st in itertools.product(ns, ps, alphas, settings) for r in range(reps)]
    children = np.random.SeedSequence(seed).spawn(len(grid))
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=10 if progress else 0)(
        delayed(_one_simulation)(*task, child, methods, k_nu) for task, child in zip(grid, children))
    if progress:
        print(f"{len(grid)} bộ dữ liệu × {len(methods)} phương pháp: {time.time() - t0:.0f}s")
    return pd.DataFrame([row for rows in out for row in rows])
