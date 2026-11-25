"""Suy luận nhân quả ở đuôi phân phối — thuật toán EASE (Gnecco, Meinshausen, Peters & Engelke, 2021,
"Causal discovery in heavy-tailed models", Ann. Statist. 49(3)) và công cụ đánh giá.

Nội dung:
- Hệ số nhân quả đuôi: Γ (đuôi trên, công thức (8) của bài báo), Γ⁻ (đuôi dưới = Γ của −X),
  Ψ (cả hai đuôi, công thức (10)–(11)).
- EASE: thứ tự nhân quả tham lam từ ma trận hệ số (mục 3.1).
- Γ quần thể của SCM tuyến tính đuôi dày (Lemma 1, công thức (6)) — dùng để kiểm thử.
- Khoảng cách can thiệp cấu trúc SID (Peters & Bühlmann, 2015) cho DAG và CPDAG.
- Sinh SCM tuyến tính đuôi dày để mô phỏng (có biến ẩn và kiểm duyệt kiểu biên độ giá).

Quy ước ma trận kề: ``A[i, j] = 1`` nghĩa là có cạnh i → j. Ma trận hệ số ``M[j, k]`` là hệ số
từ biến j tới biến k (ví dụ Γ_jk: "j gây ra k" thì Γ_jk = 1).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

COEF_KINDS = ("psi", "gamma_upper", "gamma_lower")


# ============================================================================ hạng & ngưỡng
def default_k(n: int, nu: float = 0.4) -> int:
    """Số quan sát vượt ngưỡng k_n = ⌊n^ν⌋ (bài báo chọn ν = 0,4). Luôn trong [1, n − 1]."""
    if n < 2:
        raise ValueError("Cần ít nhất 2 quan sát.")
    return int(min(max(np.floor(n ** nu), 1), n - 1))


def _ecdf_at_sample(x: np.ndarray) -> np.ndarray:
    """F̂(X_i) ≈ (1/n)·#{l : X_l ≤ X_i} (công thức (7)): (hạng trung bình − 1/2)/n.

    Với dữ liệu liên tục chỉ lệch công thức (7) một lượng 1/(2n). Khi có trùng (nhiều phiên cùng giá trần/sàn), hạng trung bình
    xử lý hai biên đối xứng: σ(F̂) = |2F̂ − 1| ở mức trần và mức sàn bằng nhau, nên Ψ̂ không thiên về đuôi nào.
    """
    return (stats.rankdata(x, method="average") - 0.5) / len(x)


def _extreme_index(x: np.ndarray, k: int, upper: bool, rng: np.random.Generator) -> np.ndarray:
    """Chỉ số của đúng k quan sát lớn nhất (upper) hoặc nhỏ nhất của x.

    Khi có giá trị trùng tại ngưỡng (rất phổ biến với lợi suất bị chặn ±7%: nhiều phiên cùng giá trần),
    định nghĩa 1{X_i > X_(n−k)} có thể cho ít hơn k điểm, thậm chí 0 điểm. Ta phá thế trùng ngẫu nhiên
    (seed cố định) để luôn có đúng k điểm — tương đương chọn ngẫu nhiên giữa các phiên cùng mức trần/sàn.
    """
    tie_breaker = rng.random(len(x))
    order = np.lexsort((tie_breaker, x))          # tăng dần theo x, trùng thì theo số ngẫu nhiên
    return order[-k:] if upper else order[:k]


# ============================================================================ hệ số nhân quả đuôi
def gamma_coef(x: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator | None = None) -> float:
    """Γ̂_{x→y} = (1/k) Σ_i F̂_y(Y_i)·1{X_i thuộc k giá trị lớn nhất của X} (công thức (8)).

    ≈ 1 nếu x gây ra y (mọi cú sốc cực trị của x lan sang y); ≈ 1/2 nếu độc lập.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    _check_pair(x, y, k)
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = _extreme_index(x, k, upper=True, rng=rng)
    return float(_ecdf_at_sample(y)[idx].mean())


def gamma_lower_coef(x: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator | None = None) -> float:
    """Γ̂⁻_{x→y}: hệ số nhân quả ở đuôi dưới (= Γ̂ áp dụng cho −x, −y). Dùng cho lan truyền cú sập giá."""
    return gamma_coef(-np.asarray(x, float), -np.asarray(y, float), k, rng)


def psi_coef(x: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator | None = None) -> float:
    """Ψ̂_{x→y} = (1/2k) [Σ_{X_i rất lớn} σ(F̂_y(Y_i)) + Σ_{X_i rất nhỏ} σ(F̂_y(Y_i))], σ(u) = |2u − 1|.

    Nhạy với cả hai đuôi và với hệ số âm (công thức (10)–(11), mục 4.1). Độc lập → 1/2.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    _check_pair(x, y, k)
    rng = rng if rng is not None else np.random.default_rng(0)
    sigma = np.abs(2 * _ecdf_at_sample(y) - 1)
    up = _extreme_index(x, k, upper=True, rng=rng)
    lo = _extreme_index(x, k, upper=False, rng=rng)
    return float(0.5 * sigma[up].mean() + 0.5 * sigma[lo].mean())


_COEF_FUN = {"psi": psi_coef, "gamma_upper": gamma_coef, "gamma_lower": gamma_lower_coef}


def _check_pair(x: np.ndarray, y: np.ndarray, k: int) -> None:
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x và y phải là mảng 1 chiều cùng độ dài.")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Dữ liệu có NaN — hãy bỏ các dòng thiếu trước.")
    if not 1 <= k < len(x):
        raise ValueError(f"k phải trong [1, n − 1], nhận k = {k}, n = {len(x)}.")


def tail_coef_matrix(X: np.ndarray | pd.DataFrame, k: int | None = None, kind: str = "psi",
                     seed: int = 0) -> pd.DataFrame | np.ndarray:
    """Ma trận M[j, l] = hệ số đuôi từ cột j tới cột l (đường chéo = NaN).

    Trả DataFrame (nhãn = tên cột) nếu đầu vào là DataFrame, ngược lại trả ndarray.
    """
    if kind not in _COEF_FUN:
        raise ValueError(f"kind phải thuộc {COEF_KINDS}")
    labels = list(X.columns) if isinstance(X, pd.DataFrame) else None
    A = np.asarray(X, float)
    if A.ndim != 2:
        raise ValueError("X phải là ma trận n × p.")
    n, p = A.shape
    k = default_k(n) if k is None else k
    if np.isnan(A).any():
        raise ValueError("Dữ liệu có NaN — hãy bỏ các dòng thiếu trước.")
    if not 1 <= k < n:
        raise ValueError(f"k phải trong [1, n − 1], nhận k = {k}, n = {n}.")
    # Cài đặt vector hoá, cho kết quả TRÙNG KHỚP với vòng lặp gamma_coef/psi_coef trên từng cặp:
    # hạng của mọi cột được tính một lần (thay vì p² lần); với mỗi nguyên nhân j, cùng luồng ngẫu nhiên
    # default_rng(seed + j) phá thế trùng như bản theo cặp. Độ phức tạp O(p·n log n + p²·k).
    F = (stats.rankdata(A, axis=0, method="average") - 0.5) / n
    if kind == "psi":
        T = np.abs(2 * F - 1)
    elif kind == "gamma_upper":
        T = F
    else:                                   # gamma_lower: F̂ của −Y = 1 − F̂ của Y (hạng trung bình đối xứng)
        T = 1 - F
    M = np.empty((p, p))
    for j in range(p):
        rng = np.random.default_rng(seed + j)
        x = A[:, j] if kind != "gamma_lower" else -A[:, j]
        up = np.lexsort((rng.random(n), x))[-k:]
        if kind == "psi":
            lo = np.lexsort((rng.random(n), x))[:k]
            M[j] = 0.5 * T[up].mean(axis=0) + 0.5 * T[lo].mean(axis=0)
        else:
            M[j] = T[up].mean(axis=0)
    np.fill_diagonal(M, np.nan)
    return pd.DataFrame(M, index=labels, columns=labels) if labels else M


# ============================================================================ EASE
def _incoming_score(G: np.ndarray, i: int, remaining: list) -> float:
    """M_i = max_{j còn lại, j ≠ i} G[j, i], bỏ qua NaN (hệ số không ước lượng được). Nếu mọi hệ số tới i đều NaN
    thì không có thông tin → điểm = 1 (bảo thủ: không chọn i làm gốc chỉ vì thiếu dữ liệu)."""
    vals = np.array([G[j, i] for j in remaining if j != i], float)
    vals = vals[~np.isnan(vals)]
    return float(vals.max()) if len(vals) else 1.0


def ease(coef: np.ndarray | pd.DataFrame) -> list:
    """Extremal Ancestral SEarch (mục 3.1). Trả thứ tự nhân quả (gốc trước).

    Ở bước s: điểm của nút i là M_i = max_{j còn lại, j≠i} coef[j, i] (mức "bị nút khác gây ra" lớn nhất);
    chọn nút có điểm nhỏ nhất làm gốc của đồ thị con còn lại. Độ phức tạp O(p²).
    """
    labels = list(coef.columns) if isinstance(coef, pd.DataFrame) else None
    G = np.asarray(coef, float)
    p = G.shape[0]
    if G.shape != (p, p):
        raise ValueError("Ma trận hệ số phải vuông.")
    remaining = list(range(p))
    order = []
    while remaining:
        if len(remaining) == 1:
            order.append(remaining.pop())
            break
        scores = [_incoming_score(G, i, remaining) for i in remaining]
        root = remaining[int(np.argmin(scores))]
        order.append(root)
        remaining.remove(root)
    return [labels[i] for i in order] if labels else order


def ease_scores_path(coef: pd.DataFrame) -> pd.DataFrame:
    """Lịch sử các bước EASE: ở mỗi bước, điểm M_i của từng nút còn lại (để giải thích kết quả)."""
    G = coef.to_numpy(float)
    names = list(coef.columns)
    remaining = list(range(len(names)))
    rows = []
    step = 1
    while len(remaining) > 1:
        scores = {names[i]: _incoming_score(G, i, remaining) for i in remaining}
        root = min(scores, key=scores.get)
        rows.append({"buoc": step, "chon": root, **scores})
        remaining.remove(names.index(root))
        step += 1
    rows.append({"buoc": step, "chon": names[remaining[0]]})
    return pd.DataFrame(rows).set_index("buoc")


def order_to_adjacency(order: Sequence[int], p: int | None = None) -> np.ndarray:
    """DAG đầy đủ ứng với thứ tự nhân quả: i → j với mọi i đứng trước j."""
    p = len(order) if p is None else p
    A = np.zeros((p, p), dtype=int)
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            A[i, j] = 1
    return A


# ============================================================================ Γ quần thể (Lemma 1)
def population_gamma(B: np.ndarray, alpha: float) -> np.ndarray:
    """Γ quần thể của SCM tuyến tính hệ số dương, nhiễu đuôi dày cùng chỉ số α (công thức (6)).

    ``B[j, h]`` = β_jh: hệ số của X_h trong phương trình của X_j (cạnh h → j).
    H = (I − B)⁻¹ thì H[j, h] = β_{h→j}; Γ_jk = 1/2 + 1/2 · Σ_{h∈An(j)∩An(k)} H[j,h]^α / Σ_{h∈An(j)} H[j,h]^α.
    """
    B = np.asarray(B, float)
    if (B < 0).any():
        raise ValueError("Lemma 1 cần hệ số không âm.")
    p = B.shape[0]
    H = np.linalg.inv(np.eye(p) - B)
    H[np.abs(H) < 1e-12] = 0.0
    anc = H > 0                     # anc[j, h]: h là tổ tiên của j (kể cả chính j)
    Ha = np.where(anc, H, 0.0) ** alpha
    G = np.full((p, p), np.nan)
    for j in range(p):
        for k in range(p):
            if j != k:
                common = anc[j] & anc[k]
                G[j, k] = 0.5 + 0.5 * Ha[j, common].sum() / Ha[j, anc[j]].sum()
    return G


# ============================================================================ SID
def _valid_adjustment(G: nx.DiGraph, x, y, Z: set) -> bool:
    """Tiêu chuẩn hiệu chỉnh tổng quát (Shpitser và cs., 2010): Z hợp lệ cho p(y | do(x)) trong DAG G?

    (i) Z không chứa nút "cấm": hậu duệ (kể cả chính nó) của các nút ≠ x nằm trên đường nhân quả x → … → y;
    (ii) Z d-tách x và y trong đồ thị đã bỏ cạnh đầu tiên của mọi đường nhân quả từ x tới y.
    """
    desc_x = nx.descendants(G, x)
    causal_nodes = {w for w in desc_x if w == y or y in nx.descendants(G, w)}
    forbidden = set()
    for w in causal_nodes:
        forbidden |= {w} | nx.descendants(G, w)
    if Z & forbidden:
        return False
    G_pbd = G.copy()
    G_pbd.remove_edges_from([(x, w) for w in causal_nodes if G.has_edge(x, w)])
    return nx.is_d_separator(G_pbd, {x}, {y}, set(Z))


def sid(true_adj: np.ndarray, est_adj: np.ndarray, observed: Sequence[int] | None = None,
        normalize: bool = True) -> float:
    """Structural Intervention Distance (Peters & Bühlmann, 2015) giữa DAG thật và DAG ước lượng.

    Với mỗi cặp (x, y), x ≠ y, DAG ước lượng Ĝ suy ra p(y | do(x)) bằng cách hiệu chỉnh theo pa_Ĝ(x).
    Cặp bị tính là SAI nếu: y ∈ pa_Ĝ(x) mà y là hậu duệ của x trong G (Ĝ nói "không có tác động"),
    hoặc y ∉ pa_Ĝ(x) và pa_Ĝ(x) không phải tập hiệu chỉnh hợp lệ trong G.

    ``observed``: chỉ số các nút quan sát được trong ``true_adj`` (khi có biến ẩn); ``est_adj`` khi đó có
    kích thước len(observed) và nút thứ a của nó ứng với nút observed[a] của G. ``normalize``: chia cho
    p(p − 1) để SID ∈ [0, 1] như bài báo.
    """
    T = np.asarray(true_adj, int)
    E = np.asarray(est_adj, int)
    obs = list(range(T.shape[0])) if observed is None else list(observed)
    if E.shape != (len(obs), len(obs)):
        raise ValueError("est_adj phải có kích thước len(observed) × len(observed).")
    G = nx.from_numpy_array(T, create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("true_adj không phải DAG.")
    p = len(obs)
    errors = 0
    for a in range(p):
        x = obs[a]
        Z = {obs[b] for b in range(p) if E[b, a]}
        desc_x = nx.descendants(G, x)
        for b in range(p):
            if a == b:
                continue
            y = obs[b]
            if y in Z:
                errors += int(y in desc_x)
            elif not _valid_adjustment(G, x, y, Z):
                errors += 1
    return errors / (p * (p - 1)) if normalize else float(errors)


def _v_structures(A: np.ndarray, skeleton: np.ndarray | None = None) -> set:
    """Tập v-structure a → c ← b (a, b không kề nhau). ``skeleton``: ma trận kề vô hướng dùng để xét "kề nhau"
    (mặc định là A ∪ Aᵀ); với CPDAG phải truyền toàn bộ khung kể cả cạnh vô hướng."""
    skel = (A | A.T) if skeleton is None else skeleton
    p = A.shape[0]
    out = set()
    for c in range(p):
        parents = np.flatnonzero(A[:, c])
        for a, b in itertools.combinations(parents, 2):
            if not skel[a, b]:
                out.add((min(a, b), c, max(a, b)))
    return out


def _orient_by_order(directed: np.ndarray, undirected: list, order: Sequence[int]) -> np.ndarray:
    pos = {v: i for i, v in enumerate(order)}
    A = directed.copy()
    for i, j in undirected:
        if pos[i] < pos[j]:
            A[i, j] = 1
        else:
            A[j, i] = 1
    return A


def _random_mcs_order(undirected: list, p: int, rng: np.random.Generator) -> list[int]:
    """Maximum cardinality search ngẫu nhiên trên phần vô hướng: mỗi bước chọn nút chưa thăm có nhiều láng giềng
    (vô hướng) đã thăm nhất, hoà thì chọn ngẫu nhiên. Định hướng cạnh vô hướng theo thứ tự này (trước → sau) cho
    mỗi thành phần chuỗi (chordal) của CPDAG một định hướng không sinh v-structure mới (Meek, 1995)."""
    nbrs = {v: set() for v in range(p)}
    for i, j in undirected:
        nbrs[i].add(j)
        nbrs[j].add(i)
    score = np.zeros(p, dtype=int)
    visited = np.zeros(p, dtype=bool)
    order = []
    for _ in range(p):
        cand = np.flatnonzero(~visited & (score == score[~visited].max()))
        v = int(rng.choice(cand))
        visited[v] = True
        order.append(v)
        for w in nbrs[v]:
            score[w] += 1
    return order


def cpdag_members(cpdag: np.ndarray, max_members: int = 64, seed: int = 0) -> list[np.ndarray]:
    """Các DAG (khác nhau) thuộc lớp tương đương Markov biểu diễn bởi CPDAG.

    ``cpdag[i, j] = 1`` và ``cpdag[j, i] = 0``: cạnh có hướng i → j; cả hai bằng 1: cạnh vô hướng.
    Thành viên của lớp = định hướng mọi cạnh vô hướng sao cho không có chu trình và tập v-structure (xét
    "kề nhau" trên toàn khung) đúng bằng của CPDAG.

    - Ít cạnh vô hướng (2^m ≤ 16·max_members): duyệt hết các định hướng.
    - Nhiều cạnh: định hướng cạnh vô hướng theo thứ tự maximum cardinality search ngẫu nhiên (không sinh
      v-structure mới trong thành phần chuỗi), giữ DAG không chu trình đúng tập v-structure, bỏ trùng.
    - CPDAG không nhất quán (PC có thể trả về: không định hướng nào giữ đúng v-structure): dùng các DAG không
      chu trình giữ hướng đã có như xấp xỉ. Phần có hướng có chu trình: trả về đồ thị rỗng.
    """
    C = np.asarray(cpdag, int)
    p = C.shape[0]
    directed = ((C == 1) & (C.T == 0)).astype(int)
    skeleton = ((C + C.T) > 0).astype(int)
    undirected = [(i, j) for i in range(p) for j in range(i + 1, p) if C[i, j] and C[j, i]]
    target_v = _v_structures(directed, skeleton)
    m = len(undirected)
    rng = np.random.default_rng(seed)

    def is_dag(A):
        return nx.is_directed_acyclic_graph(nx.from_numpy_array(A, create_using=nx.DiGraph))

    candidates = []
    if 2 ** m <= max_members * 16:
        for pat in itertools.product((0, 1), repeat=m):
            A = directed.copy()
            for (i, j), flip in zip(undirected, pat):
                A[(j, i) if flip else (i, j)] = 1
            if is_dag(A):
                candidates.append(A)
    elif is_dag(directed):
        seen = set()
        for _ in range(max_members * 16):
            A = _orient_by_order(directed, undirected, _random_mcs_order(undirected, p, rng))
            key = A.tobytes()
            if key not in seen and is_dag(A):
                seen.add(key)
                candidates.append(A)
    if not candidates:   # chu trình có hướng ngay trong phần đã định hướng
        return [np.zeros((p, p), dtype=int)]
    members = [A for A in candidates if _v_structures(A) == target_v]
    return (members or candidates)[:max_members]


def sid_cpdag(true_adj: np.ndarray, cpdag: np.ndarray, observed: Sequence[int] | None = None,
              max_members: int = 64) -> float:
    """SID trung bình trên các DAG của lớp tương đương (CPDAG, ví dụ đầu ra của Rank PC)."""
    return float(np.mean([sid(true_adj, A, observed) for A in cpdag_members(cpdag, max_members)]))


def observed_ancestral_dag(true_adj: np.ndarray, observed: Sequence[int]) -> np.ndarray:
    """DAG trên các nút quan sát: i → j nếu i là tổ tiên của j trong G (bỏ qua nhiễu ẩn).

    Dùng làm "đáp án" khi có biến ẩn (mục 4.2 của bài báo chỉ đòi khôi phục thứ tự nhân quả trên các
    biến quan sát, công thức (13)). Thứ tự đúng ⇒ SID = 0 với DAG này; nếu so với G gốc thì kể cả
    thứ tự đúng cũng bị tính sai vì không tập hiệu chỉnh quan sát được nào chặn được đường qua nút ẩn.
    """
    G = nx.from_numpy_array(np.asarray(true_adj, int), create_using=nx.DiGraph)
    obs = list(observed)
    A = np.zeros((len(obs), len(obs)), dtype=int)
    for a, i in enumerate(obs):
        desc = nx.descendants(G, i)
        for b, j in enumerate(obs):
            A[a, b] = int(j in desc)
    return A


def ancestral_order_error(true_adj: np.ndarray, order: Sequence[int], observed: Sequence[int] | None = None) -> float:
    """Tỷ lệ cặp (tổ tiên, hậu duệ) quan sát được mà thứ tự ước lượng xếp ngược. NaN nếu không có cặp nào."""
    G = nx.from_numpy_array(np.asarray(true_adj, int), create_using=nx.DiGraph)
    obs = list(range(G.number_of_nodes())) if observed is None else list(observed)
    if sorted(int(o) for o in order) != list(range(len(obs))):
        raise ValueError("order phải là hoán vị của 0..len(observed) − 1.")
    pos = {obs[node]: r for r, node in enumerate(order)}
    pairs = [(a, d) for a in obs for d in nx.descendants(G, a) if d in pos]
    if not pairs:
        return float("nan")
    return float(np.mean([pos[a] > pos[d] for a, d in pairs]))


# ============================================================================ mô phỏng
SETTINGS = ("linear", "hidden", "censored", "hidden_censored")


def random_dag(p: int, rng: np.random.Generator, edge_prob: float | None = None) -> np.ndarray:
    """DAG Erdős–Rényi theo một thứ tự ngẫu nhiên; mặc định xác suất cạnh 2/(p−1) (≈ p cạnh)."""
    edge_prob = min(1.0, 2 / max(p - 1, 1)) if edge_prob is None else edge_prob
    perm = rng.permutation(p)
    A = np.zeros((p, p), dtype=int)
    for a in range(p):
        for b in range(a + 1, p):
            if rng.random() < edge_prob:
                A[perm[a], perm[b]] = 1
    return A


def simulate_scm(n: int, p: int, alpha: float, setting: str = "linear", rng: np.random.Generator | None = None,
                 n_hidden: int = 2, censor_prob: float = 0.015) -> dict:
    """Sinh dữ liệu từ SCM tuyến tính, nhiễu Student-t bậc tự do α (chỉ số đuôi α), hệ số thực.

    - ``linear``: X = BᵀX + ε, |β| ~ U(0,3; 0,9), dấu ngẫu nhiên.
    - ``hidden``: thêm ``n_hidden`` nhân tố ẩn gốc (như "thị trường chung"), mỗi nhân tố tác động dương
      lên từng biến quan sát với xác suất 0,5; chỉ trả về các biến quan sát.
    - ``censored``: sau khi sinh, chặn mỗi biến tại phân vị ``censor_prob`` và 1 − ``censor_prob``
      (mô phỏng biên độ giá ±7% của HOSE: ~1,5% số phiên chạm trần, ~1,5% chạm sàn).
    - ``hidden_censored``: cả hai — gần nhất với dữ liệu cổ phiếu thật.

    Trả dict: X (n × p), adj (DAG đầy đủ gồm cả nút ẩn), observed (chỉ số nút quan sát trong adj).
    """
    if setting not in SETTINGS:
        raise ValueError(f"setting phải thuộc {SETTINGS}")
    rng = rng if rng is not None else np.random.default_rng()
    hidden = setting.startswith("hidden")
    h = n_hidden if hidden else 0
    A_obs = random_dag(p, rng)
    q = p + h
    A = np.zeros((q, q), dtype=int)
    A[h:, h:] = A_obs
    for f in range(h):
        A[f, h:] = (rng.random(p) < 0.5).astype(int)
    W = A * rng.uniform(0.3, 0.9, (q, q))
    W[h:, h:] *= rng.choice([-1, 1], (p, p))       # hệ số giữa biến quan sát có thể âm; nhân tố ẩn dương
    order = list(nx.topological_sort(nx.from_numpy_array(A, create_using=nx.DiGraph)))
    eps = rng.standard_t(alpha, (n, q))
    X = np.zeros((n, q))
    for j in order:
        X[:, j] = X @ W[:, j] + eps[:, j]
    X = X[:, h:]
    if setting.endswith("censored"):
        lo, hi = np.quantile(X, [censor_prob, 1 - censor_prob], axis=0)
        X = np.clip(X, lo, hi)
    return {"X": X, "adj": A, "observed": list(range(h, q))}
