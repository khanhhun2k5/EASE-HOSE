"""Kiểm thử EASE, hệ số nhân quả đuôi, SID và các baseline (dữ liệu giả lập, biết trước đáp án)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import causal_baselines as CB
import causal_tail as CT


def _chain_adj(p: int) -> np.ndarray:
    A = np.zeros((p, p), dtype=int)
    for i in range(p - 1):
        A[i, i + 1] = 1
    return A


# ----------------------------------------------------------------------------- k
@pytest.mark.parametrize("n, expected", [(2, 1), (1000, 15), (2000, 20), (10_000, 39)])
def test_default_k_is_floor_n_pow_04(n, expected):
    assert CT.default_k(n) == expected


def test_default_k_rejects_tiny_sample():
    with pytest.raises(ValueError):
        CT.default_k(1)


# ----------------------------------------------------------------------------- hệ số
def test_gamma_detects_direction_of_heavy_tailed_cause():
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.standard_t(1.5, n)
    y = x + rng.standard_t(1.5, n)
    k = CT.default_k(n)
    g_xy, g_yx = CT.gamma_coef(x, y, k), CT.gamma_coef(y, x, k)
    assert g_xy > 0.97            # x gây ra y → Γ_xy ≈ 1
    assert g_yx < g_xy - 0.05     # ngược lại < 1 (Theorem 1)


def test_independent_variables_give_one_half():
    rng = np.random.default_rng(2)
    x, y = rng.standard_t(2, (2, 50_000))
    k = 500
    for f in (CT.gamma_coef, CT.gamma_lower_coef, CT.psi_coef):
        assert abs(f(x, y, k) - 0.5) < 0.05


def test_psi_detects_negative_coefficient_but_gamma_upper_does_not():
    rng = np.random.default_rng(3)
    n = 20_000
    x = rng.standard_t(1.5, n)
    y = -x + 0.3 * rng.standard_t(1.5, n)
    k = CT.default_k(n)
    assert CT.psi_coef(x, y, k) > 0.95
    assert CT.gamma_coef(x, y, k) < 0.1   # x rất lớn → y rất nhỏ


def test_coefficients_invariant_to_monotone_transform():
    rng = np.random.default_rng(4)
    x = rng.standard_t(2, 3000)
    y = 0.7 * x + rng.standard_t(2, 3000)
    k = 40
    for f in (CT.gamma_coef, CT.psi_coef):
        a = f(x, y, k, np.random.default_rng(0))
        b = f(np.exp(x / 10), y ** 3, k, np.random.default_rng(0))
        assert a == pytest.approx(b)


def test_ties_at_censoring_bound_still_use_exactly_k_points():
    """Mô phỏng trần ±7%: 5% số điểm trùng đúng giá trị trần, nhiều hơn k → vẫn trả giá trị hữu hạn."""
    rng = np.random.default_rng(5)
    x = rng.standard_t(2, 2000)
    y = x + rng.standard_t(2, 2000)
    lo, hi = np.quantile(x, [0.05, 0.95])
    xc = np.clip(x, lo, hi)
    k = CT.default_k(2000)
    assert (xc == hi).sum() > k
    idx = CT._extreme_index(xc, k, upper=True, rng=np.random.default_rng(0))
    assert len(idx) == k and np.all(xc[idx] == hi)
    assert 0.5 < CT.gamma_coef(xc, y, k) <= 1


@pytest.mark.parametrize("bad", [
    dict(x=np.arange(5.0), y=np.arange(4.0), k=2),
    dict(x=np.arange(5.0), y=np.arange(5.0), k=0),
    dict(x=np.arange(5.0), y=np.arange(5.0), k=5),
    dict(x=np.array([1.0, np.nan, 2, 3]), y=np.arange(4.0), k=1),
])
def test_coefficient_input_validation(bad):
    with pytest.raises(ValueError):
        CT.gamma_coef(**bad)


def test_tail_coef_matrix_labels_and_diagonal():
    rng = np.random.default_rng(6)
    df = pd.DataFrame(rng.standard_t(3, (500, 3)), columns=["VHM", "NVL", "Cổ phiếu Á"])
    M = CT.tail_coef_matrix(df, kind="psi")
    assert list(M.index) == list(M.columns) == list(df.columns)
    assert np.isnan(np.diag(M.values)).all()
    assert ((M.values[~np.eye(3, dtype=bool)] >= 0) & (M.values[~np.eye(3, dtype=bool)] <= 1)).all()
    with pytest.raises(ValueError):
        CT.tail_coef_matrix(df, kind="khac")


# ----------------------------------------------------------------------------- Γ quần thể & EASE
def test_population_gamma_diamond_example():
    """Ví dụ 1 của bài báo: đồ thị kim cương 1→2,1→3,2→4,3→4 → Γ_14 = 1 > Γ_41."""
    B = np.zeros((4, 4))
    B[1, 0], B[2, 0], B[3, 1], B[3, 2] = 0.8, 0.5, 0.7, 0.6   # B[j, h]: cạnh h → j
    G = CT.population_gamma(B, alpha=1.5)
    assert G[0, 3] == pytest.approx(1.0)
    b14 = 0.7 * 0.8 + 0.6 * 0.5
    expected_41 = 0.5 + 0.5 * b14 ** 1.5 / (b14 ** 1.5 + 0.7 ** 1.5 + 0.6 ** 1.5 + 1)
    assert G[3, 0] == pytest.approx(expected_41)
    assert G[1, 2] < 1 and G[2, 1] < 1                       # nguyên nhân chung, không nhân quả trực tiếp
    order = CT.ease(G)
    assert order[0] == 0 and order[-1] == 3


@pytest.mark.parametrize("seed", range(5))
def test_ease_recovers_causal_order_in_population(seed):
    """Mệnh đề 1: với Γ quần thể, EASE luôn trả một thứ tự nhân quả đúng."""
    rng = np.random.default_rng(seed)
    A = CT.random_dag(7, rng)
    B = (A * rng.uniform(0.2, 1.0, A.shape)).T
    order = CT.ease(CT.population_gamma(B, alpha=2.0))
    err = CT.ancestral_order_error(A, order)
    assert err == 0.0 or np.isnan(err)
    assert CT.sid(A, CT.order_to_adjacency(order)) == 0


def test_ease_keeps_labels_and_score_path():
    G = pd.DataFrame(CT.population_gamma(np.array([[0, 0], [0.9, 0]]), 2.0), index=["a", "b"], columns=["a", "b"])
    assert CT.ease(G) == ["a", "b"]
    path = CT.ease_scores_path(G)
    assert list(path["chon"]) == ["a", "b"]


def test_ease_on_simulated_sample_mostly_correct():
    rng = np.random.default_rng(7)
    sim = CT.simulate_scm(5000, 5, 1.5, "linear", rng)
    order = CB.run_ease(sim["X"])["order"]
    assert CT.sid(sim["adj"], CT.order_to_adjacency(order)) <= 0.2


# ----------------------------------------------------------------------------- SID
def test_sid_identical_and_reversed_chain():
    A = _chain_adj(3)
    assert CT.sid(A, A) == 0
    assert CT.sid(A, CT.order_to_adjacency([0, 1, 2])) == 0      # DAG đầy đủ theo thứ tự đúng
    assert CT.sid(A, CT.order_to_adjacency([2, 1, 0])) == 1      # đảo ngược: sai cả 6 cặp (tính tay)
    assert CT.sid(A, CT.order_to_adjacency([2, 1, 0]), normalize=False) == 6


def test_sid_empty_truth_any_order_is_correct():
    assert CT.sid(np.zeros((4, 4), int), CT.order_to_adjacency([3, 1, 0, 2])) == 0


def test_sid_with_hidden_confounder():
    """Nút 0 (ẩn) → nút 1, nút 0 → nút 2; quan sát nút 1, 2 (không nhân quả trực tiếp).

    Ĝ: 1 → 2. Cặp (do(1), 2): Ĝ ước p(x2 | x1) nhưng đúng là p(x2) — sai do nhiễu ẩn mở đường 1 ← 0 → 2.
    Cặp (do(2), 1): Ĝ nói "không tác động" — đúng. Tổng: 1 cặp sai."""
    A = np.zeros((3, 3), int)
    A[0, 1] = A[0, 2] = 1
    assert CT.sid(A, CT.order_to_adjacency([0, 1]), observed=[1, 2], normalize=False) == 1


def test_sid_rejects_bad_shapes_and_cycles():
    with pytest.raises(ValueError):
        CT.sid(_chain_adj(3), np.zeros((2, 2), int))
    cyc = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        CT.sid(cyc, np.zeros((2, 2), int))


def test_cpdag_members_of_undirected_chain():
    C = np.zeros((3, 3), int)
    C[0, 1] = C[1, 0] = C[1, 2] = C[2, 1] = 1
    members = CT.cpdag_members(C)
    assert len(members) == 3                 # 0-1-2 không có v-structure → 3 DAG
    assert CT.sid_cpdag(_chain_adj(3), C) == pytest.approx(np.mean([CT.sid(_chain_adj(3), m) for m in members]))


def test_cpdag_with_v_structure_is_single_member():
    C = np.zeros((3, 3), int)
    C[0, 2] = C[1, 2] = 1
    assert len(CT.cpdag_members(C)) == 1


# ----------------------------------------------------------------------------- mô phỏng
@pytest.mark.parametrize("setting", CT.SETTINGS)
def test_simulate_scm_shapes_and_censoring(setting):
    sim = CT.simulate_scm(1000, 4, 2.0, setting, np.random.default_rng(0))
    assert sim["X"].shape == (1000, 4)
    assert len(sim["observed"]) == 4
    assert sim["adj"].shape[0] == 4 + (2 if setting.startswith("hidden") else 0)
    if setting.endswith("censored"):
        top_share = (sim["X"] == sim["X"].max(axis=0)).mean(axis=0)
        assert np.all(top_share >= 0.01)


def test_simulate_scm_rejects_unknown_setting():
    with pytest.raises(ValueError):
        CT.simulate_scm(100, 3, 2.0, "khong_co")


# ----------------------------------------------------------------------------- baseline
def test_all_baselines_run_and_score_in_unit_interval():
    rng = np.random.default_rng(8)
    sim = CT.simulate_scm(800, 5, 2.0, "hidden_censored", rng)
    res = CB.run_all(sim["X"], rng)
    assert set(res) == set(CB.METHODS)
    scores = CB.evaluate(res, sim["adj"], sim["observed"])
    for m, s in scores.items():
        assert 0 <= s["SID"] <= 1
        if res[m]["order"] is not None:
            assert sorted(res[m]["order"]) == list(range(5))


def test_rank_pc_returns_symmetric_undirected_part():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((2000, 3))
    X[:, 1] += X[:, 0]
    X[:, 2] += X[:, 1]
    C = CB.run_rank_pc(X)["adj"]
    assert C[0, 1] + C[1, 0] >= 1 and C[1, 2] + C[2, 1] >= 1
    assert C[0, 2] == 0 and C[2, 0] == 0     # 0 ⊥ 2 | 1


def test_run_all_rejects_unknown_method():
    with pytest.raises(ValueError):
        CB.run_all(np.zeros((10, 2)), np.random.default_rng(0), methods=("XYZ",))


# ----------------------------------------------------------------------------- hồi quy sau code review
def _is_member(A, C):
    """A là DAG, cùng khung với C, giữ hướng đã có và đúng tập v-structure (tính trên khung)."""
    import networkx as nx
    skel = ((C + C.T) > 0).astype(int)
    directed = ((C == 1) & (C.T == 0)).astype(int)
    return (nx.is_directed_acyclic_graph(nx.from_numpy_array(A, create_using=nx.DiGraph))
            and np.array_equal((A + A.T) > 0, skel > 0) and np.all(A[directed == 1] == 1)
            and CT._v_structures(A) == CT._v_structures(directed, skel))


def test_cpdag_members_sampling_branch_returns_only_true_members():
    """Chuỗi 12 nút toàn cạnh vô hướng (nhánh lấy mẫu): mọi DAG trả về phải thuộc lớp (không có v-structure)."""
    p = 12
    C = np.zeros((p, p), int)
    for i in range(p - 1):
        C[i, i + 1] = C[i + 1, i] = 1
    members = CT.cpdag_members(C, max_members=4)
    assert 1 <= len(members) <= 4
    assert len({m.tobytes() for m in members}) == len(members)          # không trùng
    assert all(_is_member(m, C) for m in members)


def test_cpdag_members_undirected_edge_between_parents_is_not_a_v_structure():
    """3→2, 0→2, 1→2, 0—1 vô hướng và chuỗi 4—5—6: lớp có đúng 2 × 3 = 6 DAG."""
    C = np.zeros((7, 7), int)
    C[3, 2] = C[0, 2] = C[1, 2] = 1
    C[0, 1] = C[1, 0] = 1
    C[4, 5] = C[5, 4] = C[5, 6] = C[6, 5] = 1
    members = CT.cpdag_members(C)
    assert len(members) == 6
    assert all(_is_member(m, C) for m in members)


def test_ancestral_order_error_rejects_incomplete_order():
    with pytest.raises(ValueError):
        CT.ancestral_order_error(_chain_adj(3), [0, 2])


def test_ecdf_ties_symmetric_at_both_bounds():
    """Cùng số phiên trùng ở trần và ở sàn → σ(F̂) như nhau ở hai biên."""
    x = np.r_[np.full(10, -1.0), np.linspace(-0.5, 0.5, 80), np.full(10, 1.0)]
    sig = np.abs(2 * CT._ecdf_at_sample(x) - 1)
    assert sig[0] == pytest.approx(sig[-1])



@pytest.mark.parametrize("kind", CT.COEF_KINDS)
def test_vectorised_matrix_equals_pairwise_loop(kind):
    """Bản vector hoá phải trùng khớp tuyệt đối với vòng lặp theo cặp (kể cả khi có giá trị trùng ở biên)."""
    rng = np.random.default_rng(12)
    X = rng.standard_t(3, (600, 6))
    X[:30, 2] = X[:, 2].max()                    # 30 phiên "trần" trùng giá trị
    X[:25, 4] = X[:, 4].min()
    M = CT.tail_coef_matrix(X, k=20, kind=kind, seed=9)
    fun = CT._COEF_FUN[kind]
    for j in range(6):
        for l in range(6):
            if j != l:
                assert M[j, l] == pytest.approx(fun(X[:, j], X[:, l], 20, np.random.default_rng(9 + j)), abs=1e-12)


def test_tail_coef_matrix_input_validation():
    with pytest.raises(ValueError):
        CT.tail_coef_matrix(np.zeros(10))
    with pytest.raises(ValueError):
        CT.tail_coef_matrix(np.array([[1.0, np.nan], [2.0, 3.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        CT.tail_coef_matrix(np.zeros((5, 2)), k=5)


def test_simulation_study_reproducible_regardless_of_jobs():
    import causal_baselines as CB
    kw = dict(ns=(300,), ps=(4,), alphas=(2.0,), settings=("linear",), reps=3, methods=("EASE", "Ngẫu nhiên"))
    a = CB.simulation_study(**kw, n_jobs=1)
    b = CB.simulation_study(**kw, n_jobs=2)
    pd.testing.assert_frame_equal(a, b)



def test_ica_lingam_maps_permuted_columns_back():
    """Thứ tự ICA-LiNGAM trả về theo chỉ số cột GỐC dù đã xáo trộn cột bên trong."""
    rng = np.random.default_rng(13)
    n = 4000
    e = rng.uniform(-1, 1, (n, 3)) ** 3                    # nhiễu phi Gauss mạnh
    X = np.empty_like(e)
    X[:, 2] = e[:, 2]; X[:, 0] = 0.9 * X[:, 2] + e[:, 0]; X[:, 1] = 0.9 * X[:, 0] + e[:, 1]   # 2 → 0 → 1
    for s in range(3):
        assert CB.run_ica_lingam(X, seed=s)["order"] == [2, 0, 1]
    order, pos = CB.ica_lingam_consensus(X, seeds=range(4))
    assert order == [2, 0, 1] and pos.shape == (4, 3)
