"""Kiểm tra src/eda_utils.py: đọc truy vấn đặt tên, kết nối, lưu bảng, vẽ hình."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eda_utils as U


# ----------------------------------------------------------------------------- truy vấn đặt tên
def test_named_queries_parses_names_and_strips_semicolon(tmp_path):
    f = tmp_path / "q.sql"
    f.write_text(
        "-- tiêu đề, không thuộc truy vấn nào\n"
        "-- name: first\nSELECT 1 AS a;\n\n"
        "-- name: second_q\n-- chú thích tiếng Việt có dấu\nSELECT 2 AS b\nFROM range(1);\n",
        encoding="utf-8")
    q = U.named_queries(f)
    assert list(q) == ["first", "second_q"]
    assert q["first"] == "SELECT 1 AS a"
    assert q["second_q"].endswith("FROM range(1)")
    assert "chú thích" in q["second_q"]


def test_named_queries_empty_file(tmp_path):
    f = tmp_path / "empty.sql"
    f.write_text("-- không có truy vấn nào\n", encoding="utf-8")
    assert U.named_queries(f) == {}


def test_project_query_file_has_expected_names():
    assert {"profile_by_symbol", "missing_sessions", "return_summary", "worst_days"} <= set(U.named_queries())


def test_query_accepts_name_or_raw_sql(con):
    by_name = U.query(con, "return_summary")
    raw = U.query(con, U.named_queries()["return_summary"])
    pd.testing.assert_frame_equal(by_name, raw)
    assert U.query(con, "SELECT 42 AS x").x.iloc[0] == 42


# ----------------------------------------------------------------------------- kết nối
def test_connect_missing_db_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "DB_PATH", tmp_path / "khong_co.duckdb")
    with pytest.raises(FileNotFoundError, match="build_database"):
        U.connect()


def test_connect_read_only(monkeypatch, db_path):
    monkeypatch.setattr(U, "DB_PATH", db_path)
    with U.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM symbol_meta").fetchone()[0] == 5
        with pytest.raises(Exception):
            c.execute("CREATE TABLE x (a INT)")


# ----------------------------------------------------------------------------- lưu bảng
def test_save_table_utf8_bom_and_vietnamese(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "TABLE_DIR", tmp_path / "bang")
    df = pd.DataFrame({"giá trị": [1.5, 2.0]}, index=pd.Index(["Bất động sản", "Ngân hàng"], name="ngành"))
    path = U.save_table(df, "thu")
    assert path == tmp_path / "bang" / "thu.csv"
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")          # BOM để Excel đọc đúng tiếng Việt
    back = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(back, df)


# ----------------------------------------------------------------------------- hình vẽ
def test_apply_publication_style_sets_rcparams():
    with mpl.rc_context():
        U.apply_publication_style(font_size=11, axes_linewidth=2)
        assert mpl.rcParams["font.size"] == 11
        assert mpl.rcParams["axes.linewidth"] == 2
        assert not mpl.rcParams["axes.spines.top"] and not mpl.rcParams["axes.spines.right"]
        assert not mpl.rcParams["legend.frameon"]
        assert mpl.rcParams["savefig.dpi"] == 300


def test_symbol_colors_unique_and_in_palette():
    assert list(U.SYMBOL_COLORS) == U.SYMBOLS
    assert len(set(U.SYMBOL_COLORS.values())) == len(U.SYMBOLS)
    import matplotlib.colors as mcolors
    assert all(mcolors.is_color_like(c) for c in U.SYMBOL_COLORS.values())
    for sec in U.SECTORS:                                             # cùng ngành -> cùng họ màu (bảng màu ngành)
        assert sec in U.SECTOR_CMAPS


def test_make_heatmap_annotates_cells_and_skips_nan():
    m = pd.DataFrame([[1.0, np.nan], [0.25, -1.0]], index=["A", "B"], columns=["A", "B"])
    fig, ax = plt.subplots()
    U.make_heatmap(ax, m, title="Thử", cbar_label="ρ")
    texts = [t.get_text() for t in ax.texts]
    assert sorted(texts) == sorted(["1.00", "0.25", "-1.00"])     # ô NaN không ghi chữ
    assert ax.get_title() == "Thử"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
    # chữ trắng trên nền đậm (|giá trị| lớn), chữ đen trên nền nhạt
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["1.00"] == "white" and colors["0.25"] == "black"
    plt.close(fig)


def test_finalize_figure_writes_requested_formats(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "FIG_DIR", tmp_path / "hinh")
    monkeypatch.setattr(U, "PDF_DIR", tmp_path / "pdf")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = U.finalize_figure(fig, "duong_thang", formats=("png", "pdf"))
    plt.close(fig)
    assert [p.name for p in paths] == ["duong_thang.png", "duong_thang.pdf"]
    # png ở thư mục hình, pdf tách sang thư mục pdf riêng
    assert paths[0].parent == tmp_path / "hinh" and paths[1].parent == tmp_path / "pdf"
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert paths[0].read_bytes()[:4] == b"\x89PNG"
