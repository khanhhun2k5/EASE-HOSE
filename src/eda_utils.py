"""Tiện ích dùng chung cho các notebook EDA: kết nối DuckDB, đọc truy vấn đặt tên,
và style hình theo chuẩn công bố khoa học (bảng màu + rcParams thống nhất)."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "interim" / "stocks.duckdb"
QUERY_FILE = PROJECT_ROOT / "sql" / "05_eda_queries.sql"
FIG_DIR = PROJECT_ROOT / "report" / "figure" / "eda"
TABLE_DIR = PROJECT_ROOT / "report" / "table" / "eda"

from config import ORIGINAL_SYMBOLS, SECTOR, SECTORS, SYMBOLS  # noqa: E402  (danh mục mã dùng chung)

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E", "gray_dark": "#4D4D4D",
}
# Mỗi ngành một họ màu, mỗi mã một sắc độ trong họ màu đó (cố định trong mọi hình).
SECTOR_CMAPS = {"Ngân hàng": "Blues", "Bất động sản": "Reds", "Tiêu dùng": "Greens", "Năng lượng": "Oranges",
                "Thép": "Purples", "Công nghệ": "PuRd", "Chứng khoán": "Greys"}
SECTOR_COLORS = {sec: mpl.colormaps[SECTOR_CMAPS.get(sec, "Greys")](0.75) for sec in SECTORS}


def _symbol_colors() -> dict:
    out = {}
    for sec in SECTORS:
        members = [s for s in SYMBOLS if SECTOR[s] == sec]
        cmap = mpl.colormaps[SECTOR_CMAPS.get(sec, "Greys")]
        shades = [0.75] if len(members) == 1 else list(np.linspace(0.45, 0.95, len(members)))
        out.update({s: mpl.colors.to_hex(cmap(v)) for s, v in zip(members, shades)})
    return out


SYMBOL_COLORS = _symbol_colors()


# ----------------------------------------------------------------------------- dữ liệu
def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Chưa có {DB_PATH}. Chạy: python src/build_database.py")
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def named_queries(path: Path = QUERY_FILE) -> dict[str, str]:
    """Tách file SQL thành {tên: câu truy vấn} theo dòng '-- name: <tên>'."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def query(con: duckdb.DuckDBPyConnection, sql_or_name: str) -> pd.DataFrame:
    sql = named_queries().get(sql_or_name, sql_or_name)
    return con.execute(sql).df()


def save_table(df: pd.DataFrame, name: str, folder: str | None = None) -> Path:
    """Lưu bảng vào report/table/eda (mặc định) hoặc report/table/<folder>."""
    out_dir = TABLE_DIR if folder is None else PROJECT_ROOT / "report" / "table" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, encoding="utf-8-sig")
    return path


# ----------------------------------------------------------------------------- hình vẽ
def apply_publication_style(font_size: int = 13, axes_linewidth: float = 1.5) -> None:
    mpl.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })


def make_heatmap(ax: plt.Axes, matrix: pd.DataFrame, cmap: str = "RdBu_r", vmin: float = -1,
                 vmax: float = 1, fmt: str | None = "{:.2f}", cbar_label: str | None = None,
                 title: str | None = None, annot_fontsize: float = 10, tick_fontsize: float | None = None):
    """Heatmap có chú thích giá trị trong ô (``fmt=None``: không ghi số — dùng cho ma trận lớn như 20 × 20);
    chữ tự đổi màu theo độ đậm của nền. ``tick_fontsize``: cỡ chữ nhãn trục (mặc định theo rcParams)."""
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    rot = 90 if matrix.shape[1] > 10 else 0
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=rot, fontsize=tick_fontsize)
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=tick_fontsize)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if pd.isna(val) or fmt is None:
                continue
            # Chữ trắng khi nền tối: tính độ sáng của đúng màu ô (đúng cho cả bảng màu phân kỳ lẫn tuần tự).
            r_, g_, b_, _ = im.cmap(im.norm(val))
            dark = 0.299 * r_ + 0.587 * g_ + 0.114 * b_ < 0.5
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=annot_fontsize,
                    color="white" if dark else "black")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    if cbar_label:
        cbar.set_label(cbar_label)
    if title:
        ax.set_title(title)
    return im


def finalize_figure(fig: plt.Figure, name: str, formats: tuple[str, ...] = ("png", "pdf"),
                    pad: float = 1.0, folder: str | None = None) -> list[Path]:
    """Căn bố cục và lưu hình vào report/figure/eda/<name>.<fmt> (hoặc report/figure/<folder>/)."""
    out_dir = FIG_DIR if folder is None else PROJECT_ROOT / "report" / "figure" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    return paths
