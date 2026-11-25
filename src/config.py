"""Cấu hình dùng chung: danh mục mã nghiên cứu (nguồn duy nhất là data/reference/symbol_meta.csv).

- ``SYMBOLS``: 20 mã HOSE theo thứ tự ``sort_order`` (nhóm theo ngành).
- ``ORIGINAL_SYMBOLS``: 5 mã của giai đoạn đầu, giữ để so sánh kết quả cũ – mới.
- ``SECTOR``: {mã: ngành}; ``SECTORS``: danh sách ngành theo thứ tự xuất hiện.

``listing_date`` để trống trong CSV nghĩa là: lấy ngày đầu tiên có dữ liệu (dữ liệu nguồn VCI bắt đầu từ
18/06/2014, sau ngày niêm yết thật của các mã này, nên không có giá "ảo" trước niêm yết). Chỉ ghi rõ ngày
khi cần loại giai đoạn giá ảo (VHM) hoặc khi mã niêm yết trong khoảng có dữ liệu.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL_META_CSV = PROJECT_ROOT / "data" / "reference" / "symbol_meta.csv"
ORIGINAL_SYMBOLS = ["VHM", "NVL", "STB", "VPB", "HPG"]


def load_symbol_meta(path: Path = SYMBOL_META_CSV) -> pd.DataFrame:
    """Đọc danh mục mã, kiểm tra khoá duy nhất và thứ tự, trả DataFrame sắp theo sort_order."""
    meta = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8")
    required = {"symbol", "company_name", "sector", "exchange", "listing_date", "sort_order"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"symbol_meta.csv thiếu cột: {sorted(missing)}")
    if meta["symbol"].duplicated().any():
        raise ValueError("symbol_meta.csv có mã trùng.")
    if meta["sort_order"].duplicated().any():
        raise ValueError("symbol_meta.csv có sort_order trùng.")
    return meta.sort_values("sort_order").reset_index(drop=True)


_META = load_symbol_meta()
SYMBOLS: list[str] = _META["symbol"].tolist()
SECTOR: dict[str, str] = dict(zip(_META["symbol"], _META["sector"]))
SECTORS: list[str] = list(dict.fromkeys(_META["sector"]))
