"""Cấu hình dùng chung: danh mục mã nghiên cứu (nguồn duy nhất là data/reference/symbol_meta.csv).

- ``SYMBOLS``: 60 mã HOSE theo thứ tự ``sort_order`` (nhóm theo ngành): 20 mã vốn hoá lớn ban đầu và 40 mã bổ sung
  chọn bằng ``download_extended.select_universe`` (26/11/2026).
- ``ORIGINAL_SYMBOLS``: 5 mã của giai đoạn đầu; ``BASE_SYMBOLS``: 20 mã của giai đoạn thứ hai — giữ để so sánh kết quả cũ – mới.
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
# 20 mã vốn hoá lớn của giai đoạn trước khi mở rộng lên 60 mã (26/11/2026) — dùng để đối chiếu kết quả cũ – mới
# và để download_extended loại trừ khi chọn 40 mã bổ sung.
BASE_SYMBOLS = ["VCB", "BID", "CTG", "TCB", "MBB", "STB", "VPB", "VIC", "VHM", "NVL", "KDH", "VNM", "MSN", "SAB",
                "MWG", "GAS", "PLX", "HPG", "FPT", "SSI"]


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
