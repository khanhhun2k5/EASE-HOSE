"""Tải chỉ số VN-Index (biến thị trường quan sát được) cho các giai đoạn Validation và Calibration.

Từ khi mở rộng lên 20 mã, các mã vốn hoá lớn từng dùng dựng nhân tố "ngoài mẫu" đã nằm trong mẫu nghiên cứu,
nên nhân tố thị trường được dựng theo hai cách bổ sung cho nhau:
- **VNI**: lợi suất log VN-Index (file này tải) — đại diện chính thức, nhưng chứa chính 20 mã (gia quyền vốn hoá).
- **MKT_LOO**: với mỗi mã i, trung bình đều lợi suất của 19 mã còn lại (``validation.loo_market``) — không chứa i.

Cách chạy (vnstock 3.x, nguồn VCI, không cần khoá; xem ghi chú phiên bản trong download_vnstock.py):
    python src/download_market.py

Kết quả: data/raw/vnindex_ohlcv.csv (time, open, high, low, close, volume).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_SYMBOL = "VNINDEX"


def fetch(symbol: str, start: str, end: str, source: str = "VCI") -> pd.DataFrame:
    from vnstock import Quote

    df = Quote(source=source, symbol=symbol).history(start=start, end=end, interval="1D")
    return df[["time", "open", "high", "low", "close", "volume"]].copy()


def index_log_returns(index_ohlcv: pd.DataFrame, name: str = "VNI") -> pd.Series:
    """Lợi suất log ngày của chỉ số, chỉ mục theo ngày (phiên đầu bị bỏ)."""
    s = index_ohlcv.assign(time=pd.to_datetime(index_ohlcv["time"])).set_index("time")["close"].sort_index()
    if (s <= 0).any():
        raise ValueError("Giá đóng cửa của chỉ số phải dương.")
    return np.log(s / s.shift(1)).dropna().rename(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = fetch(INDEX_SYMBOL, args.start, args.end)
    idx.to_csv(OUTPUT_DIR / "vnindex_ohlcv.csv", index=False)
    print(f"{INDEX_SYMBOL}: {len(idx)} phiên, {idx['time'].min()} → {idx['time'].max()}")


if __name__ == "__main__":
    main()
