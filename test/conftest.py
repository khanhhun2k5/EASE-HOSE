"""Fixture dùng chung: dữ liệu giả lập nhỏ, biết trước đáp án, và CSDL DuckDB dựng từ sql/01-04.

Không test nào gọi mạng hay đụng vào data/ thật của đồ án.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # vẽ không cần màn hình

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 5 mã giả lập, theo sort_order của danh mục thật (thứ tự cột của v_returns_wide).
SYMBOLS = ["STB", "VPB", "VHM", "NVL", "HPG"]
VHM_LISTING = pd.Timestamp("2018-05-17")
CEILING_DAY = pd.Timestamp("2018-07-02")   # HPG tăng trần, đóng cửa = giá cao nhất
FLOOR_DAY = pd.Timestamp("2018-07-03")     # STB giảm sàn, đóng cửa = giá thấp nhất
DUP_DAY = pd.Timestamp("2018-07-04")       # NVL tăng 6,8% nhưng close < high -> KHÔNG chạm trần


def make_synthetic_ohlcv() -> pd.DataFrame:
    """5 mã, phiên ngày thường 02/04/2018 -> 28/09/2018.

    - VHM trước 17/05/2018: giá đứng 8,41, khối lượng 0 (mô phỏng giai đoạn chưa niêm yết).
    - Ngày 17/05/2018 VHM nhảy lên 30,99 (lợi suất "ảo" +268% phải bị loại).
    - Các mã còn lại: bước ngẫu nhiên có seed cố định, |r| < 3%.
    """
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2018-04-02", "2018-09-28")
    rows = []
    for k, sym in enumerate(SYMBOLS):
        price = 20.0 + 5 * k
        for i, d in enumerate(days):
            if sym == "VHM" and d < VHM_LISTING:
                rows.append((sym, d, 8.41, 8.41, 8.41, 8.41, 0))
                continue
            if sym == "VHM" and d == VHM_LISTING:
                price = 30.99
                rows.append((sym, d, 30.99, 30.99, 30.99, 30.99, 18000))
                continue
            r = rng.uniform(-0.03, 0.03)
            if sym == "HPG" and d == CEILING_DAY:
                r = 0.068
            if sym == "STB" and d == FLOOR_DAY:
                r = -0.069
            if sym == "NVL" and d == DUP_DAY:
                r = 0.068
            prev, price = price, round(price * (1 + r), 2)
            high, low = max(prev, price), min(prev, price)
            if sym == "NVL" and d == DUP_DAY:
                high = price + 0.2          # đóng cửa dưới giá cao nhất
            vol = 1_000_000 + 10_000 * i + 1000 * k
            rows.append((sym, d, prev, high, low, price, vol))
    df = pd.DataFrame(rows, columns=["symbol", "time", "open", "high", "low", "close", "volume"])
    df["exchange"] = "HOSE"
    return df


def write_symbol_meta(root: Path, symbols=None) -> Path:
    """Ghi <root>/data/reference/symbol_meta.csv: các dòng của danh mục thật ứng với ``symbols``."""
    import config
    meta = config.load_symbol_meta()
    meta = meta[meta.symbol.isin(symbols or SYMBOLS)]
    ref = root / "data" / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    meta.to_csv(ref / "symbol_meta.csv", index=False)
    return ref / "symbol_meta.csv"


def build_db(root: Path, df: pd.DataFrame, db_name: str = "test.duckdb") -> Path:
    """Ghi CSV vào <root>/data/raw, danh mục vào <root>/data/reference rồi dựng CSDL như build_database.build."""
    import build_database
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw / "hose_ohlcv_long.csv", index=False)
    write_symbol_meta(root, sorted(df.symbol.unique()))
    db_path = root / db_name
    cwd = os.getcwd()
    os.chdir(root)
    try:
        with duckdb.connect(str(db_path)) as con:
            build_database.build(con)
    finally:
        os.chdir(cwd)
    return db_path


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_ohlcv()


@pytest.fixture(scope="session")
def db_path(tmp_path_factory, synthetic_df) -> Path:
    return build_db(tmp_path_factory.mktemp("db"), synthetic_df)


@pytest.fixture()
def con(db_path):
    with duckdb.connect(str(db_path), read_only=True) as c:
        yield c
