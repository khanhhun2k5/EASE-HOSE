"""Dựng CSDL DuckDB từ dữ liệu thô bằng các file trong sql/ và đo tốc độ truy vấn.

Chạy từ thư mục gốc đồ án:
    python src/build_database.py

Kết quả:
- data/interim/stocks.duckdb          : CSDL (bảng ohlcv, daily_features, monthly_stats, ...)
- data/processed/daily_features.parquet: bản Parquet của bảng đặc trưng ngày
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"
DB_PATH = PROJECT_ROOT / "data" / "interim" / "stocks.duckdb"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "daily_features.parquet"
BUILD_SCRIPTS = ["01_schema.sql", "02_load_raw.sql", "03_features.sql", "04_indexes_views.sql"]

# Truy vấn đối chứng: lợi suất log 1 mã trong 1 năm, tính trực tiếp trên CSV thô
# (phải dùng hàm cửa sổ mỗi lần) so với đọc bảng đã tính sẵn.
BENCH_CSV = """
SELECT symbol, time, LN(close / LAG(close) OVER (PARTITION BY symbol ORDER BY time)) AS log_ret
FROM read_csv('data/raw/hose_ohlcv_long.csv', header = true, auto_detect = true)
QUALIFY symbol = 'HPG' AND time BETWEEN '2024-01-01' AND '2024-12-31'
"""
BENCH_DB = """
SELECT symbol, time, log_ret FROM daily_features
WHERE symbol = 'HPG' AND time BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'
"""


def run_sql_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    con.execute(path.read_text(encoding="utf-8"))


def create_returns_wide_view(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Tạo view v_returns_wide (time + một cột log_ret mỗi mã, theo sort_order của symbol_meta).

    Danh sách mã được đọc từ symbol_meta và ghi rõ vào PIVOT ... IN (...) vì DuckDB không cho PIVOT động
    trong view. Mã chỉ gồm chữ/số (kiểm tra trước khi ghép chuỗi để tránh SQL injection từ file danh mục).
    """
    symbols = [r[0] for r in con.execute("SELECT symbol FROM symbol_meta ORDER BY sort_order").fetchall()]
    bad = [s for s in symbols if not s.isalnum()]
    if bad:
        raise ValueError(f"Mã không hợp lệ trong symbol_meta: {bad}")
    in_list = ", ".join(f"'{s}'" for s in symbols)
    con.execute(f"""
        CREATE OR REPLACE VIEW v_returns_wide AS
        PIVOT (SELECT time, symbol, log_ret FROM v_analysis)
        ON symbol IN ({in_list}) USING MAX(log_ret) GROUP BY time ORDER BY time""")
    return symbols


def build(con: duckdb.DuckDBPyConnection, verbose: bool = False) -> None:
    """Chạy sql/01-04 rồi tạo view dạng rộng (thư mục làm việc phải là gốc đồ án hoặc gốc dữ liệu thử)."""
    for name in BUILD_SCRIPTS:
        t0 = time.perf_counter()
        run_sql_file(con, SQL_DIR / name)
        if verbose:
            print(f"  {name:<24} {time.perf_counter() - t0:6.2f}s")
    create_returns_wide_view(con)


def bench(con: duckdb.DuckDBPyConnection, sql: str, repeat: int = 20) -> float:
    con.execute(sql).fetchall()  # khởi động
    t0 = time.perf_counter()
    for _ in range(repeat):
        con.execute(sql).fetchall()
    return (time.perf_counter() - t0) / repeat * 1000


def main() -> None:
    os.chdir(PROJECT_ROOT)  # các file SQL dùng đường dẫn tương đối từ thư mục gốc
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DB_PATH)) as con:
        build(con, verbose=True)

        for table in ["symbol_meta", "ohlcv", "daily_features", "monthly_stats", "yearly_stats"]:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<16} {n:>7} dòng")

        con.execute(f"COPY daily_features TO '{PARQUET_PATH.as_posix()}' (FORMAT parquet)")

        t_csv, t_db = bench(con, BENCH_CSV), bench(con, BENCH_DB)
        print(f"\nTruy vấn mẫu (HPG, 2024): CSV thô {t_csv:.2f} ms | bảng tính sẵn {t_db:.2f} ms "
              f"(nhanh hơn {t_csv / t_db:.1f} lần)")
    print(f"Đã ghi {DB_PATH.relative_to(PROJECT_ROOT)} và {PARQUET_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
