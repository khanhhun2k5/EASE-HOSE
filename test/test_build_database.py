"""Kiểm tra src/build_database.py: dựng CSDL đầu-cuối trong thư mục tạm."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

import build_database as B


@pytest.fixture()
def project(tmp_path, monkeypatch, synthetic_df):
    """Thư mục dự án giả: chỉ có data/raw; SQL vẫn lấy từ sql/ thật."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    synthetic_df.to_csv(raw / "hose_ohlcv_long.csv", index=False)
    from conftest import write_symbol_meta
    write_symbol_meta(tmp_path)
    monkeypatch.setattr(B, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(B, "DB_PATH", tmp_path / "data" / "interim" / "stocks.duckdb")
    monkeypatch.setattr(B, "PARQUET_PATH", tmp_path / "data" / "processed" / "daily_features.parquet")
    monkeypatch.chdir(tmp_path)        # main() tự chdir; trả lại thư mục cũ sau test
    return tmp_path


def test_build_scripts_exist_in_order():
    assert B.BUILD_SCRIPTS == sorted(B.BUILD_SCRIPTS)
    for name in B.BUILD_SCRIPTS:
        assert (B.SQL_DIR / name).is_file(), name


def test_main_builds_db_and_parquet(project, synthetic_df, capsys):
    B.main()
    assert B.DB_PATH.exists() and B.PARQUET_PATH.exists()
    with duckdb.connect(str(B.DB_PATH), read_only=True) as con:
        tables = set(con.execute("SELECT table_name FROM duckdb_tables()").df().table_name)
        assert {"symbol_meta", "ohlcv", "daily_features", "monthly_stats", "yearly_stats",
                "analysis_config"} <= tables
        db_rows = con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
    parquet = pd.read_parquet(B.PARQUET_PATH)
    assert len(parquet) == db_rows == len(synthetic_df)
    out = capsys.readouterr().out
    assert "nhanh hơn" in out and "daily_features" in out


def test_main_is_idempotent(project):
    """Chạy lại lần hai không lỗi trùng khoá và không nhân đôi dữ liệu."""
    B.main()
    B.main()
    with duckdb.connect(str(B.DB_PATH), read_only=True) as con:
        assert con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0] == \
               con.execute("SELECT COUNT(DISTINCT (symbol, time)) FROM ohlcv").fetchone()[0]


def test_benchmark_queries_return_same_rows(project):
    B.main()
    with duckdb.connect(str(B.DB_PATH), read_only=True) as con:
        a = con.execute(B.BENCH_CSV.replace("2024", "2018")).df().sort_values("time")
        b = con.execute(B.BENCH_DB.replace("2024", "2018")).df().sort_values("time")
    assert len(a) == len(b) > 0


def test_bench_returns_positive_ms():
    with duckdb.connect() as con:
        assert B.bench(con, "SELECT 1", repeat=3) > 0


def test_run_sql_file(tmp_path):
    f = tmp_path / "x.sql"
    f.write_text("CREATE TABLE t AS SELECT 'Hòa Phát' AS ten;", encoding="utf-8")
    with duckdb.connect() as con:
        B.run_sql_file(con, f)
        assert con.execute("SELECT ten FROM t").fetchone()[0] == "Hòa Phát"


def test_missing_symbol_in_meta_fails_early(tmp_path, synthetic_df):
    """Mã có trong ohlcv nhưng thiếu trong symbol_meta.csv -> dựng CSDL báo lỗi rõ ràng."""
    from conftest import write_symbol_meta
    raw = tmp_path / "data" / "raw"; raw.mkdir(parents=True)
    synthetic_df.to_csv(raw / "hose_ohlcv_long.csv", index=False)
    write_symbol_meta(tmp_path, ["VHM", "NVL"])
    cwd = __import__("os").getcwd()
    __import__("os").chdir(tmp_path)
    try:
        with duckdb.connect(str(tmp_path / "x.duckdb")) as con, pytest.raises(duckdb.Error, match="symbol_meta"):
            B.build(con)
    finally:
        __import__("os").chdir(cwd)


def test_wide_view_rejects_unsafe_symbol():
    con = duckdb.connect()
    con.execute("CREATE TABLE symbol_meta (symbol VARCHAR, sort_order INT)")
    con.execute("INSERT INTO symbol_meta VALUES ('VHM', 1), ('X''); DROP TABLE t; --', 2)")
    with pytest.raises(ValueError, match="không hợp lệ"):
        B.create_returns_wide_view(con)
