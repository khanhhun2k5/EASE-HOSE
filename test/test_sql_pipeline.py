"""Kiểm tra logic SQL (sql/01-05) trên dữ liệu giả lập có đáp án biết trước."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from conftest import CEILING_DAY, DUP_DAY, FLOOR_DAY, SYMBOLS, VHM_LISTING, build_db

import eda_utils


def features(con, symbol: str) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM daily_features WHERE symbol = ? ORDER BY time", [symbol]).df()


# ----------------------------------------------------------------------------- nạp dữ liệu
def test_all_rows_loaded(con, synthetic_df):
    assert con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0] == len(synthetic_df)
    assert con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0] == len(synthetic_df)


def test_symbol_meta_has_five_hose_symbols(con):
    meta = con.execute("SELECT symbol, exchange, listing_date, sort_order FROM symbol_meta").df()
    assert sorted(meta.symbol) == sorted(SYMBOLS)
    assert set(meta.exchange) == {"HOSE"}
    vhm = meta.set_index("symbol").loc["VHM", "listing_date"]
    assert pd.Timestamp(vhm) == VHM_LISTING


def test_primary_key_rejects_duplicates(tmp_path, synthetic_df):
    dup = pd.concat([synthetic_df, synthetic_df.iloc[[10]]])
    with pytest.raises(duckdb.ConstraintException):
        build_db(tmp_path, dup)


def test_analysis_config(con, synthetic_df):
    start, end = con.execute("SELECT analysis_start, analysis_end FROM analysis_config").fetchone()
    assert pd.Timestamp(start) == pd.Timestamp("2018-06-01")
    assert pd.Timestamp(end) == synthetic_df.time.max()


# ----------------------------------------------------------------------------- lợi suất
def test_returns_match_pandas(con, synthetic_df):
    f = features(con, "HPG")
    raw = synthetic_df[synthetic_df.symbol == "HPG"].sort_values("time")
    expected = raw.close.pct_change().to_numpy()
    np.testing.assert_allclose(f.ret.to_numpy()[1:], expected[1:], rtol=1e-12)
    np.testing.assert_allclose(f.log_ret.to_numpy()[1:], np.log1p(expected[1:]), rtol=1e-12)
    assert pd.isna(f.ret.iloc[0]) and pd.isna(f.log_ret.iloc[0])       # phiên đầu không có lợi suất


def test_range_and_gap(con):
    f = features(con, "STB").dropna(subset=["ret"])
    np.testing.assert_allclose(f.range_pct, (f.high - f.low) / f.prev_close)
    np.testing.assert_allclose(f.gap_ret, f.open / f.prev_close - 1)
    np.testing.assert_allclose(f.trade_value, f.close * f.volume)


def test_vhm_pre_listing_has_no_returns(con):
    f = features(con, "VHM").set_index("time")
    pre = f.loc[: VHM_LISTING - pd.Timedelta(days=1)]
    assert not pre.is_listed.any()
    # Phiên niêm yết: giá trước đó là giá "ảo" -> mọi lợi suất phải NULL
    listing = f.loc[VHM_LISTING]
    assert listing.is_listed
    for col in ("ret", "log_ret", "range_pct", "gap_ret"):
        assert pd.isna(listing[col]), col
    assert f.ret.abs().max() < 0.1      # không còn lợi suất +268%


def test_is_traded_flag(con):
    f = features(con, "VHM")
    assert (f.is_traded == (f.volume > 0)).all()


# ----------------------------------------------------------------------------- cửa sổ trượt
def test_vol_20d_requires_full_window_and_matches_pandas(con):
    f = features(con, "NVL")
    expected = f.log_ret.rolling(20, min_periods=20).std()
    assert f.vol_20d.iloc[:20].isna().all()     # phiên 1 chưa có log_ret -> cần tới phiên 21
    np.testing.assert_allclose(f.vol_20d.iloc[20:], expected.iloc[20:], rtol=1e-9)


def test_vol_20d_ignores_vhm_listing_jump(con):
    f = features(con, "VHM").dropna(subset=["vol_20d"])
    assert f.vol_20d.max() < 0.05      # nếu lẫn lợi suất ảo 1,3 thì vol sẽ > 0,25


def test_rel_volume_uses_previous_20_sessions(con):
    f = features(con, "VPB")
    expected = f.volume / f.volume.shift(1).rolling(20, min_periods=20).mean()
    assert f.rel_volume.iloc[:20].isna().all()
    np.testing.assert_allclose(f.rel_volume.iloc[20:], expected.iloc[20:], rtol=1e-9)


def test_rel_volume_null_while_window_has_untraded_sessions(con):
    f = features(con, "VHM").set_index("time")
    after = f.loc[VHM_LISTING:]
    # 20 phiên sau niêm yết: cửa sổ trước đó còn chứa phiên chưa niêm yết -> NULL
    assert after.rel_volume.iloc[:20].isna().all()
    assert after.rel_volume.iloc[21:].notna().all()


def test_windows_do_not_leak_across_symbols(con):
    """Phiên đầu tiên của mỗi mã không được dùng giá của mã đứng trước nó."""
    first = con.execute("""
        SELECT symbol, prev_close FROM daily_features
        QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY time) = 1""").df()
    assert first.prev_close.isna().all()


# ----------------------------------------------------------------------------- trần/sàn
def test_ceiling_and_floor_flags(con):
    hpg = features(con, "HPG").set_index("time")
    stb = features(con, "STB").set_index("time")
    nvl = features(con, "NVL").set_index("time")
    assert hpg.loc[CEILING_DAY, "hit_ceiling"]
    assert stb.loc[FLOOR_DAY, "hit_floor"]
    assert not nvl.loc[DUP_DAY, "hit_ceiling"]          # tăng 6,8% nhưng close < high
    total = con.execute("SELECT SUM(hit_ceiling::INT) + SUM(hit_floor::INT) FROM daily_features").fetchone()[0]
    assert total == 2


# ----------------------------------------------------------------------------- bảng tổng hợp
def test_monthly_return_compounds_daily_log_returns(con):
    f = features(con, "HPG").dropna(subset=["log_ret"])
    f["month"] = f.time.dt.to_period("M").dt.to_timestamp()
    expected = np.expm1(f.groupby("month").log_ret.sum())
    m = con.execute("SELECT month, ret_month FROM monthly_stats WHERE symbol = 'HPG' ORDER BY month").df()
    np.testing.assert_allclose(m.ret_month, expected.to_numpy(), rtol=1e-9)


def test_yearly_stats_excludes_pre_listing(con):
    n = con.execute("SELECT n_days FROM yearly_stats WHERE symbol = 'VHM'").fetchone()[0]
    expected = con.execute("""
        SELECT COUNT(*) FROM daily_features
        WHERE symbol = 'VHM' AND is_listed AND log_ret IS NOT NULL""").fetchone()[0]
    assert n == expected


# ----------------------------------------------------------------------------- view
def test_v_analysis_window(con):
    lo, hi, nulls = con.execute(
        "SELECT MIN(time), MAX(time), COUNT(*) FILTER (WHERE log_ret IS NULL) FROM v_analysis").fetchone()
    assert pd.Timestamp(lo) >= pd.Timestamp("2018-06-01")
    assert nulls == 0


def test_v_returns_wide(con):
    wide = con.execute("SELECT * FROM v_returns_wide").df()
    assert list(wide.columns) == ["time", *SYMBOLS]
    assert wide.time.is_monotonic_increasing and wide.time.is_unique
    long = con.execute("SELECT log_ret FROM v_analysis WHERE symbol = 'STB' ORDER BY time").df()
    np.testing.assert_allclose(wide.STB.to_numpy(), long.log_ret.to_numpy())


# ----------------------------------------------------------------------------- truy vấn mẫu (05)
@pytest.mark.parametrize("name", list(eda_utils.named_queries()))
def test_named_queries_run(con, name):
    df = eda_utils.query(con, name)
    assert isinstance(df, pd.DataFrame)


def test_profile_query_counts(con, synthetic_df):
    p = eda_utils.query(con, "profile_by_symbol").set_index("symbol")
    assert p.n_rows.sum() == len(synthetic_df)
    n_pre = (synthetic_df[synthetic_df.symbol == "VHM"].time < VHM_LISTING).sum()
    assert p.loc["VHM", "n_pre_listing"] == n_pre
    assert p.n_invalid_ohlc.sum() == 0


def test_missing_sessions_empty_for_complete_calendar(con):
    assert eda_utils.query(con, "missing_sessions").empty
