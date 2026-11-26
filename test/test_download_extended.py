"""Kiểm thử tiêu chí chọn mã bổ sung (download_extended.select_universe) — dữ liệu giả, không gọi mạng."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import download_extended as DE


def _series(sym, days, close, volume=1000.0):
    return pd.DataFrame({"symbol": sym, "time": days, "close": close, "volume": volume})


@pytest.fixture()
def data():
    days = pd.bdate_range("2018-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    calm = lambda: 10 * np.exp(np.cumsum(rng.uniform(-0.02, 0.02, len(days))))    # |r| < 7,5%
    frames = [
        _series("AAA", days, calm(), 5000.0),                        # đạt, thanh khoản cao nhất
        _series("BBB", days, calm(), 3000.0),                        # đạt
        _series("CCC", days, calm(), 1000.0),                        # đạt nhưng thấp nhất → ngoài top 2
        _series("LATE", days[days >= "2019-01-01"], 10.0, 9000.0),   # bắt đầu sau cửa sổ
        _series("VCB", days, calm(), 9000.0),                        # mã nghiên cứu → loại
    ]
    gap = calm()
    frames.append(_series("GAPS", days, gap, 9000.0).iloc[::2])      # thiếu ~50% phiên
    jumpy = calm()
    jumpy[400::50] *= 1.09                                           # nhiều phiên |r| > 7,5% (giống HNX ±10%)
    frames.append(_series("HNX", days, jumpy, 9000.0))
    return pd.concat(frames, ignore_index=True)


def test_select_universe_applies_each_rule(data):
    out = DE.select_universe(data, n=2, window_start="2018-06-01").set_index("symbol")
    assert set(out.index[out["selected"]]) == {"AAA", "BBB"}
    assert out.loc["AAA", "rank"] == 1 and out.loc["BBB", "rank"] == 2
    assert out.loc["CCC", "eligible"] and "top n" in out.loc["CCC", "reason"]
    assert "sau" in out.loc["LATE", "reason"]
    assert "20 mã" in out.loc["VCB", "reason"]
    assert "thiếu" in out.loc["GAPS", "reason"] and out.loc["GAPS", "missing_share"] > 0.4
    assert "HOSE" in out.loc["HNX", "reason"] and out.loc["HNX", "n_breaches"] > 2
    assert not out.loc[["LATE", "VCB", "GAPS", "HNX"], "selected"].any()
    assert list(out.index[:2]) == ["AAA", "BBB"]                     # mã được chọn xếp đầu


def test_select_universe_respects_reference_dates_and_thresholds(data):
    ref = pd.bdate_range("2018-06-01", "2020-12-31")
    out = DE.select_universe(data, n=5, reference_dates=ref, max_breaches=100,
                             exclude=[]).set_index("symbol")
    assert out.loc["HNX", "selected"] and out.loc["VCB", "selected"]  # nới ngưỡng, không loại mã nghiên cứu
    assert out["selected"].sum() == 5


def test_select_universe_validates_input(data):
    with pytest.raises(ValueError):
        DE.select_universe(data, n=0)
    with pytest.raises(ValueError):
        DE.select_universe(data.drop(columns="volume"))


def test_gap_jump_is_not_counted_as_breach():
    """Mã tạm ngừng 2 phiên rồi nhảy ~14% (hai lần biên độ): không phải vi phạm biên độ ±7%."""
    days = pd.bdate_range("2018-01-01", "2019-12-31")
    close = np.full(len(days), 10.0)
    close[300:] = 11.4
    d = _series("GAP", days, close, 1000.0).drop(index=[298, 299])          # 2 phiên không giao dịch
    ref = pd.DatetimeIndex(days[days >= "2018-06-01"])
    out = DE.select_universe(d, n=1, reference_dates=ref, exclude=[]).set_index("symbol")
    assert out.loc["GAP", "n_breaches"] == 0 and out.loc["GAP", "selected"]


def test_candidates_are_unique_and_have_sectors():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(DE))                              # dict literal: khoá trùng bị ghi đè âm thầm
    node = next(n.value for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "CANDIDATES")
    keys = [k.value for k in node.keys]
    assert len(keys) == len(set(keys)) == len(DE.CANDIDATES)
    assert all(isinstance(v, str) and v for v in DE.CANDIDATES.values())
    assert len([s for s in DE.CANDIDATES if s not in DE.BASE_SYMBOLS]) == 90
    import config
    assert set(DE.BASE_SYMBOLS) <= set(config.SYMBOLS) and len(DE.BASE_SYMBOLS) == 20
