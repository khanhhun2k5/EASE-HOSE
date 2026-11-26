"""Kiểm tra src/download_vnstock.py với vnstock GIẢ (không gọi mạng, không cài gói)."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import download_vnstock as D


# ----------------------------------------------------------------------------- vnstock giả
def fake_ohlcv(symbol: str, tz: str | None = "Asia/Ho_Chi_Minh") -> pd.DataFrame:
    """Nến ngày từ 2017-12-28 tới 2018-01-10 (nguồn VCI hay trả rộng hơn khoảng yêu cầu)."""
    times = pd.bdate_range("2017-12-28", "2018-01-10") + pd.Timedelta(hours=7)
    if tz:
        times = times.tz_localize(tz)
    base = 10 + 5 * D.SYMBOLS.index(symbol)                         # mỗi mã một mức giá
    n = len(times)
    return pd.DataFrame({"time": times, "open": [base] * n, "high": [base + 1] * n,
                         "low": [base - 1] * n, "close": [base + 0.5] * n, "volume": [1000] * n})


def install_fake_vnstock(monkeypatch, version: str, fail: set[str] = frozenset()):
    """Đưa module 'vnstock' giả vào sys.modules; ghi lại mọi lời gọi."""
    calls = []
    mod = types.ModuleType("vnstock")
    mod.__version__ = version

    class Quote:
        def __init__(self, source, symbol):
            self.source, self.symbol = source, symbol

        def history(self, start, end, interval):
            calls.append(("Quote", self.source, self.symbol, start, end, interval))
            if self.symbol in fail:
                raise ConnectionError("mất mạng")
            return fake_ohlcv(self.symbol)

    class _Equity:
        def __init__(self, source, symbol):
            self.source, self.symbol = source, symbol

        def ohlcv(self, start, end, interval):
            calls.append(("Market", self.source, self.symbol, start, end, interval))
            if self.symbol in fail:
                raise RuntimeError("Chưa có khoá: TCBS_API_KEY")
            return fake_ohlcv(self.symbol, tz=None)

    class Market:
        def __init__(self, source):
            self.source = source

        def equity(self, symbol):
            return _Equity(self.source, symbol)

    mod.Quote, mod.Market = Quote, Market
    monkeypatch.setitem(sys.modules, "vnstock", mod)
    return calls


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)


@pytest.fixture()
def clean_env(monkeypatch, tmp_path):
    """Xoá khoá API trong môi trường và trỏ PROJECT_ROOT sang thư mục tạm (không đọc .env thật)."""
    for env_vars in D.V6_SOURCE_KEYS.values():
        for v in env_vars:
            monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(D, "PROJECT_ROOT", tmp_path)
    return tmp_path


# ----------------------------------------------------------------------------- hằng số
def test_symbols_and_exchange():
    import config
    assert D.SYMBOLS == config.SYMBOLS and len(D.SYMBOLS) == 60
    assert set(config.ORIGINAL_SYMBOLS) <= set(D.SYMBOLS)
    assert D.EXCHANGE == "HOSE"


# ----------------------------------------------------------------------------- chọn nguồn v6
def test_pick_source_explicit_is_lowercased(clean_env):
    assert D.pick_v6_source("SSI") == "ssi"


def test_pick_source_default_tcbs_without_keys(clean_env):
    assert D.pick_v6_source(None) == "tcbs"


def test_pick_source_first_with_complete_keys(clean_env, monkeypatch):
    monkeypatch.setenv("SSI_API_KEY", "k")            # thiếu SSI_API_SECRET -> chưa đủ
    monkeypatch.setenv("DNSE_API_KEY", "k")
    monkeypatch.setenv("DNSE_API_SECRET", "s")
    assert D.pick_v6_source(None) == "dnse"


def test_pick_source_reads_dotenv(clean_env, monkeypatch):
    pytest.importorskip("dotenv")
    (clean_env / ".env").write_text("FHSC_API_KEY=abc\n", encoding="utf-8")
    assert D.pick_v6_source(None) == "fhsc"
    monkeypatch.delenv("FHSC_API_KEY", raising=False)  # load_dotenv ghi vào os.environ


# ----------------------------------------------------------------------------- phiên bản
@pytest.mark.parametrize("version,major", [("3.4.2", 3), ("6.0.1", 6), ("10.2", 10)])
def test_vnstock_major_version(monkeypatch, version, major):
    install_fake_vnstock(monkeypatch, version)
    assert D.vnstock_major_version() == major


# ----------------------------------------------------------------------------- gọi API đúng thế hệ
def test_fetch_one_legacy_uses_quote_uppercase_source(monkeypatch):
    calls = install_fake_vnstock(monkeypatch, "3.4.2")
    D.fetch_one("HPG", "2018-01-01", "2018-01-05", major=3, source="vci")
    assert calls == [("Quote", "VCI", "HPG", "2018-01-01", "2018-01-05", "1D")]


def test_fetch_one_v6_uses_market(monkeypatch):
    calls = install_fake_vnstock(monkeypatch, "6.0.1")
    D.fetch_one("STB", "2018-01-01", "2018-01-05", major=6, source="tcbs")
    assert calls == [("Market", "tcbs", "STB", "2018-01-01", "2018-01-05", "1D")]


# ----------------------------------------------------------------------------- download()
def test_download_combines_filters_and_strips_timezone(monkeypatch):
    install_fake_vnstock(monkeypatch, "3.4.2")
    data, failed = D.download("2018-01-02", "2018-01-05", major=3, source="vci", pause=0)
    assert failed == {}
    assert list(data.symbol.unique()) == D.SYMBOLS
    assert data.time.dt.tz is None
    assert (data.time == data.time.dt.normalize()).all()            # bỏ phần giờ 07:00
    assert data.time.min() == pd.Timestamp("2018-01-02")             # lọc khoảng ngày
    assert data.time.max() == pd.Timestamp("2018-01-05")
    assert (data.exchange == "HOSE").all()
    assert len(data) == len(D.SYMBOLS) * 4                           # 4 phiên x số mã
    assert list(data.columns[:2]) == ["symbol", "time"]


def test_download_tz_naive_source(monkeypatch):
    install_fake_vnstock(monkeypatch, "6.0.1")
    data, _ = D.download("2018-01-02", "2018-01-05", major=6, source="tcbs", pause=0)
    assert data.time.dt.tz is None and len(data) == len(D.SYMBOLS) * 4


def test_download_partial_failure_keeps_other_symbols(monkeypatch):
    install_fake_vnstock(monkeypatch, "3.4.2", fail={"NVL"})
    data, failed = D.download("2018-01-02", "2018-01-05", major=3, source="vci", pause=0)
    assert set(failed) == {"NVL"} and "ConnectionError" in failed["NVL"]
    assert "NVL" not in set(data.symbol) and data.symbol.nunique() == len(D.SYMBOLS) - 1


def test_download_all_fail_returns_none(monkeypatch):
    install_fake_vnstock(monkeypatch, "6.0.1", fail=set(D.SYMBOLS))
    data, failed = D.download("2018-01-02", "2018-01-05", major=6, source="tcbs", pause=0)
    assert data is None and set(failed) == set(D.SYMBOLS)


def test_download_empty_range(monkeypatch):
    install_fake_vnstock(monkeypatch, "3.4.2")
    data, failed = D.download("2030-01-01", "2030-12-31", major=3, source="vci", pause=0)
    assert failed == {} and data.empty


# ----------------------------------------------------------------------------- cài lại thư viện
def test_reinstall_uses_official_index_and_v6_pin(monkeypatch):
    seen = {}
    monkeypatch.setattr(D.subprocess, "check_call", lambda cmd: seen.setdefault("cmd", cmd))
    D.reinstall_vnstock()
    cmd = seen["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "pip"]
    assert "--force-reinstall" in cmd
    assert cmd[cmd.index("--extra-index-url") + 1] == D.VNSTOCK_INDEX
    assert "vnstock>=6.0.0,<7" in cmd


# ----------------------------------------------------------------------------- main()
def run_main(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(D, "OUTPUT_DIR", tmp_path / "raw")
    monkeypatch.setattr(D, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_vnstock.py", *args])
    return D.main()


def test_main_no_upgrade_writes_long_and_wide_csv(monkeypatch, tmp_path):
    install_fake_vnstock(monkeypatch, "3.4.2")
    monkeypatch.setattr(D, "reinstall_vnstock", lambda: pytest.fail("không được cài lại khi có --no-upgrade"))
    code = run_main(monkeypatch, tmp_path, "--no-upgrade", "--start", "2018-01-02", "--end", "2018-01-05")
    assert code == 0
    long = pd.read_csv(tmp_path / "raw" / "hose_ohlcv_long.csv", encoding="utf-8-sig")
    wide = pd.read_csv(tmp_path / "raw" / "hose_close_wide.csv", encoding="utf-8-sig", index_col=0)
    assert len(long) == len(D.SYMBOLS) * 4
    assert list(wide.columns) == D.SYMBOLS                            # đúng thứ tự mã
    assert wide.loc["2018-01-02", "HPG"] == 10 + 5 * D.SYMBOLS.index("HPG") + 0.5


def test_main_upgrade_by_default(monkeypatch, tmp_path):
    install_fake_vnstock(monkeypatch, "3.4.2")
    called = []
    monkeypatch.setattr(D, "reinstall_vnstock", lambda: called.append(True))
    run_main(monkeypatch, tmp_path, "--start", "2018-01-02", "--end", "2018-01-05")
    assert called == [True]


def test_main_returns_1_on_partial_failure(monkeypatch, tmp_path):
    install_fake_vnstock(monkeypatch, "3.4.2", fail={"VPB"})
    code = run_main(monkeypatch, tmp_path, "--no-upgrade", "--start", "2018-01-02", "--end", "2018-01-05")
    assert code == 1
    assert (tmp_path / "raw" / "hose_ohlcv_long.csv").exists()      # vẫn lưu phần tải được


def test_main_v6_without_keys_prints_guidance(monkeypatch, tmp_path, clean_env, capsys):
    install_fake_vnstock(monkeypatch, "6.0.1", fail=set(D.SYMBOLS))
    code = run_main(monkeypatch, tmp_path, "--no-upgrade")
    assert code == 1
    assert "vnstock keys set" in capsys.readouterr().out
    assert not (tmp_path / "raw").exists()
