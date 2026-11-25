"""Kiểm thử danh mục mã (src/config.py, data/reference/symbol_meta.csv)."""

from __future__ import annotations

import pandas as pd
import pytest

import config
import eda_utils as U


def test_real_catalogue_has_20_symbols_grouped_by_sector():
    assert len(config.SYMBOLS) == 20 and len(set(config.SYMBOLS)) == 20
    assert set(config.ORIGINAL_SYMBOLS) <= set(config.SYMBOLS)
    assert all(s.isalnum() and s.isupper() for s in config.SYMBOLS)
    # mỗi ngành là một khối liền nhau theo sort_order
    sectors = [config.SECTOR[s] for s in config.SYMBOLS]
    assert sectors == sorted(sectors, key=config.SECTORS.index)


def test_every_symbol_has_a_distinct_colour():
    assert set(U.SYMBOL_COLORS) == set(config.SYMBOLS)
    assert len(set(U.SYMBOL_COLORS.values())) == 20
    assert set(U.SECTOR_COLORS) == set(config.SECTORS)


@pytest.mark.parametrize("bad, msg", [
    ({"symbol": ["A", "A"], "sort_order": [1, 2]}, "trùng"),
    ({"symbol": ["A", "B"], "sort_order": [1, 1]}, "sort_order"),
])
def test_load_symbol_meta_validates(tmp_path, bad, msg):
    df = pd.DataFrame({**bad, "company_name": ["Công ty Á", "B"], "sector": ["Ngân hàng"] * 2,
                       "exchange": ["HOSE"] * 2, "listing_date": [None, "2018-01-02"]})
    path = tmp_path / "m.csv"; df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=msg):
        config.load_symbol_meta(path)


def test_load_symbol_meta_missing_column(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame({"symbol": ["A"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="thiếu cột"):
        config.load_symbol_meta(path)
