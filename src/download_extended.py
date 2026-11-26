"""Tải và chọn 40 mã HOSE bổ sung (ngoài 20 mã nghiên cứu) để ước lượng nhân tố chung trên một bảng rộng hơn.

Vì sao cần: Calibration cho thấy phải khử nhân tố chung trước EASE, và nhân tố ước lượng từ chính 20 mã đang xét
chứa nhiễu riêng của chúng. Các mô hình nhân tố tiềm ẩn (PCA, autoencoder — Gu, Kelly & Xiu 2021,
doi:10.1016/j.jeconom.2020.07.009) cần nhiều tài sản hơn số nhân tố rất nhiều. 40 mã này CHỈ dùng để ước lượng
nhân tố; EASE vẫn chạy trên 20 mã của ``symbol_meta.csv``.

Chọn mã bằng tiêu chí dữ liệu (``select_universe``), không chọn tay từng mã:
1. Có dữ liệu từ trước ngày đầu cửa sổ phân tích (2018-06-01) và thiếu ≤ ``max_missing`` số phiên của cửa sổ.
2. Lịch sử giống HOSE: số phiên |lợi suất đơn| > ``limit`` (7,5%, trên biên độ ±7%) ≤ ``max_breaches``.
   Mã chuyển từ HNX/UPCoM (biên độ ±10%/±15%) trong cửa sổ thường vi phạm nhiều lần nên bị loại.
   (20 mã nghiên cứu có 0 phiên vượt 7,5% trong cửa sổ.)
3. Trong các mã đạt (1)–(2), lấy ``n`` mã có giá trị giao dịch trung vị (close × volume) cao nhất trong 2 năm cuối.

Danh sách ứng viên ``CANDIDATES`` là các mã đang niêm yết trên HOSE (ngành ghi theo hoạt động chính, để mô tả;
không dùng trong ước lượng). Mã không tải được hoặc không đạt tiêu chí sẽ bị loại tự động.

Cách chạy (vnstock 3.x, nguồn VCI):
    python src/download_extended.py

Kết quả:
- data/raw/extended_ohlcv_long.csv         : OHLCV mọi ứng viên tải được (dạng dài, như hose_ohlcv_long.csv)
- data/reference/extended_universe.csv     : bảng chọn mã (thống kê từng tiêu chí, cột ``selected``)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import BASE_SYMBOLS  # 20 mã trước khi mở rộng: loại trừ khi chọn (không dùng SYMBOLS, nay đã gồm cả 40 mã chọn ở đây)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "extended_ohlcv_long.csv"
UNIVERSE_PATH = PROJECT_ROOT / "data" / "reference" / "extended_universe.csv"
WINDOW_START = "2018-06-01"

CANDIDATES: dict[str, str] = {
    # Ngân hàng
    "ACB": "Ngân hàng", "HDB": "Ngân hàng", "TPB": "Ngân hàng", "EIB": "Ngân hàng", "LPB": "Ngân hàng",
    "VIB": "Ngân hàng", "MSB": "Ngân hàng", "SHB": "Ngân hàng", "OCB": "Ngân hàng", "SSB": "Ngân hàng",
    # Bất động sản & khu công nghiệp
    "VRE": "Bất động sản", "DXG": "Bất động sản", "NLG": "Bất động sản", "PDR": "Bất động sản",
    "KBC": "Bất động sản", "DIG": "Bất động sản", "HDG": "Bất động sản", "CII": "Bất động sản",
    "SCR": "Bất động sản", "IJC": "Bất động sản", "NTL": "Bất động sản", "HDC": "Bất động sản",
    "TCH": "Bất động sản", "SZC": "Bất động sản", "DXS": "Bất động sản", "AGG": "Bất động sản",
    # Chứng khoán
    "HCM": "Chứng khoán", "VND": "Chứng khoán", "VCI": "Chứng khoán", "FTS": "Chứng khoán",
    "CTS": "Chứng khoán", "AGR": "Chứng khoán", "VIX": "Chứng khoán", "BSI": "Chứng khoán", "ORS": "Chứng khoán",
    # Tiêu dùng & bán lẻ
    "PNJ": "Tiêu dùng", "DGW": "Tiêu dùng", "FRT": "Tiêu dùng", "KDC": "Tiêu dùng", "SBT": "Tiêu dùng",
    "VHC": "Tiêu dùng", "ANV": "Tiêu dùng", "MSH": "Tiêu dùng", "TCM": "Tiêu dùng", "STK": "Tiêu dùng",
    # Vật liệu, thép, hoá chất
    "HSG": "Vật liệu", "NKG": "Vật liệu", "DGC": "Vật liệu", "DCM": "Vật liệu", "DPM": "Vật liệu",
    "HT1": "Vật liệu", "BMP": "Vật liệu", "CSV": "Vật liệu", "GVR": "Vật liệu", "PHR": "Vật liệu",
    "DPR": "Vật liệu", "AAA": "Vật liệu", "BFC": "Vật liệu",
    # Năng lượng & tiện ích
    "PVD": "Năng lượng", "PVT": "Năng lượng", "POW": "Năng lượng", "REE": "Năng lượng", "NT2": "Năng lượng",
    "PC1": "Năng lượng", "GEX": "Năng lượng", "GEG": "Năng lượng", "PPC": "Năng lượng", "VSH": "Năng lượng",
    "BWE": "Năng lượng", "TDM": "Năng lượng",
    # Công nghiệp, vận tải, xây dựng
    "GMD": "Công nghiệp", "VJC": "Công nghiệp", "HVN": "Công nghiệp", "VSC": "Công nghiệp",
    "CTD": "Công nghiệp", "VCG": "Công nghiệp", "LCG": "Công nghiệp", "HHV": "Công nghiệp",
    "PTB": "Công nghiệp", "HAH": "Công nghiệp", "SKG": "Công nghiệp",
    # Công nghệ, y tế, nông nghiệp
    "CMG": "Công nghệ", "ELC": "Công nghệ", "DHG": "Y tế", "IMP": "Y tế", "DBD": "Y tế",
    "HAG": "Nông nghiệp", "HNG": "Nông nghiệp", "DBC": "Nông nghiệp", "BAF": "Nông nghiệp",
}


def select_universe(data: pd.DataFrame, n: int = 40, window_start: str = WINDOW_START,
                    window_end: str | None = None, max_missing: float = 0.02, limit: float = 0.075,
                    max_breaches: int = 2, liquidity_years: float = 2.0,
                    reference_dates: pd.DatetimeIndex | None = None,
                    exclude: list[str] | None = None) -> pd.DataFrame:
    """Áp ba tiêu chí (xem docstring module) lên dữ liệu dạng dài (symbol, time, close, volume).

    ``reference_dates``: các phiên giao dịch của cửa sổ (mặc định: hợp các ngày có dữ liệu trong cửa sổ).
    ``exclude``: mã không được chọn (mặc định ``BASE_SYMBOLS``).
    Trả DataFrame một dòng/mã: first_date, missing_share, n_breaches, median_value (tỷ đồng, nếu giá tính
    bằng nghìn đồng), eligible, reason, selected, rank (thứ hạng thanh khoản trong các mã đạt).
    """
    if n < 1:
        raise ValueError("n phải ≥ 1.")
    need = {"symbol", "time", "close", "volume"}
    if need - set(data.columns):
        raise ValueError(f"Thiếu cột: {sorted(need - set(data.columns))}")
    exclude = set(BASE_SYMBOLS if exclude is None else exclude)
    d = data.assign(time=pd.to_datetime(data["time"])).sort_values(["symbol", "time"])
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end) if window_end else d["time"].max()
    if reference_dates is None:
        reference_dates = pd.DatetimeIndex(sorted(d.loc[d["time"].between(start, end), "time"].unique()))
    ref = pd.DatetimeIndex(reference_dates)
    liq_start = end - pd.DateOffset(days=int(round(365.25 * liquidity_years)))
    rows = []
    for sym, g in d.groupby("symbol", sort=True):
        g = g.drop_duplicates("time").set_index("time")
        win = g.loc[start:end]
        # lợi suất chỉ giữa hai phiên liền nhau của lịch tham chiếu: cú nhảy qua phiên thiếu (tạm ngừng) không bị
        # tính là một phiên vượt biên độ
        ret = win["close"].reindex(ref).pct_change().dropna()
        liq = g.loc[liq_start:end]
        row = {"symbol": sym, "first_date": g.index.min().date() if len(g) else None,
               "missing_share": float(1 - ref.isin(win.index).mean()) if len(ref) else 1.0,
               "n_breaches": int((ret.abs() > limit).sum()),
               "median_value": float((liq["close"] * liq["volume"]).median() / 1e6) if len(liq) else 0.0}
        if sym in exclude:
            reason = "đã trong 20 mã nghiên cứu"
        elif len(g) == 0 or g.index.min() > start:
            reason = f"dữ liệu bắt đầu sau {window_start}"
        elif row["missing_share"] > max_missing:
            reason = f"thiếu {row['missing_share']:.1%} phiên > {max_missing:.0%}"
        elif row["n_breaches"] > max_breaches:
            reason = f"{row['n_breaches']} phiên |r| > {limit:.1%} (lịch sử ngoài HOSE?)"
        elif not row["median_value"] > 0:
            reason = "không có giao dịch"
        else:
            reason = ""
        row["eligible"], row["reason"] = reason == "", reason
        rows.append(row)
    out = pd.DataFrame(rows)
    out["rank"] = out["median_value"].where(out["eligible"]).rank(ascending=False, method="first")
    out["selected"] = out["rank"] <= n
    out.loc[out["eligible"] & ~out["selected"], "reason"] = "thanh khoản thấp hơn top n"
    return out.sort_values(["selected", "rank"], ascending=[False, True]).reset_index(drop=True)


def main() -> int:
    import download_vnstock as D

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--n", type=int, default=40)
    # Gói khách của vnstock: 20 request/phút, vượt thì thư viện tự dừng tiến trình → mặc định 3,5 s/mã.
    ap.add_argument("--pause", type=float, default=3.5)
    ap.add_argument("--fresh", action="store_true", help="Tải lại từ đầu (mặc định: tải tiếp các mã còn thiếu)")
    args = ap.parse_args()

    if D.vnstock_major_version() >= 6:
        print("Script này dùng vnstock 3.x (nguồn VCI, không cần khoá). Bản ≥ 6 cần khoá API của công ty chứng khoán: "
              'cài lại `pip install "vnstock>=3.4.2,<4"` rồi chạy lại.')
        return 1
    symbols = [s for s in CANDIDATES if s not in BASE_SYMBOLS]
    study = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "hose_ohlcv_long.csv", parse_dates=["time"])
    ref = pd.DatetimeIndex(sorted(study.loc[study["time"] >= WINDOW_START, "time"].unique()))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    have = pd.DataFrame()
    if RAW_PATH.exists() and not args.fresh:
        have = pd.read_csv(RAW_PATH, parse_dates=["time"])
        have = have[have["symbol"].isin(symbols)]                    # bỏ mã không còn trong danh sách ứng viên
        last = have.groupby("symbol")["time"].max()
        stale = last.index[last < ref.max()].tolist()                # dữ liệu cũ hơn dữ liệu 20 mã → tải lại
        have = have[~have["symbol"].isin(stale)]
    todo = [s for s in symbols if have.empty or s not in set(have["symbol"])]
    print(f">> {len(symbols)} ứng viên; đã có {len(symbols) - len(todo)}, cần tải {len(todo)} (VCI, "
          f"{args.start} → {args.end})")
    failed: dict[str, str] = {}
    for attempt in range(2):                                         # lượt 2: thử lại các mã lỗi mạng
        if attempt == 1:
            print(f">> Chờ 60 giây rồi thử lại {len(todo)} mã lỗi")
            time.sleep(60)                                           # hết cửa sổ giới hạn 20 request/phút
        for s in todo:
            df, err = D.download(args.start, args.end, major=3, source="vci", pause=args.pause, symbols=[s])
            if df is None:
                failed[s] = err[s]
                time.sleep(args.pause)                               # lỗi cũng tính là một request: vẫn giãn nhịp
                continue
            failed.pop(s, None)
            have = pd.concat([have, df], ignore_index=True)
            have.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")   # ghi sau mỗi mã: bị dừng vẫn giữ được
        todo = list(failed)
        if not todo:
            break
    if have.empty:
        print("Không tải được mã nào.")
        return 1
    data = have
    uni =select_universe(data, n=args.n, reference_dates=ref, window_end=str(ref.max().date()))
    uni.insert(1, "sector", uni["symbol"].map(CANDIDATES))
    for s, msg in failed.items():                                   # mã lỗi tải vẫn ghi vào bảng để minh bạch
        uni.loc[len(uni)] = {"symbol": s, "sector": CANDIDATES[s], "eligible": False, "selected": False,
                             "reason": f"lỗi tải: {msg[:60]}"}
    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(UNIVERSE_PATH, index=False, encoding="utf-8")
    sel = uni[uni["selected"].astype(bool)]
    print(f"\nĐạt tiêu chí: {int(uni['eligible'].fillna(False).astype(bool).sum())} mã; chọn {len(sel)}: "
          f"{', '.join(sel['symbol'])}")
    print(uni.loc[~uni["selected"].astype(bool), ["symbol", "reason"]].to_string(index=False))
    if len(sel) < args.n:
        print(f"CẢNH BÁO: chỉ có {len(sel)} < {args.n} mã đạt tiêu chí.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
