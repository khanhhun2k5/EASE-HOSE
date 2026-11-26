"""Tải giá lịch sử (OHLCV ngày) của các mã HOSE trong data/reference/symbol_meta.csv bằng thư viện vnstock.

Mã: 60 mã chia theo ngành (xem src/config.py). Gói khách của vnstock giới hạn 20 request/phút và tự dừng tiến trình
khi vượt, nên mặc định nghỉ 3,5 giây giữa hai mã (≈ 4 phút cho 60 mã).

Cách chạy (từ thư mục gốc đồ án):
    python src/download_vnstock.py                      # cài lại vnstock mới nhất rồi tải
    python src/download_vnstock.py --no-upgrade         # dùng bản vnstock đang cài
    python src/download_vnstock.py --start 2018-01-01 --source tcbs

Lưu ý về phiên bản vnstock:
- Từ thế hệ 6 (vnstock >= 6.0), thư viện phát hành qua kho riêng
  https://vnstocks.com/api/packages (không còn trên PyPI) và mọi nguồn giá cổ phiếu
  Việt Nam (tcbs, ssi, dnse, fhsc) đều cần khoá Open API của chính công ty chứng khoán đó.
  Cất khoá bằng `vnstock keys set <nguồn>` hoặc biến môi trường / file .env, ví dụ
  TCBS_API_KEY, FHSC_API_KEY, SSI_API_KEY + SSI_API_SECRET, DNSE_API_KEY + DNSE_API_SECRET.
- Bản cũ (< 6) dùng lớp Quote với nguồn VCI/KBS, không cần khoá. Script tự nhận biết
  bản đang cài và gọi đúng API.

Kết quả ghi vào data/raw/:
- hose_ohlcv_long.csv : dạng dài (symbol, time, open, high, low, close, volume)
- hose_close_wide.csv : giá đóng cửa dạng rộng, mỗi mã một cột (nghìn đồng)
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from config import SYMBOLS  # noqa: E402  (60 mã, danh mục data/reference/symbol_meta.csv)
EXCHANGE = "HOSE"
VNSTOCK_INDEX = "https://vnstocks.com/api/packages"

# Nguồn giá cổ phiếu của thế hệ 6 và biến môi trường chứa khoá tương ứng.
V6_SOURCE_KEYS = {
    "tcbs": ["TCBS_API_KEY"],
    "fhsc": ["FHSC_API_KEY"],
    "ssi": ["SSI_API_KEY", "SSI_API_SECRET"],
    "dnse": ["DNSE_API_KEY", "DNSE_API_SECRET"],
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def reinstall_vnstock() -> None:
    """Cài lại vnstock bản mới nhất từ kho chính thức của Vnstock."""
    cmd = [
        sys.executable, "-m", "pip", "install", "-U", "--force-reinstall",
        "--extra-index-url", VNSTOCK_INDEX,
        "vnstock>=6.0.0,<7", "vnstock-oapi>=0.1.0,<1",
    ]
    print(">> Cài lại vnstock:", " ".join(cmd[3:]))
    subprocess.check_call(cmd)
    importlib.invalidate_caches()


def vnstock_major_version() -> int:
    import vnstock

    version = getattr(vnstock, "__version__", None)
    if version is None:
        from importlib.metadata import version as pkg_version

        version = pkg_version("vnstock")
    print(f">> vnstock {version}")
    return int(str(version).split(".")[0])


def pick_v6_source(requested: str | None) -> str:
    """Chọn nguồn cho thế hệ 6: nguồn người dùng chỉ định, hoặc nguồn đầu tiên đã có khoá."""
    if requested:
        return requested.lower()
    try:  # nạp .env nếu có python-dotenv (vnstock-oapi cũng tự đọc .env và keyring)
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    for source, env_vars in V6_SOURCE_KEYS.items():
        if all(os.environ.get(v) for v in env_vars):
            return source
    # Không thấy khoá trong biến môi trường: thử tcbs, vnstock-oapi sẽ tìm tiếp trong keyring.
    return "tcbs"


def fetch_one(symbol: str, start: str, end: str, major: int, source: str):
    if major >= 6:
        from vnstock import Market

        return Market(source).equity(symbol).ohlcv(start=start, end=end, interval="1D")
    from vnstock import Quote

    return Quote(source=source.upper(), symbol=symbol).history(start=start, end=end, interval="1D")


def download(start: str, end: str, major: int, source: str, pause: float, symbols: list[str] | None = None):
    """Tải lần lượt từng mã (mặc định ``SYMBOLS``); trả (DataFrame dạng dài hoặc None, {mã lỗi: thông báo})."""
    import pandas as pd

    frames, failed = [], {}
    for symbol in (SYMBOLS if symbols is None else symbols):
        try:
            df = fetch_one(symbol, start, end, major, source)
        except Exception as exc:  # ghi lỗi từng mã, không dừng cả lượt tải
            failed[symbol] = f"{type(exc).__name__}: {exc}"
            print(f"   {symbol}: LỖI - {failed[symbol]}")
            continue
        df = df.copy()
        df.insert(0, "symbol", symbol)
        frames.append(df)
        print(f"   {symbol}: {len(df)} phiên, {df['time'].min()} -> {df['time'].max()}")
        time.sleep(pause)  # giãn nhịp gọi để không bị nguồn chặn
    if not frames:
        return None, failed

    data = pd.concat(frames, ignore_index=True)
    # Bỏ múi giờ (nếu có) để file CSV gọn và dễ đọc lại.
    data["time"] = pd.to_datetime(data["time"])
    if data["time"].dt.tz is not None:
        data["time"] = data["time"].dt.tz_localize(None)
    data["time"] = data["time"].dt.normalize()
    # Một số nguồn (VCI) bỏ qua start/end -> lọc lại cho đúng khoảng ngày yêu cầu.
    data = data[(data["time"] >= start) & (data["time"] <= end)].reset_index(drop=True)
    data["exchange"] = EXCHANGE
    return data, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2015-01-01", help="Ngày bắt đầu YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="Ngày kết thúc YYYY-MM-DD")
    parser.add_argument("--source", default=None,
                        help="Nguồn dữ liệu. v6: tcbs|fhsc|ssi|dnse; bản cũ: vci|kbs")
    parser.add_argument("--no-upgrade", action="store_true", help="Không cài lại vnstock")
    parser.add_argument("--pause", type=float, default=3.5, help="Số giây nghỉ giữa hai mã")
    args = parser.parse_args()

    if not args.no_upgrade:
        reinstall_vnstock()

    major = vnstock_major_version()
    source = pick_v6_source(args.source) if major >= 6 else (args.source or "vci").lower()
    print(f">> Tải {', '.join(SYMBOLS)} (sàn {EXCHANGE}) từ {args.start} đến {args.end}, nguồn {source}")

    data, failed = download(args.start, args.end, major, source, args.pause)
    if data is None:
        print("\nKhông tải được mã nào.")
        if major >= 6:
            print("vnstock >= 6 cần khoá Open API của nguồn: chạy `vnstock keys set tcbs` "
                  "(hoặc fhsc/ssi/dnse) rồi chạy lại, hoặc dùng --source để chọn nguồn đã có khoá.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long_path = OUTPUT_DIR / "hose_ohlcv_long.csv"
    wide_path = OUTPUT_DIR / "hose_close_wide.csv"
    data.to_csv(long_path, index=False, encoding="utf-8-sig")
    close_wide = data.pivot_table(index="time", columns="symbol", values="close").sort_index()
    close_wide = close_wide[[s for s in SYMBOLS if s in close_wide.columns]]
    close_wide.to_csv(wide_path, encoding="utf-8-sig")

    print(f"\nĐã lưu {len(data)} dòng -> {long_path.relative_to(PROJECT_ROOT)}")
    print(f"Đã lưu giá đóng cửa {close_wide.shape} -> {wide_path.relative_to(PROJECT_ROOT)}")
    if failed:
        print("Mã lỗi:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
