# Từ điển dữ liệu

Dữ liệu giá ngày của 60 cổ phiếu niêm yết trên **HOSE**, nhóm theo 10 ngành (danh mục duy nhất: `data/reference/symbol_meta.csv`, đọc qua `src/config.py`):

| Ngành | Mã |
|---|---|
| Ngân hàng (7) | VCB, BID, CTG, TCB, MBB, STB, VPB |
| Bất động sản (4) | VIC, VHM, NVL, KDH |
| Tiêu dùng (4) | VNM, MSN, SAB, MWG |
| Năng lượng (2) | GAS, PLX |
| Thép (1) | HPG |
| Công nghệ (1) | FPT |
| Chứng khoán (1) | SSI |

Năm mã gốc của giai đoạn đầu (VHM, NVL, STB, VPB, HPG) được giữ để đối chiếu kết quả cũ – mới.

## 1. Luồng dữ liệu

```text
vnstock (nguồn VCI)
  └─ src/download_vnstock.py ──► data/raw/hose_ohlcv_long.csv, hose_close_wide.csv       (thô, không sửa tay)
       └─ src/build_database.py + sql/01..04 ──► data/interim/stocks.duckdb                (CSDL phân tích)
            └─ COPY ──► data/processed/daily_features.parquet                              (bảng đặc trưng)
                 └─ notebook/EDA/01..04 ──► report/figure/eda, report/table/eda
                 └─ notebook/Processing/02 ──► data/processed/garch_std_resid.parquet          (phần dư GARCH)
                      └─ notebook/Processing/01, 02, Processing_finding ──► report/figure/processing, report/table/processing
vnstock (VCI) └─ src/download_market.py ──► data/raw/vnindex_ohlcv.csv                         (Validation)
       └─ notebook/Validation/01 ──► data/processed/garch_std_resid_market.parquet
            └─ notebook/Validation/01..04, Validation_finding ──► report/figure/validation, report/table/validation
            └─ notebook/Calibration/01..03, Calibration_finding ──► report/figure/calibration, report/table/calibration
vnstock (VCI) └─ src/download_extended.py ──► data/raw/extended_ohlcv_long.csv, data/reference/extended_universe.csv
       └─ (40 mã được chọn ghép vào hose_ohlcv_long.csv, khai báo trong symbol_meta.csv)
       └─ notebook/Calibration/04 ──► data/processed/ae_nonlinear_resid.parquet
```

Dựng lại toàn bộ từ đầu:

```bash
python src/download_vnstock.py --no-upgrade   # tải dữ liệu (xem ghi chú phiên bản vnstock trong file)
python src/build_database.py                  # dựng CSDL DuckDB
python src/download_extended.py               # (tuỳ chọn) tái lập bước chọn 40 mã bổ sung (≈ 6 phút)
```

## 2. Quy ước chung

| Mục | Quy ước |
|---|---|
| Giá (`open`, `high`, `low`, `close`, `prev_close`) | **Nghìn đồng**, đã điều chỉnh theo sự kiện quyền (cổ tức, chia tách) bởi nguồn VCI |
| Khối lượng (`volume`) | Số cổ phiếu khớp lệnh trong phiên |
| Giá trị (`trade_value`) | Nghìn đồng (= `close × volume`, xấp xỉ) |
| Thời gian (`time`) | Ngày giao dịch, kiểu `DATE`, giờ Việt Nam, không có múi giờ |
| Lợi suất | Số thập phân (0,01 = 1%), tính theo ngày giao dịch liền trước của cùng mã |
| Khoá | Mỗi dòng là duy nhất theo (`symbol`, `time`) |
| Khung phân tích | 2018-06-01 → phiên cuối (bảng `analysis_config`), khi cả 60 mã cùng có dữ liệu giao dịch thực (TCB niêm yết 04/06/2018 nên thiếu 2 phiên đầu khung; POW thiếu 9, DGC thiếu 6 phiên quanh thời điểm chuyển sàn; 2.120 phiên, 2.103 phiên chung đủ 60 mã) |

## 3. Tệp thô — `data/raw/`

### 3.1. `hose_ohlcv_long.csv` (dạng dài, 167.962 dòng, 60 mã: 20 mã tải 25/11/2026 + 40 mã tải 26/11/2026, cắt tới 25/11/2026)

| Cột | Kiểu | Đơn vị | Mô tả |
|---|---|---|---|
| `symbol` | chuỗi | — | Mã cổ phiếu |
| `time` | ngày | — | Ngày giao dịch (YYYY-MM-DD) |
| `open` | số thực | nghìn đồng | Giá mở cửa |
| `high` | số thực | nghìn đồng | Giá cao nhất |
| `low` | số thực | nghìn đồng | Giá thấp nhất |
| `close` | số thực | nghìn đồng | Giá đóng cửa |
| `volume` | số nguyên | cổ phiếu | Khối lượng khớp lệnh |
| `exchange` | chuỗi | — | Sàn niêm yết, luôn là `HOSE` |

### 3.2. `hose_close_wide.csv` (dạng rộng)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `time` | ngày | Ngày giao dịch |
| một cột mỗi mã theo `sort_order` (60 cột, 2.969 dòng) | số thực | Giá đóng cửa (nghìn đồng); để trống nếu mã chưa có dữ liệu ngày đó |

### 3.3. `vnindex_ohlcv.csv` (giai đoạn Validation, Calibration)

Tải bằng `python src/download_market.py` (vnstock 3.x, nguồn VCI, không cần khoá), từ 18/06/2014 đến phiên cuối.

| Tệp | Cột | Nội dung |
|---|---|---|
| `vnindex_ohlcv.csv` | `time`, `open`, `high`, `low`, `close`, `volume` | Chỉ số VN-Index (điểm), khối lượng toàn sàn |

Nhân tố thị trường dùng trong phân tích có hai dạng: `VNI` (lợi suất log VN-Index, lọc GARCH) — đại diện chính thức nhưng **chứa chính các mã nghiên cứu**
(gia quyền vốn hoá); và thị trường **leave-one-out** $\bar z_{-i,t}$ = trung bình phần dư GARCH của 19 mã khác $i$ (`validation.loo_market`), không chứa mã $i$.
(Giai đoạn 5 mã từng dùng nhân tố "ngoài mẫu" từ 12 mã lớn khác; các mã đó nay đã nằm trong danh mục 20 mã nên cách này không còn dùng.)

### 3.4. `extended_ohlcv_long.csv` và `data/reference/extended_universe.csv` (Calibration 04)

| Tệp | Cột | Ý nghĩa |
|---|---|---|
| `extended_ohlcv_long.csv` | như `hose_ohlcv_long.csv` | OHLCV của 90 mã ứng viên HOSE ngoài 20 mã giai đoạn trước (`download_extended.CANDIDATES`), tải 26/11/2026; 40 mã được chọn đã ghép vào `hose_ohlcv_long.csv` |
| `extended_universe.csv` | `symbol`, `sector`, `first_date`, `missing_share`, `n_breaches`, `median_value`, `eligible`, `reason`, `rank`, `selected` | Kết quả `select_universe`: tỷ lệ phiên thiếu trong cửa sổ từ 01/06/2018; số phiên $|r| > 7{,}5\%$; giá trị giao dịch trung vị 2 năm cuối (tỷ đồng); lý do loại; 40 mã `selected` |

Ngành trong `extended_universe.csv` ghi theo hoạt động chính, chỉ để mô tả. DGC (2 phiên) và POW (1 phiên) có lợi suất vượt 7,5% trong cửa sổ
(giai đoạn trước khi lên HOSE) nhưng vẫn trong ngưỡng ≤ 2 phiên.

## 4. CSDL DuckDB — `data/interim/stocks.duckdb`

### 4.1. `symbol_meta` — danh mục mã (60 dòng; tên công ty của 40 mã bổ sung lấy từ danh mục niêm yết vnstock, nguồn KBS)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `symbol` | VARCHAR, khoá chính | Mã cổ phiếu |
| `company_name` | VARCHAR | Tên doanh nghiệp |
| `sector` | VARCHAR | Nhóm ngành phân loại thủ công: Ngân hàng (7 mã), Bất động sản (4), Tiêu dùng (4), Năng lượng (2), Thép (1), Công nghệ (1), Chứng khoán (1) |
| `exchange` | VARCHAR | `HOSE` |
| `listing_date` | DATE | Ngày bắt đầu giao dịch trên HOSE (xem mục 6); mã không ghi ngày trong CSV lấy ngày đầu tiên có dữ liệu |
| `sort_order` | INTEGER, duy nhất | Thứ tự hiển thị (nhóm theo ngành); quyết định thứ tự cột của `v_returns_wide` |

### 4.2. `ohlcv` — giá ngày gốc (167.961 dòng; bỏ 1 dòng GEX có giá đóng cửa = 0)

Cùng cột với `hose_ohlcv_long.csv`. Khoá chính (`symbol`, `time`); dữ liệu được lưu sắp xếp theo khoá.

### 4.3. `daily_features` — đặc trưng ngày (167.961 dòng, 19 cột)

| Cột | Kiểu | Công thức / ý nghĩa |
|---|---|---|
| `symbol`, `time`, `open`, `high`, `low`, `close`, `volume` | — | Như bảng `ohlcv` |
| `prev_close` | DOUBLE | `close` của phiên liền trước (cùng mã) |
| `ret` | DOUBLE | Lợi suất giản đơn `close / prev_close − 1`. **NULL** ở phiên đầu tiên và ở phiên mà phiên trước đó chưa niêm yết |
| `log_ret` | DOUBLE | Lợi suất log `ln(close / prev_close)`; NULL như `ret` |
| `range_pct` | DOUBLE | Biên độ trong phiên `(high − low) / prev_close`; NULL như `ret` |
| `gap_ret` | DOUBLE | Chênh lệch mở cửa `open / prev_close − 1`; NULL như `ret` |
| `trade_value` | DOUBLE | `close × volume` (nghìn đồng) |
| `is_traded` | BOOLEAN | `volume > 0` |
| `is_listed` | BOOLEAN | `time ≥ listing_date` |
| `vol_20d` | DOUBLE | Độ lệch chuẩn mẫu của `log_ret` trong 20 phiên gần nhất (gồm phiên hiện tại), **chưa năm hoá**; nhân `√252` để năm hoá. NULL khi cửa sổ có ít hơn 20 giá trị `log_ret` |
| `rel_volume` | DOUBLE | `volume` / trung bình `volume` của 20 phiên **trước** đó. NULL khi 20 phiên đó chưa đều là phiên đã niêm yết và có giao dịch |
| `hit_ceiling` | BOOLEAN | `ret ≥ 0,065` **và** `close = high`: đóng cửa ở giá trần (giá trần được làm tròn theo bước giá nên lợi suất thực có thể chỉ 6,5–7%) |
| `hit_floor` | BOOLEAN | `ret ≤ −0,065` **và** `close = low`: đóng cửa ở giá sàn |

### 4.4. `monthly_stats` — thống kê tháng (chỉ các phiên `is_listed`)

| Cột | Mô tả |
|---|---|
| `symbol`, `month` | Mã, ngày đầu tháng |
| `n_days` | Số phiên có lợi suất |
| `ret_month` | Lợi suất tháng `exp(Σ log_ret) − 1` |
| `vol_ann` | Độ lệch chuẩn `log_ret` × √252 |
| `avg_volume`, `total_value` | Khối lượng bình quân phiên, tổng giá trị giao dịch (nghìn đồng) |
| `close_eom` | Giá đóng cửa phiên cuối tháng |

### 4.5. `yearly_stats` — thống kê năm (chỉ các phiên `is_listed`)

`symbol`, `year`, `n_days`, `ret_year` (lợi suất năm), `vol_ann`, `skew` (độ lệch), `excess_kurt` (độ nhọn vượt chuẩn),
`worst_day`, `best_day` (lợi suất ngày thấp/cao nhất), `n_ceiling`, `n_floor` (số phiên chạm trần/sàn).

### 4.6. Bảng phụ và view

| Đối tượng | Mô tả |
|---|---|
| `analysis_config` | Một dòng: `analysis_start` = 2018-06-01, `analysis_end` = phiên cuối cùng |
| `v_analysis` | View: `daily_features` trong khung phân tích, chỉ các dòng có `log_ret` |
| `v_returns_wide` | View: `log_ret` dạng rộng (`time` + 60 cột mã theo `sort_order`), dùng cho tương quan và PCA |

Chỉ mục: `idx_features_symbol_time` (duy nhất), `idx_features_time`, `idx_monthly_symbol_month`.

## 5. Tệp đã xử lý — `data/processed/daily_features.parquet`

Bản Parquet của `daily_features`, cùng cột và kiểu, dùng khi không cần mở DuckDB (ví dụ `pd.read_parquet`).

### 5.1. `data/processed/garch_std_resid.parquet` — phần dư GARCH chuẩn hoá

Tạo bởi `notebook/Processing/02_ease_real_data.ipynb` (hàm `tail_preprocessing.garch_standardize`). Chỉ mục `time`
(2.103 phiên chung của 60 mã trong khung phân tích), 60 cột theo `sort_order` của danh mục.

| Giá trị | Công thức |
|---|---|
| $z_{i,t}$ (không đơn vị) | $(r_{i,t} - \hat\mu_i)/\hat\sigma_{i,t}$ với $\hat\sigma_{i,t}$ từ GARCH(1,1), nhiễu Student-t, khớp trên $100\,r_{i,t}$ |

Dùng cho EASE vì đã loại hiệu ứng ARCH (Ljung–Box của $z^2$ không còn hiệu ứng ARCH đáng kể).

### 5.2. `data/processed/garch_std_resid_market.parquet` — phần dư GARCH kèm nhân tố thị trường

Tạo bởi `notebook/Validation/01_market_factor.ipynb`. Cùng cách tính như 5.1, 61 cột: 60 mã + `VNI` (lợi suất log VN-Index),
chỉ giữ các phiên cả 61 chuỗi đều có giá trị. Dùng cho Validation và Calibration.

### 5.3. (đã bỏ) `garch_std_resid_extended.parquet`

Trước khi mở rộng mẫu, Calibration 04 ghép 20 mã nghiên cứu với 40 mã bổ sung thành một bảng riêng. Từ khi 60 mã là mẫu nghiên cứu,
mọi notebook dùng chung 5.1 nên tệp này không còn được tạo.

### 5.4. `data/processed/ae_nonlinear_resid.parquet` — phần dư sau khử nhân tố phi tuyến

Tạo bởi `notebook/Calibration/04`. 60 cột như 5.1: $e_{i,t} = z_{i,t} - \hat z_{i,t}$ với $\hat z$ là dự báo **ngoài mẫu** (cross-fitting 5 khối thời gian)
của autoencoder hệ số tải phi tuyến một nhân tố (`nonlinear_factor`, `arch="nonlinear"`, $K = 1$, trung bình 5 seed).

## 6. Ghi chú chất lượng dữ liệu

Với 40 mã bổ sung (26/11/2026): GEX 12/08/2016 giá đóng cửa = 0 (loại khi nạp, `sql/02_load_raw.sql`); HCM 05/06/2020 giá mở cửa thấp hơn giá thấp nhất và NLG 21–24/08/2015 giá mở cửa
cao hơn giá cao nhất (giữ, chỉ ảnh hưởng `gap_ret`); trước 2018 ANV, CTS, DGC, GEX có phiên |r| > 7,5% (giao dịch ở sàn khác); trong khung, DGC 2 phiên và POW 1 phiên.
Chuỗi giá PNJ của nguồn VCI giảm liên tục về cuối mẫu (−85% từ đỉnh) mà không có phiên nào vượt biên độ; chưa đối chiếu được với nguồn khác.

| Vấn đề | Chi tiết | Cách xử lý |
|---|---|---|
| VHM trước niêm yết | 629 phiên từ 2015 đến 16/05/2018 có giá đứng yên, khối lượng gần như bằng 0 (19 mã còn lại không có phiên nào trước niêm yết trong dữ liệu) | `is_listed = false`; không tính lợi suất ở phiên 17/05/2018 (giá trước đó là giá "ảo") |
| Phiên khối lượng 0 | 23–24/01/2018 có 19 mã (trừ TCB, chưa niêm yết) khối lượng 0; 18 và 21/05/2018 ở VHM | Nằm ngoài khung phân tích |
| Biên độ ±7% | Lợi suất log ngày bị chặn: trên ≈ ln(1,07) = 0,068, dưới ≈ ln(0,93) = −0,073 (lệch nhẹ do bước giá). Trong khung phân tích, 0,28–2,31% số phiên đóng cửa ở giá trần (thấp nhất VNM, cao nhất SSI) và 0,33–2,45% ở giá sàn (thấp nhất VNM, cao nhất NVL) tuỳ mã | Giữ nguyên; cần lưu ý khi phân tích đuôi phân phối (notebook 03) |
| Ngày đứng giá | 5,6–12,5% số phiên có lợi suất đúng bằng 0 (thấp nhất FPT, cao nhất KDH) do bước giá rời rạc | Giữ nguyên |
| Giá trị thiếu, trùng lặp, OHLC sai | Không phát hiện trường hợp nào (notebook 01) | — |
| Ngày niêm yết | NVL (28/12/2016), VPB (17/08/2017), VHM (17/05/2018), TCB (04/06/2018), SAB (06/12/2016), PLX (21/04/2017) khớp với ngày đầu tiên có dữ liệu (MWG ghi 14/07/2014, trước điểm bắt đầu của dữ liệu). STB (12/07/2006) và HPG (15/11/2007) ghi theo hiểu biết chung, **chưa đối chiếu nguồn chính thức** | Không ảnh hưởng phân tích vì dữ liệu của hai mã này (và các mã niêm yết trước 2015) bắt đầu từ 05/01/2015 |
