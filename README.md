# Suy luận nhân quả ở đuôi phân phối lợi suất cổ phiếu HOSE

Đồ án II — phân tích các cú sốc cực trị (tăng/giảm rất mạnh) của **60 cổ phiếu trên HOSE** thuộc 10 ngành (2018–2026) và câu hỏi:
*cú sốc cực trị của mã này có kéo theo cú sốc của mã khác không, và theo chiều nào?*

Phương pháp chính là **hệ số nhân quả đuôi** và thuật toán **EASE** (Gnecco, Meinshausen, Peters & Engelke, 2021),
đối chiếu với các phương pháp nền (Pairwise LiNGAM, ICA-LiNGAM, Rank PC), rồi kiểm chứng và hiệu chỉnh cho đặc thù thị trường
Việt Nam: biên độ giá ±7%, biến động cụm, nhân tố thị trường chung.

## Quy trình

| Giai đoạn | Thư mục | Nội dung |
|---|---|---|
| Dữ liệu | `src/download_vnstock.py`, `sql/`, `src/build_database.py` | Tải giá ngày (vnstock, nguồn VCI), dựng CSDL DuckDB, bảng đặc trưng |
| EDA | `notebook/EDA/` | Mô tả, phân phối, đuôi dày, biên độ ±7%, biến động cụm, phụ thuộc đuôi, PCA, Granger |
| Processing | `notebook/Processing/` | Lọc GARCH-t, hệ số nhân quả đuôi Ψ/Γ, EASE, mô phỏng và so sánh baseline |
| Validation | `notebook/Validation/` | Nhân tố thị trường, kiểm định hoán vị, ổn định theo thời gian, mô phỏng hiệu chỉnh |
| Calibration | `notebook/Calibration/` | Chọn siêu tham số (GARCH, số nhân tố, k), EASE sau khi khử nhân tố, Granger trong đuôi, khử nhân tố phi tuyến bằng autoencoder trên bảng 60 mã (04) |

Mỗi giai đoạn có một notebook tổng hợp phát hiện (`*_finding.ipynb`). Hình lưu ở `report/figure/<giai đoạn>/`,
bảng ở `report/table/<giai đoạn>/`. Từ điển dữ liệu: [`data/data_dictionary.md`](data/data_dictionary.md).
Danh mục 60 mã và ngành (20 mã vốn hoá lớn + 40 mã bổ sung chọn bằng tiêu chí dữ liệu): [`data/reference/symbol_meta.csv`](data/reference/symbol_meta.csv). Tài liệu tham khảo: [`Paper/README.md`](Paper/README.md).

## Cấu trúc mã nguồn

| Module | Vai trò |
|---|---|
| `src/config.py` | Danh mục mã (đọc từ `data/reference/symbol_meta.csv`) |
| `src/eda_utils.py` | Kết nối CSDL, style hình, lưu bảng/hình |
| `src/causal_tail.py` | Hệ số nhân quả đuôi (vector hoá), EASE, SID, mô phỏng SCM đuôi dày |
| `src/causal_baselines.py` | Baseline LiNGAM/Rank PC, nghiên cứu mô phỏng song song |
| `src/tail_preprocessing.py` | Lọc GARCH, ước lượng GPD, bootstrap |
| `src/validation.py` | Nhân tố leave-one-out, kiểm định hoán vị (max-T, hoán đổi), mô phỏng nhân tố |
| `src/download_extended.py` | Tải 90 ứng viên HOSE, chọn 40 mã bổ sung theo tiêu chí dữ liệu (đủ lịch sử, thuần HOSE, thanh khoản cao) |
| `src/nonlinear_factor.py` | PCA, autoencoder tuyến tính / ReLU (Spilak–Härdle) / hệ số tải phi tuyến (theo Gu–Kelly–Xiu), cross-fitting, R² đuôi, β tăng/giảm, mô phỏng nhân tố bất đối xứng |
| `src/calibration.py` | Số nhân tố Ahn–Horenstein, mô hình nhân tố Kendall, hệ số đuôi có điều kiện thị trường, Granger trong đuôi |

## Cài đặt và chạy

```bash
pip install -r requirements.txt          # xem ghi chú cài lingam trong requirements.txt
python src/download_vnstock.py --no-upgrade   # tải giá 60 mã (vnstock 3.x, không cần khoá API; ≈ 4 phút)
python src/download_market.py                 # tải VN-Index
python src/build_database.py                  # dựng data/interim/stocks.duckdb và data/processed/*.parquet
python src/download_extended.py               # (tuỳ chọn) tái lập bước chọn 40 mã bổ sung (≈ 6 phút do giới hạn 20 request/phút)
python -m pytest test                         # kiểm thử
```

Sau đó chạy notebook theo thứ tự EDA → Processing → Validation → Calibration. Dữ liệu thô đã có sẵn trong `data/raw/`
(tải ngày 25–26/11/2026), nên có thể bỏ qua bước tải.

## Tình trạng

Ngày 26/11/2026 mẫu nghiên cứu được mở rộng từ 20 lên **60 mã** (40 mã bổ sung chọn bằng `src/download_extended.py`) và toàn bộ 18 notebook
(EDA → Processing → Validation → Calibration, kể cả các notebook tổng hợp `*_finding.ipynb`) đã chạy lại và viết lại phần nhận xét cho 60 mã.
Chạy toàn bộ mất khoảng 3 giờ trên 12 lõi (nặng nhất: mô phỏng $p = 60$ ở Processing 01 ≈ 55 phút, Granger trong đuôi ở Calibration 03).
