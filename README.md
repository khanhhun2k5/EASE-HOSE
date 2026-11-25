# Suy luận nhân quả ở đuôi phân phối lợi suất cổ phiếu HOSE

Đồ án II — phân tích các cú sốc cực trị (tăng/giảm rất mạnh) của **20 cổ phiếu vốn hoá lớn trên HOSE** (2018–2026) và câu hỏi:
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
| Calibration | `notebook/Calibration/` | Chọn siêu tham số (GARCH, số nhân tố, k), EASE sau khi khử nhân tố, Granger trong đuôi |

Mỗi giai đoạn có một notebook tổng hợp phát hiện (`*_finding.ipynb`). Hình lưu ở `report/figure/<giai đoạn>/`,
bảng ở `report/table/<giai đoạn>/`. Từ điển dữ liệu: [`data/data_dictionary.md`](data/data_dictionary.md).
Danh mục 20 mã và ngành: [`data/reference/symbol_meta.csv`](data/reference/symbol_meta.csv). Tài liệu tham khảo: [`Paper/README.md`](Paper/README.md).

## Cấu trúc mã nguồn

| Module | Vai trò |
|---|---|
| `src/config.py` | Danh mục mã (đọc từ `data/reference/symbol_meta.csv`) |
| `src/eda_utils.py` | Kết nối CSDL, style hình, lưu bảng/hình |
| `src/causal_tail.py` | Hệ số nhân quả đuôi (vector hoá), EASE, SID, mô phỏng SCM đuôi dày |
| `src/causal_baselines.py` | Baseline LiNGAM/Rank PC, nghiên cứu mô phỏng song song |
| `src/tail_preprocessing.py` | Lọc GARCH, ước lượng GPD, bootstrap |
| `src/validation.py` | Nhân tố leave-one-out, kiểm định hoán vị (max-T, hoán đổi), mô phỏng nhân tố |
| `src/calibration.py` | Số nhân tố Ahn–Horenstein, mô hình nhân tố Kendall, hệ số đuôi có điều kiện thị trường, Granger trong đuôi |

## Cài đặt và chạy

```bash
pip install -r requirements.txt          # xem ghi chú cài lingam trong requirements.txt
python src/download_vnstock.py --no-upgrade   # tải giá 20 mã (vnstock 3.x, không cần khoá API)
python src/download_market.py                 # tải VN-Index
python src/build_database.py                  # dựng data/interim/stocks.duckdb và data/processed/*.parquet
python -m pytest test                         # kiểm thử
```

Sau đó chạy notebook theo thứ tự EDA → Processing → Validation → Calibration. Dữ liệu thô đã có sẵn trong `data/raw/`
(tải ngày 25–26/11/2026), nên có thể bỏ qua bước tải.

## Tình trạng

Giai đoạn EDA đã hoàn tất cho 20 mã. Các notebook Processing, Validation, Calibration đang được cập nhật cho bộ 20 mã;
phần nhận xét và notebook tổng hợp của ba giai đoạn này sẽ được bổ sung.
