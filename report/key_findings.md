# Phát hiện chính — Nhân quả ở đuôi lợi suất 60 cổ phiếu HOSE

Dữ liệu: 60 mã, 10 ngành, 2.103 phiên chung (01/06/2018 → 25/11/2026). Phương pháp lõi: EASE (Gnecco và cs., 2021) trên phần dư GARCH-t.
Chi tiết, công thức và hình: `notebook/*/…_finding.ipynb`.

## Kết luận một câu
Thứ tự "nhân quả" EASE trên dữ liệu thô **chủ yếu phản ánh độ nhạy với thị trường (β)**, không phải chuỗi lan truyền giữa các mã.
Sau khi khử nhân tố thị trường, phụ thuộc đuôi còn lại **chỉ nằm trong các ngành hẹp**, và chiều nhân quả giữa các mã chỉ còn ở mức gợi ý.

## 1. Đặc điểm dữ liệu (EDA)
| Phát hiện | Bằng chứng | Hệ quả |
|---|---|---|
| Biên độ ±7% cắt đuôi | 3.048 phiên sát trần/sàn; 54/60 mã có ≥ 14 phiên mỗi biên | Không dùng được Hill; cần phương pháp dựa trên hạng |
| Biến động cụm mạnh | Ljung–Box của $r^2$: p ≈ 0 ở 60/60 mã | Lọc GARCH trước mọi phân tích đuôi |
| Một nhân tố thị trường thống trị | PC1 = 37,5% phương sai, tải dương ở cả 60 mã | Nguồn gây nhiễu chính cho suy luận nhân quả |
| Cùng sập vượt mức Gaussian | $\hat\lambda_L$ 0,39 so với 0,16 (cả 1.770/1.770 cặp) | Phải xử lý riêng đuôi dưới |
| Không có Granger ở thân phân phối | 0/3.540 cặp qua Bonferroni | Quan hệ chủ yếu là đồng thời |

## 2. EASE trên dữ liệu thô (Processing)
- **Mô phỏng:** EASE dùng được với dữ liệu HOSE. Biên ±7% không làm lệch kết quả (ΔSID = −0,001); **nhân tố ẩn** mới là thứ gây hại (+0,042).
- **Dữ liệu thật:** $\overline{\hat\Psi}$ = 0,66, đa số cặp phụ thuộc theo cả hai chiều, tức là mẫu hình **nguyên nhân chung**.
- **Cú sập lan mạnh hơn cú tăng:** $\bar\Gamma^-$ 0,835 so với $\bar\Gamma^+$ 0,611.
- **Thứ tự EASE:** không có gốc duy nhất (NVL 30%, VJC 27%). Mã ít nhạy thị trường đứng đầu; chứng khoán và ngân hàng đứng cuối.
- EASE và LiNGAM **không đồng thuận** (0,51–0,52, ngang ngẫu nhiên): hai họ phương pháp đo hai thứ khác nhau.

## 3. Kiểm chứng (Validation) — phát hiện đảo chiều diễn giải
- **Thứ tự EASE ≈ thứ tự β:** Spearman ρ = 0,75 (p = 7·10⁻¹²), ổn định theo thời gian (ρ 0,60–0,85).
- **Mô hình một nhân tố, không có cạnh nào giữa các mã,** tái hiện thứ tự EASE (ρ = 0,77).
- **Loại thị trường:** số cặp phụ thuộc đuôi giảm từ 1.105 xuống **36**/3.540 (max-T, FWER 5%). Không cặp cùng ngành nào có chiều.
- **Giới hạn phát hiện:** EASE chỉ thấy cạnh khi b ≥ 0,3. Kiểm định hoán đổi **bác bỏ sai 24%** khi hai mã có hệ số tải khác nhau.

## 4. Hiệu chỉnh (Calibration)
- **Quy trình cuối:** GARCH-t → khử 1 nhân tố (PCA/LOO hoặc AE) → EASE với $k = \lfloor n^{0,35}\rfloor = 14$ → Granger trong đuôi.
- **Khử nhân tố là bắt buộc:** SID giảm 0,028 → 0,011–0,014. Với 60 mã, các cách khử cho kết quả ngang nhau.
- **Phụ thuộc còn lại theo ngành hẹp:** 12/12 hệ số lớn nhất là cặp cùng ngành (MBB→TCB, HSG→NKG, SSI→HCM, VHC→ANV…).
  Có 5 cặp mã đầu ngành → mã nhỏ có bất đối xứng (ví dụ VHC→ANV, HPG→NKG), nhưng chưa hiệu chỉnh đa so sánh nên mới là gợi ý.
- **Granger trong đuôi:** trễ 1 có 131 cặp qua FDR, chủ yếu đổ vào mã nhỏ. Theo ngưỡng mô phỏng nghiêm ngặt chỉ còn 7 cặp (nhiễu kỳ vọng ~3,5). Kết quả này nghi do **trễ thanh khoản**, chưa phải lan truyền.
- **Autoencoder phi tuyến:** hệ số tải khác nhau giữa chiều giảm và chiều tăng ở 17/60 mã. AE không tăng R² so với PCA, nhưng giảm 28% độ lệch đuôi của phần dư ở các mã đó (p = 0,003). Thứ tự EASE gần như không đổi (đồng thuận 0,92).

## 5. Các kết luận bị đảo khi mở rộng 20 → 60 mã
- ρ(EASE, β): 0,91 → 0,75. NVL không còn là gốc áp đảo (82% → 30%).
- Hệ số có điều kiện **không còn** là cách khử nhân tố tốt nhất.
- Gợi ý MBB → TCB của bản 20 mã không còn đứng vững.

## 6. Hạn chế và hướng tiếp
1. Kiểm soát thanh khoản và giao dịch không đồng bộ trong Granger trong đuôi.
2. Đưa nhân tố ngành hẹp (chứng khoán, thép, phân bón) vào mô hình nhân tố.
3. Kiểm chứng ngoài mẫu.
