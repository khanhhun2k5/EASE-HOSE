# Tài liệu tham khảo

Bản PDF không được đưa vào kho mã (bản quyền thuộc tác giả/nhà xuất bản). Tải theo DOI hoặc arXiv dưới đây và đặt vào
thư mục này nếu cần đọc kèm.

| # | Tài liệu | Liên kết | Dùng ở |
|---|---|---|---|
| 1 | Gnecco, N., Meinshausen, N., Peters, J., Engelke, S. (2021). *Causal discovery in heavy-tailed models*. Annals of Statistics 49(3), 1755–1778. | [doi:10.1214/20-AOS2021](https://doi.org/10.1214/20-AOS2021) | Phương pháp chính: hệ số nhân quả đuôi, thuật toán EASE (Processing, Validation) |
| 2 | Bodik, J., Paluš, M., Pawlas, Z. (2024). *Causality in extremes of time series*. Extremes 27, 67–121. | [doi:10.1007/s10687-023-00479-5](https://doi.org/10.1007/s10687-023-00479-5) | Nhân quả đuôi theo thời gian (Calibration) |
| 3 | Bodik, J., Pasche, O. C. *Granger causality in extremes*. | [arXiv:2407.09632](https://arxiv.org/abs/2407.09632) | Granger trong đuôi, kiểm định bootstrap khối (Calibration 03) |
| 4 | Xu, T., Shu, L., Chen, Y. (2024). *Tail risk dynamics under price-limited constraint: a censored autoregressive conditional Fréchet model*. Entropy 26(7), 555. | [doi:10.3390/e26070555](https://doi.org/10.3390/e26070555) | Biên độ ±7%: CAcF có kiểm duyệt cho lỗ lớn nhất thị trường, GARCH–t có kiểm duyệt từng mã (`src/price_limit.py`, Calibration 05) |
| 5 | Passeggeri, R., Wintenberger, O. (2024). *Extremes for stationary regularly varying random fields over arbitrary index sets*. Extremes 27. | [doi:10.1007/s10687-023-00481-x](https://doi.org/10.1007/s10687-023-00481-x) | Nền tảng lý thuyết biến thiên đều |
| 6 | Agarwal, A., Singh, R. (2026). *Causal inference with noisy covariates: heteroscedastic measurement error and differential privacy*. Econometrica 94(5), 1887–1906. | Econometrica, tập 94 số 5 | Tham khảo suy luận nhân quả với biến có nhiễu |
| 7 | Yang, S., Liang, V. C., Meel, K. S. *INC: A scalable incremental weighted sampler*. | [arXiv:2306.10824](https://arxiv.org/abs/2306.10824) | Tham khảo phụ |
| 8 | Gu, S., Kelly, B., Xiu, D. (2021). *Autoencoder asset pricing models*. Journal of Econometrics 222(1), 429–450. | [doi:10.1016/j.jeconom.2020.07.009](https://doi.org/10.1016/j.jeconom.2020.07.009) | Nhân tố là danh mục tuyến tính, AE tuyến tính ≡ PCA, R² tổng ngoài mẫu (Calibration 04) |
| 9 | Spilak, B., Härdle, W. K. (2022). *Does non-linear factorization of financial returns help build better and stabler portfolios?* | [arXiv:2204.02757](https://arxiv.org/abs/2204.02757) | AE ReLU ràng buộc không âm, phạt trực giao/tương quan (Calibration 04) |
| 10 | Bodik, J., Chavez-Demoulin, V. (2026). *Structural restrictions in local causal discovery: identifying direct causes of a target variable*. Biometrika 113(1), asaf042. | [doi:10.1093/biomet/asaf042](https://doi.org/10.1093/biomet/asaf042) | Tham khảo: khám phá nguyên nhân trực tiếp của một biến mục tiêu (định hướng kiểm tra chéo, chưa dùng trong code) |

Các phương pháp khác được trích dẫn trực tiếp trong docstring của `src/` và trong notebook (Ahn & Horenstein 2013;
Fan, Liu, Wang 2018; Pasche và cs. 2023; Peters & Bühlmann 2015; Shimizu và cs. 2006; Hyvärinen & Smith 2013;
Harris & Drton 2013; Westfall & Young 1993; Benjamini & Hochberg 1995).
