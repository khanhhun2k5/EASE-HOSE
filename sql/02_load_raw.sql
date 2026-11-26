-- =============================================================================
-- 02_load_raw.sql — Nạp dữ liệu thô từ data/raw vào DuckDB
-- Đường dẫn tương đối tính từ thư mục gốc đồ án.
-- =============================================================================

-- Nạp SẮP XẾP theo (symbol, time): dữ liệu cùng mã nằm liền nhau trên đĩa, zone map của
-- cột symbol/time hẹp lại -> có tác dụng bỏ qua khối khi bảng vượt một row group.
INSERT INTO ohlcv
SELECT symbol,
       CAST(time AS DATE) AS time,
       open, high, low, close,
       CAST(volume AS BIGINT) AS volume,
       exchange
FROM read_csv('data/raw/hose_ohlcv_long.csv', header = true, auto_detect = true)
-- Giá đóng cửa ≤ 0 là lỗi nguồn (GEX 12/08/2016: close = 0 trong khi open/high/low > 0) → bỏ dòng,
-- nếu không LN(close / prev_close) = ±∞ làm hỏng mọi thống kê. Ghi trong data/data_dictionary.md mục 6.
WHERE close > 0
ORDER BY symbol, time;

-- Danh mục mã từ file tham chiếu (nguồn duy nhất, sửa tay). listing_date để trống -> ngày đầu tiên có
-- dữ liệu của mã (nguồn VCI bắt đầu từ 18/06/2014, sau ngày niêm yết thật nên không có giá ảo).
INSERT INTO symbol_meta
SELECT s.symbol, s.company_name, s.sector, s.exchange,
       COALESCE(CAST(s.listing_date AS DATE), f.first_day) AS listing_date,
       s.sort_order
FROM read_csv('data/reference/symbol_meta.csv', header = true,
              columns = {'symbol': 'VARCHAR', 'company_name': 'VARCHAR', 'sector': 'VARCHAR',
                         'exchange': 'VARCHAR', 'listing_date': 'VARCHAR', 'sort_order': 'INTEGER'}) s
LEFT JOIN (SELECT symbol, MIN(time) AS first_day FROM ohlcv GROUP BY symbol) f USING (symbol);

-- Mọi mã trong ohlcv phải có trong danh mục (lỗi sớm nếu tải thêm mã mà quên khai báo).
SELECT CASE WHEN COUNT(*) > 0 THEN error('Mã có trong ohlcv nhưng thiếu trong symbol_meta.csv: ' || STRING_AGG(symbol, ', '))
       END
FROM (SELECT DISTINCT symbol FROM ohlcv) o ANTI JOIN symbol_meta m USING (symbol);

-- Khung phân tích: từ đầu tháng sau ngày niêm yết muộn nhất (VHM 17/05/2018; TCB 04/06/2018 -> lợi suất
-- TCB có từ 05/06/2018) tới phiên cuối cùng. LƯU Ý: v_returns_wide GIỮ các phiên thiếu (giá trị NULL, ví dụ TCB
-- 01/06 và 04/06/2018); nơi dùng cần .dropna() — các hàm phân tích đều từ chối NaN.
INSERT INTO analysis_config
SELECT DATE '2018-06-01', MAX(time) FROM ohlcv;
