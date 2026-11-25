-- =============================================================================
-- 03_features.sql — Tính sẵn (materialize) các đại lượng dẫn xuất
-- Hàm cửa sổ (LAG, rolling) tính MỘT lần lúc dựng CSDL thay vì mỗi lần truy vấn.
-- =============================================================================

CREATE TABLE daily_features AS
WITH base AS (
    SELECT o.*,
           LAG(close) OVER w AS prev_close,
           -- Phiên liền trước đã niêm yết chưa? Nếu chưa thì giá trước đó là giá "ảo"
           -- (VHM trước 17/05/2018) -> không tính lợi suất cho phiên này.
           COALESCE(LAG(o.time) OVER w >= m.listing_date, FALSE) AS prev_listed,
           m.listing_date
    FROM ohlcv o
    JOIN symbol_meta m USING (symbol)
    WINDOW w AS (PARTITION BY symbol ORDER BY time)
),
rets AS (
    SELECT *,
           CASE WHEN prev_listed THEN close / prev_close - 1 END       AS ret,       -- lợi suất giản đơn
           CASE WHEN prev_listed THEN LN(close / prev_close) END       AS log_ret,   -- lợi suất log
           CASE WHEN prev_listed THEN (high - low) / prev_close END    AS range_pct, -- biên độ trong phiên
           CASE WHEN prev_listed THEN open / prev_close - 1 END        AS gap_ret,   -- chênh lệch mở cửa
           close * volume                                 AS trade_value,  -- nghìn đồng
           (volume > 0)                                   AS is_traded,
           (time >= listing_date)                         AS is_listed
    FROM base
)
SELECT symbol, time, open, high, low, close, volume, prev_close,
       ret, log_ret, range_pct, gap_ret, trade_value, is_traded, is_listed,
       -- Biến động lịch sử 20 phiên (độ lệch chuẩn lợi suất log, chưa năm hoá).
       -- NULL khi cửa sổ chưa đủ 20 lợi suất hợp lệ.
       CASE WHEN COUNT(log_ret) OVER w20 = 20 THEN STDDEV_SAMP(log_ret) OVER w20 END AS vol_20d,
       -- Khối lượng tương đối so với trung bình 20 phiên trước đó; chỉ tính khi cả 20 phiên
       -- trước đều đã niêm yết và có giao dịch (tránh mẫu số là giai đoạn KL ≈ 0 của VHM).
       CASE WHEN COUNT(CASE WHEN is_listed AND is_traded THEN 1 END) OVER wprev = 20
            THEN volume / NULLIF(AVG(volume) OVER wprev, 0) END                  AS rel_volume,
       -- Chạm trần/sàn ±7%. Giá trần/sàn được làm tròn theo bước giá nên lợi suất thực
       -- có thể chỉ 6,5–7%; vì vậy yêu cầu thêm giá đóng cửa = giá cao (thấp) nhất phiên.
       (ret >=  0.065 AND close = high) AS hit_ceiling,
       (ret <= -0.065 AND close = low)  AS hit_floor
FROM rets
WINDOW w20   AS (PARTITION BY symbol ORDER BY time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
       wprev AS (PARTITION BY symbol ORDER BY time ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
ORDER BY symbol, time;

-- Thống kê tháng: tổng hợp sẵn cho biểu đồ/so sánh theo thời gian.
CREATE TABLE monthly_stats AS
SELECT symbol,
       DATE_TRUNC('month', time)::DATE      AS month,
       COUNT(*)                             AS n_days,
       EXP(SUM(log_ret)) - 1                AS ret_month,
       STDDEV_SAMP(log_ret) * SQRT(252)     AS vol_ann,
       AVG(volume)                          AS avg_volume,
       SUM(trade_value)                     AS total_value,
       LAST(close ORDER BY time)            AS close_eom
FROM daily_features
WHERE is_listed AND log_ret IS NOT NULL
GROUP BY ALL
ORDER BY symbol, month;

-- Thống kê năm.
CREATE TABLE yearly_stats AS
SELECT symbol,
       YEAR(time)                           AS year,
       COUNT(*)                             AS n_days,
       EXP(SUM(log_ret)) - 1                AS ret_year,
       STDDEV_SAMP(log_ret) * SQRT(252)     AS vol_ann,
       SKEWNESS(log_ret)                    AS skew,
       KURTOSIS(log_ret)                    AS excess_kurt,
       MIN(ret)                             AS worst_day,
       MAX(ret)                             AS best_day,
       SUM(hit_ceiling::INT)                AS n_ceiling,
       SUM(hit_floor::INT)                  AS n_floor
FROM daily_features
WHERE is_listed AND log_ret IS NOT NULL
GROUP BY ALL
ORDER BY symbol, year;
