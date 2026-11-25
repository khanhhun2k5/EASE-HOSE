-- =============================================================================
-- 05_eda_queries.sql — Truy vấn mẫu cho EDA (notebook đọc từng khối theo tên)
-- Mỗi truy vấn bắt đầu bằng dòng "-- name: <tên>" để Python tách ra dùng lại.
-- =============================================================================

-- name: profile_by_symbol
SELECT symbol,
       COUNT(*)                                   AS n_rows,
       MIN(time)                                  AS first_date,
       MAX(time)                                  AS last_date,
       SUM((NOT is_listed)::INT)                  AS n_pre_listing,
       SUM((NOT is_traded)::INT)                  AS n_zero_volume,
       SUM((high < GREATEST(open, close) OR low > LEAST(open, close))::INT) AS n_invalid_ohlc,
       SUM(hit_ceiling::INT)                      AS n_ceiling,
       SUM(hit_floor::INT)                        AS n_floor
FROM daily_features
GROUP BY symbol
ORDER BY symbol;

-- name: missing_sessions
-- Phiên có ở ít nhất một mã nhưng thiếu ở mã khác (trong khung phân tích).
WITH cal AS (SELECT DISTINCT time FROM v_analysis),
     sym AS (SELECT symbol FROM symbol_meta)
SELECT s.symbol, COUNT(*) AS n_missing
FROM cal CROSS JOIN sym s
LEFT JOIN v_analysis a ON a.time = cal.time AND a.symbol = s.symbol
WHERE a.symbol IS NULL
GROUP BY s.symbol
ORDER BY s.symbol;

-- name: return_summary
SELECT symbol,
       COUNT(*)                                   AS n,
       AVG(log_ret) * 252                         AS mean_ann,
       STDDEV_SAMP(log_ret) * SQRT(252)           AS vol_ann,
       SKEWNESS(log_ret)                          AS skew,
       KURTOSIS(log_ret)                          AS excess_kurt,
       QUANTILE_CONT(log_ret, 0.01)               AS q01,
       QUANTILE_CONT(log_ret, 0.99)               AS q99,
       MIN(log_ret)                               AS min,
       MAX(log_ret)                               AS max
FROM v_analysis
GROUP BY symbol
ORDER BY symbol;

-- name: worst_days
SELECT symbol, time, close, ret, volume
FROM v_analysis
QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ret) <= 5
ORDER BY symbol, ret;
