-- =============================================================================
-- 04_indexes_views.sql — Chỉ mục, view dùng chung và cập nhật thống kê
-- =============================================================================

-- Chỉ mục ART duy nhất: (1) bảo đảm không trùng (symbol, time) khi dựng lại bảng đặc trưng,
-- (2) tăng tốc tra cứu điểm kiểu "WHERE symbol = ? AND time = ?". Truy vấn lọc theo KHOẢNG
-- (time BETWEEN ...) thì DuckDB vẫn quét tuần tự (đã kiểm tra bằng EXPLAIN) — ở quy mô này
-- quét toàn bảng chỉ mất ~1 ms nên không cần thêm chỉ mục theo time.
CREATE UNIQUE INDEX IF NOT EXISTS idx_features_symbol_time ON daily_features (symbol, time);
CREATE UNIQUE INDEX IF NOT EXISTS idx_monthly_symbol_month ON monthly_stats (symbol, month);

-- View: chỉ giữ phiên trong khung phân tích chung của các mã.
CREATE OR REPLACE VIEW v_analysis AS
SELECT f.*
FROM daily_features f, analysis_config c
WHERE f.time BETWEEN c.analysis_start AND c.analysis_end
  AND f.log_ret IS NOT NULL;

-- View lợi suất log dạng rộng v_returns_wide (mỗi mã một cột, theo sort_order) được tạo bởi
-- build_database.create_returns_wide_view(): DuckDB không cho PIVOT với danh sách cột lấy từ dữ liệu
-- trong view, nên danh sách mã được sinh từ symbol_meta rồi ghi rõ vào câu lệnh.

-- Cập nhật thống kê để bộ tối ưu chọn kế hoạch tốt.
ANALYZE;
CHECKPOINT;
