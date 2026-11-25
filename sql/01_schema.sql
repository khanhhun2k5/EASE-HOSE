-- =============================================================================
-- 01_schema.sql — Lược đồ CSDL DuckDB cho dữ liệu giá các mã HOSE (danh mục: data/reference/symbol_meta.csv)
-- Chạy bằng: python src/build_database.py   (chạy lần lượt 01 -> 04 từ thư mục gốc;
-- 05_eda_queries.sql là thư viện truy vấn mẫu, notebook đọc theo tên)
--
-- Nguồn tăng tốc chính (đo trong build_database.py):
--   1. Không phải đọc và phân tích lại CSV mỗi lần truy vấn (DuckDB lưu theo cột, nén sẵn).
--   2. Lợi suất, biến động... được tính sẵn MỘT lần (03_features.sql) thay vì chạy lại
--      hàm cửa sổ mỗi truy vấn.
-- Dữ liệu được nạp SẮP XẾP theo (symbol, time) để chuẩn bị cho khi mở rộng: với bảng lớn
-- hơn một row group (~122.880 dòng), zone map min/max giúp bỏ qua khối khi lọc theo mã/ngày.
-- Ở quy mô hiện tại (~13.500 dòng, 1 row group) lợi ích này chưa xuất hiện.
-- =============================================================================

DROP VIEW  IF EXISTS v_returns_wide;
DROP VIEW  IF EXISTS v_analysis;
DROP TABLE IF EXISTS yearly_stats;
DROP TABLE IF EXISTS monthly_stats;
DROP TABLE IF EXISTS daily_features;
DROP TABLE IF EXISTS ohlcv;
DROP TABLE IF EXISTS symbol_meta;
DROP TABLE IF EXISTS analysis_config;

-- Danh mục mã: một dòng mỗi mã.
CREATE TABLE symbol_meta (
    symbol        VARCHAR PRIMARY KEY,
    company_name  VARCHAR NOT NULL,
    sector        VARCHAR NOT NULL,      -- nhóm ngành (tham khảo, phân loại thủ công)
    exchange      VARCHAR NOT NULL,
    listing_date  DATE    NOT NULL,      -- ngày bắt đầu có giao dịch thực trên HOSE
    sort_order    INTEGER NOT NULL UNIQUE -- thứ tự hiển thị (nhóm theo ngành); quyết định thứ tự cột v_returns_wide
);

-- Giá ngày. Khoá chính (symbol, time) tạo chỉ mục ART -> tra 1 mã/1 ngày rất nhanh
-- và chặn trùng dòng khi nạp lại.
CREATE TABLE ohlcv (
    symbol    VARCHAR  NOT NULL,
    time      DATE     NOT NULL,
    open      DOUBLE   NOT NULL,         -- nghìn đồng, đã điều chỉnh sự kiện quyền
    high      DOUBLE   NOT NULL,
    low       DOUBLE   NOT NULL,
    close     DOUBLE   NOT NULL,
    volume    BIGINT   NOT NULL,         -- cổ phiếu
    exchange  VARCHAR  NOT NULL,
    PRIMARY KEY (symbol, time)
);

-- Tham số phân tích dùng chung cho mọi truy vấn/notebook (một dòng).
CREATE TABLE analysis_config (
    analysis_start DATE NOT NULL,        -- đầu tháng sau ngày niêm yết muộn nhất trong danh mục (xem 02_load_raw.sql)
    analysis_end   DATE NOT NULL
);
