-- 🚀 OSCEAN 统一元数据数据库 Schema v4.0
-- 采用"统一物理数据库 + 逻辑分层分类"架构
-- 目标: 实现高性能跨领域查询，同时保证数据一致性和最小化冗余

PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- =============================================================================
-- 核心表定义 (所有数据类型通用)
-- =============================================================================

-- 🎯 1. 核心文件注册表 - 所有文件的唯一入口点
CREATE TABLE IF NOT EXISTS file_info (
    file_id TEXT PRIMARY KEY,                       -- 文件唯一标识 (e.g., UUID)
    file_path TEXT NOT NULL UNIQUE,                 -- 完整文件路径
    file_path_hash TEXT,                            -- 路径hash，加速查找
    logical_name TEXT,                              -- 逻辑名称
    file_size INTEGER,
    last_modified INTEGER,                          -- Unix时间戳
    file_format TEXT,                               -- 文件格式 (NetCDF, GeoTIFF, etc.)
    format_variant TEXT,                            -- 格式变体 (classic, 64bit, netcdf4等)
    format_specific_attributes TEXT,                -- (JSON) NetCDF全局属性, GeoTIFF元数据等
    
    -- 核心架构变更 v4.0: 逻辑分层分类
    primary_category TEXT,                          -- 第一层分类: 文件归属的主要逻辑库 (e.g., 'OCEAN_ENVIRONMENT')
    
    quality_score REAL DEFAULT 1.0,                 -- 数据质量评分
    completeness_score REAL DEFAULT 1.0,            -- 完整性评分
    variable_summary TEXT,                          -- 预聚合变量信息 (JSON)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 🎯 2. 文件-数据类型关联表 - 实现多类型查询的关键
CREATE TABLE IF NOT EXISTS file_data_types (
    file_id TEXT NOT NULL,
    data_type TEXT NOT NULL,                        -- 第二层分类: 详细数据类型 (e.g., 'TEMPERATURE', 'BATHYMETRY')
    confidence_score REAL DEFAULT 1.0,              -- 该分类的置信度
    PRIMARY KEY (file_id, data_type),
    FOREIGN KEY (file_id) REFERENCES file_info(file_id) ON DELETE CASCADE
);

-- 🎯 3. 空间覆盖表 - 优化空间查询
CREATE TABLE IF NOT EXISTS spatial_coverage (
    file_id TEXT PRIMARY KEY,
    min_longitude REAL NOT NULL,
    max_longitude REAL NOT NULL,
    min_latitude REAL NOT NULL,
    max_latitude REAL NOT NULL,
    min_depth REAL,
    max_depth REAL,
    spatial_resolution_x REAL,
    spatial_resolution_y REAL,
    crs_wkt TEXT,
    crs_epsg_code INTEGER,
    geohash_6 TEXT,                                 -- 精度6的Geohash
    geohash_8 TEXT,                                 -- 精度8的Geohash
    FOREIGN KEY (file_id) REFERENCES file_info(file_id) ON DELETE CASCADE
);

-- 🎯 4. 时间覆盖表 - 优化时间查询
CREATE TABLE IF NOT EXISTS temporal_coverage (
    file_id TEXT PRIMARY KEY,
    start_time TEXT NOT NULL,                       -- ISO 8601格式
    end_time TEXT NOT NULL,                         -- ISO 8601格式
    start_timestamp INTEGER,                        -- Unix时间戳, 优化范围查询
    end_timestamp INTEGER,                          -- Unix时间戳, 优化范围查询
    time_resolution_seconds REAL,                   -- 时间分辨率(秒)
    time_resolution_category TEXT,                  -- 中文时间分辨率分类(年、月、日、时、分、秒)
    time_calendar TEXT,                             -- 日历类型
    FOREIGN KEY (file_id) REFERENCES file_info(file_id) ON DELETE CASCADE
);

-- 🎯 5. 变量目录表 - 文件内变量的快速索引
CREATE TABLE IF NOT EXISTS variable_info (
    variable_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL,
    variable_name TEXT NOT NULL,
    standard_name TEXT,
    long_name TEXT,
    units TEXT,
    data_type TEXT,                                 -- 变量的原始数据类型 (e.g., float, int)
    dimensions TEXT,                                -- 维度列表(JSON)
    variable_category TEXT,                         -- 变量的分类 (e.g., 'temperature', 'salinity')
    is_coordinate INTEGER DEFAULT 0,
    FOREIGN KEY (file_id) REFERENCES file_info(file_id) ON DELETE CASCADE
);

-- 🎯 6. 变量属性表 - 存储所有变量的详细属性
CREATE TABLE IF NOT EXISTS variable_attributes (
    attr_id INTEGER PRIMARY KEY AUTOINCREMENT,
    variable_id INTEGER NOT NULL,
    attribute_name TEXT NOT NULL,
    attribute_value TEXT,
    FOREIGN KEY (variable_id) REFERENCES variable_info(variable_id) ON DELETE CASCADE
);


-- =============================================================================
-- 专用数据表 (原先分散在不同数据库中的专用表)
-- =============================================================================

-- ✅ 地形底质专用表
CREATE TABLE IF NOT EXISTS topography_variables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL,
    variable_id INTEGER NOT NULL,
    topo_parameter TEXT,             -- 地形参数类型 (e.g., 'elevation', 'slope')
    vertical_datum TEXT,             -- 垂直基准 (e.g., 'WGS84', 'MSL')
    FOREIGN KEY (file_id) REFERENCES file_info(file_id) ON DELETE CASCADE,
    FOREIGN KEY (variable_id) REFERENCES variable_info(variable_id) ON DELETE CASCADE
);

-- ✅ 声纳传播专用表
CREATE TABLE IF NOT EXISTS sonar_variables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL,
    variable_id INTEGER NOT NULL,
    platform_id TEXT,
    sensor_id TEXT,
    working_mode_id TEXT,
    frequency_hz REAL,
    FOREIGN KEY (file_id) REFERENCES file_info(file_id) ON DELETE CASCADE,
    FOREIGN KEY (variable_id) REFERENCES variable_info(variable_id) ON DELETE CASCADE
);

-- (此处可以继续添加其他专用表，如 boundary_variables, tactical_variables 等)

-- =============================================================================
-- 索引和视图 (用于性能优化)
-- =============================================================================

-- 🚀 核心索引
CREATE INDEX IF NOT EXISTS idx_file_path_hash ON file_info(file_path_hash);
CREATE INDEX IF NOT EXISTS idx_file_primary_category ON file_info(primary_category);
CREATE INDEX IF NOT EXISTS idx_file_data_types_file_id ON file_data_types(file_id);
CREATE INDEX IF NOT EXISTS idx_file_data_types_type ON file_data_types(data_type);
CREATE INDEX IF NOT EXISTS idx_spatial_geohash_6 ON spatial_coverage(geohash_6);
CREATE INDEX IF NOT EXISTS idx_temporal_range ON temporal_coverage(start_timestamp, end_timestamp);
CREATE INDEX IF NOT EXISTS idx_variable_name ON variable_info(variable_name);
CREATE INDEX IF NOT EXISTS idx_variable_category ON variable_info(variable_category);

-- ✅ 专用索引
CREATE INDEX IF NOT EXISTS idx_sonar_platform_sensor ON sonar_variables(platform_id, sensor_id);


-- 🚀 统一查询视图 - 简化常用查询
CREATE VIEW IF NOT EXISTS unified_metadata_view AS
SELECT 
    fi.file_id,
    fi.file_path,
    fi.logical_name,
    fi.primary_category,
    fi.quality_score,
    -- 使用 GROUP_CONCAT 将一个文件的所有数据类型聚合为逗号分隔的字符串
    (SELECT GROUP_CONCAT(fdt.data_type) FROM file_data_types fdt WHERE fdt.file_id = fi.file_id) as all_data_types,
    sc.min_longitude, sc.max_longitude, sc.min_latitude, sc.max_latitude,
    tc.start_time, tc.end_time
FROM file_info fi
LEFT JOIN spatial_coverage sc ON fi.file_id = sc.file_id
LEFT JOIN temporal_coverage tc ON fi.file_id = tc.file_id;

-- OSCEAN 统一元数据数据库 Schema
-- 版本: 1.0

-- 文件元数据主表
CREATE TABLE IF NOT EXISTS file_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_hash TEXT,
    format_name TEXT,
    format_version TEXT,
    last_modified TEXT NOT NULL,
    created_at TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

-- 空间范围信息表
CREATE TABLE IF NOT EXISTS spatial_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    min_x REAL,
    min_y REAL,
    max_x REAL,
    max_y REAL,
    crs_id TEXT,
    FOREIGN KEY (file_id) REFERENCES file_metadata (id)
);

-- 时间范围信息表
CREATE TABLE IF NOT EXISTS temporal_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    start_time TEXT,
    end_time TEXT,
    time_resolution TEXT,
    FOREIGN KEY (file_id) REFERENCES file_metadata (id)
);

-- 数据变量信息表
CREATE TABLE IF NOT EXISTS variables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    data_type TEXT,
    units TEXT,
    dimensions TEXT,
    attributes_json TEXT,
    FOREIGN KEY (file_id) REFERENCES file_metadata (id)
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_file_metadata_path ON file_metadata (file_path);
CREATE INDEX IF NOT EXISTS idx_spatial_info_file_id ON spatial_info (file_id);
CREATE INDEX IF NOT EXISTS idx_temporal_info_file_id ON temporal_info (file_id);
CREATE INDEX IF NOT EXISTS idx_variables_file_id ON variables (file_id);
CREATE INDEX IF NOT EXISTS idx_variables_name ON variables (name); 