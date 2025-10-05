"""
Configuration and constants for EDA Agent
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class EDAConfig:
    """Configuration settings for EDA Agent"""
    
    # Snowflake connection defaults
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_WAREHOUSE = os.getenv("INGESTION_WAREHOUSE", "COMPUTE_WH")
    SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
    SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
    
    # Default workflow schema for storing results
    WORKFLOW_SCHEMA = "WORKFLOW_EDA"
    
    # Analysis settings
    DEFAULT_SAMPLE_SIZE = 10000
    MAX_CATEGORICAL_CARDINALITY = 100  # Max unique values to treat as categorical
    TOP_VALUES_LIMIT = 10  # Number of top values to store
    BOTTOM_VALUES_LIMIT = 5  # Number of bottom values to store
    
    # Percentiles to compute for numeric columns
    PERCENTILES = [5, 10, 25, 50, 75, 90, 95, 99]
    
    # Outlier detection
    IQR_MULTIPLIER = 1.5  # Standard IQR method: Q1 - 1.5*IQR, Q3 + 1.5*IQR
    
    # Performance settings
    BATCH_SIZE = 50  # Number of columns to process in one batch
    QUERY_TIMEOUT = 300  # Query timeout in seconds
    
    # Data type mappings
    NUMERIC_TYPES = ['NUMBER', 'DECIMAL', 'NUMERIC', 'INT', 'INTEGER', 'BIGINT', 
                     'SMALLINT', 'TINYINT', 'BYTEINT', 'FLOAT', 'DOUBLE', 'REAL']
    
    CATEGORICAL_TYPES = ['VARCHAR', 'CHAR', 'STRING', 'TEXT', 'BOOLEAN']
    
    DATETIME_TYPES = ['DATE', 'DATETIME', 'TIME', 'TIMESTAMP', 'TIMESTAMP_LTZ', 
                      'TIMESTAMP_NTZ', 'TIMESTAMP_TZ']
    
    TEXT_TYPES = ['VARCHAR', 'CHAR', 'STRING', 'TEXT']
    
    @classmethod
    def get_connection_params(cls, **overrides: Any) -> Dict[str, Any]:
        """
        Get connection parameters with optional overrides
        
        Args:
            **overrides: Override default connection parameters
            
        Returns:
            dict: Connection parameters for Snowflake
        """
        params = {
            'account': cls.SNOWFLAKE_ACCOUNT,
            'user': cls.SNOWFLAKE_USER,
            'password': cls.SNOWFLAKE_PASSWORD,
            'warehouse': cls.SNOWFLAKE_WAREHOUSE,
            'database': cls.SNOWFLAKE_DATABASE,
            'schema': cls.SNOWFLAKE_SCHEMA
        }
        params.update(overrides)
        return params
    
    @classmethod
    def is_numeric_type(cls, data_type: str) -> bool:
        """Check if data type is numeric"""
        return any(nt in data_type.upper() for nt in cls.NUMERIC_TYPES)
    
    @classmethod
    def is_categorical_type(cls, data_type: str) -> bool:
        """Check if data type is categorical"""
        return any(ct in data_type.upper() for ct in cls.CATEGORICAL_TYPES)
    
    @classmethod
    def is_datetime_type(cls, data_type: str) -> bool:
        """Check if data type is datetime"""
        return any(dt in data_type.upper() for dt in cls.DATETIME_TYPES)
    
    @classmethod
    def is_text_type(cls, data_type: str) -> bool:
        """Check if data type is text"""
        return any(tt in data_type.upper() for tt in cls.TEXT_TYPES)


# Table creation SQL
EDA_SUMMARY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {schema}.workflow_eda_summary (
    analysis_id VARCHAR(100) PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    schema_name VARCHAR(255),
    database_name VARCHAR(255),
    total_rows BIGINT,
    total_columns INT,
    duplicate_rows BIGINT,
    duplicate_percentage FLOAT,
    target_column VARCHAR(255),
    analysis_type VARCHAR(50)
)
"""

EDA_COLUMN_STATS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {schema}.workflow_eda_column_stats (
    analysis_id VARCHAR(100) NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    data_type VARCHAR(50),
    
    -- Basic metrics (all columns)
    total_count BIGINT,
    null_percentage FLOAT,
    unique_count BIGINT,
    cardinality_ratio FLOAT,
    
    -- Numeric metrics
    mean_value FLOAT,
    median_value FLOAT,
    mode_value VARIANT,
    std_dev FLOAT,
    min_value VARIANT,
    max_value VARIANT,
    q1 FLOAT,
    q3 FLOAT,
    skewness FLOAT,
    outlier_percentage FLOAT,
    
    -- Categorical metrics
    entropy FLOAT,
    
    PRIMARY KEY (analysis_id, column_name)
)
"""
