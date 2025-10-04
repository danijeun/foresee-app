import snowflake.connector
import pandas as pd
from pathlib import Path
import os
from .config import Config

class SnowflakeCSVUploader:
    """
    Class to upload CSV files of any size to Snowflake
    """
    
    def __init__(self, account=None, user=None, password=None, warehouse=None, database=None, schema=None):
        """
        Initializes Snowflake connection
        
        Args:
            account: Your Snowflake account (e.g., 'xy12345.us-east-1'). If not provided, uses environment variable.
            user: Snowflake username. If not provided, uses environment variable.
            password: Password. If not provided, uses environment variable.
            warehouse: Warehouse to use. If not provided, uses environment variable.
            database: Database. If not provided, uses environment variable.
            schema: Schema. If not provided, uses environment variable.
        """
        # Save configuration
        self.database = database or Config.SNOWFLAKE_DATABASE
        self.schema = schema or Config.SNOWFLAKE_SCHEMA
        
        # Connect to Snowflake (without database/schema initially to avoid errors)
        self.conn = snowflake.connector.connect(
            user=user or Config.SNOWFLAKE_USER,
            password=password or Config.SNOWFLAKE_PASSWORD,
            account=account or Config.SNOWFLAKE_ACCOUNT,
            warehouse=warehouse or Config.INGESTION_WAREHOUSE
        )
        self.cursor = self.conn.cursor()
        
        # Create and use database if specified
        if self.database:
            try:
                self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                self.cursor.execute(f"USE DATABASE {self.database}")
            except Exception as e:
                print(f"⚠️ Warning: Could not create/use database {self.database}: {e}")
        
        # Create and use schema if specified
        if self.schema:
            try:
                self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                self.cursor.execute(f"USE SCHEMA {self.schema}")
            except Exception as e:
                print(f"⚠️ Warning: Could not create/use schema {self.schema}: {e}")
    
    def upload_csv(self, csv_path, table_name, stage_name='csv_stage'):
        """
        Uploads CSV files of any size to Snowflake.
        Uses Snowflake's internal STAGE for maximum efficiency and compatibility.
        
        Args:
            csv_path: Path to the CSV file
            table_name: Name of the table in Snowflake
            stage_name: Name of the temporary stage (default: 'csv_stage')
        
        Returns:
            dict: Information about the upload (file size, rows loaded, etc.)
        """
        # Get file size in MB
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        
        print(f"📁 File: {csv_path}")
        print(f"📊 Size: {file_size_mb:.2f} MB")
        print(f"🚀 Using STAGE method (works for any size)")
        print(f"Loading CSV: {csv_path}")
        
        # 1. Create internal stage if it doesn't exist (with fully qualified name)
        stage_full_name = f"{self.database}.{self.schema}.{stage_name}"
        self.cursor.execute(f"CREATE STAGE IF NOT EXISTS {stage_full_name}")
        print(f"✓ Stage '{stage_name}' ready")
        
        # 2. Upload file to stage
        self.cursor.execute(f"PUT file://{csv_path} @{stage_full_name} AUTO_COMPRESS=TRUE")
        print(f"✓ File uploaded to stage")
        
        # 3. Infer table structure by reading a sample
        df_sample = pd.read_csv(csv_path, nrows=5)
        
        # 4. Create table with appropriate structure (with fully qualified name)
        table_full_name = f"{self.database}.{self.schema}.{table_name}"
        create_table_sql = self._generate_create_table(table_full_name, df_sample)
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_full_name}")
        self.cursor.execute(create_table_sql)
        print(f"✓ Table '{table_name}' created")
        
        # 5. Load data from stage to table
        file_name = Path(csv_path).name
        copy_sql = f"""
        COPY INTO {table_full_name}
        FROM @{stage_full_name}/{file_name}.gz
        FILE_FORMAT = (
            TYPE = 'CSV'
            FIELD_DELIMITER = ','
            SKIP_HEADER = 1
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            TRIM_SPACE = TRUE
            ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
        )
        ON_ERROR = 'CONTINUE'
        """
        self.cursor.execute(copy_sql)
        print(f"✓ Data loaded to table")
        
        # 6. Clean up stage
        self.cursor.execute(f"REMOVE @{stage_full_name}/{file_name}.gz")
        print(f"✓ Stage cleaned")
        
        # 7. Show statistics
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_full_name}")
        row_count = self.cursor.fetchone()[0]
        print(f"✓ Total rows loaded: {row_count:,}")
        
        return {
            'method': 'stage',
            'file_size_mb': round(file_size_mb, 2),
            'rows_loaded': row_count,
            'table_name': table_name
        }
        
    def _generate_create_table(self, table_name, df_sample):
        """
        Generates SQL to create table based on DataFrame
        """
        type_mapping = {
            'int64': 'NUMBER',
            'float64': 'FLOAT',
            'object': 'VARCHAR(16777216)',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP'
        }
        
        columns = []
        for col, dtype in df_sample.dtypes.items():
            snowflake_type = type_mapping.get(str(dtype), 'VARCHAR(16777216)')
            columns.append(f'"{col}" {snowflake_type}')
        
        return f"CREATE TABLE {table_name} ({', '.join(columns)})"
    
    def query(self, sql):
        """
        Executes a query and returns results
        """
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def close(self):
        """
        Closes the connection
        """
        self.cursor.close()
        self.conn.close()


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Connection setup - Automatically uses environment variables
    uploader = SnowflakeCSVUploader()
    
    try:
        # ✨ upload_csv works for files of any size
        result = uploader.upload_csv('my_file.csv', 'my_table')
        print(f"\n✅ Upload completed:")
        print(f"   - Method: {result['method']}")
        print(f"   - Size: {result['file_size_mb']} MB")
        print(f"   - Rows: {result['rows_loaded']:,}")
        
        # Verify data
        print("\nFirst 5 rows:")
        results = uploader.query(f"SELECT * FROM {result['table_name']} LIMIT 5")
        for row in results:
            print(row)
            
    finally:
        uploader.close()