import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Conexión
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")

    # Warehouses
    INGESTION_WAREHOUSE = os.getenv("INGESTION_WAREHOUSE")
    
    # Database and Schema
    SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
    SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")