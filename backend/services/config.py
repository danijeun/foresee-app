import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Conexión
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")

    # Warehouses
    INTEGRATION_WAREHOUSE = os.getenv("INTEGRATION_WAREHOUSE")