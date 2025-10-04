import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SNOWFLAKE_ACCOUNT = os.getenv("IAKSMIY-UA74892")
    SNOWFLAKE_USER = os.getenv("DANIJEUN")
    SNOWFLAKE_PASSWORD = os.getenv("EXg86QQ6FCxBqz4")