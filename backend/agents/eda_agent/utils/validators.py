"""
Input validation utilities for EDA Agent
"""
from typing import Optional, Tuple
import snowflake.connector
from .logger import get_logger

logger = get_logger(__name__)


def validate_table_exists(
    cursor: snowflake.connector.cursor.SnowflakeCursor,
    table_name: str,
    schema_name: Optional[str] = None,
    database_name: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a table exists in Snowflake
    
    Args:
        cursor: Snowflake cursor
        table_name: Name of the table
        schema_name: Schema name (optional)
        database_name: Database name (optional)
        
    Returns:
        tuple: (exists: bool, error_message: Optional[str])
    """
    try:
        fully_qualified = table_name
        if schema_name:
            fully_qualified = f"{schema_name}.{table_name}"
        if database_name:
            fully_qualified = f"{database_name}.{fully_qualified}"
        
        query = f"SELECT 1 FROM {fully_qualified} LIMIT 1"
        cursor.execute(query)
        logger.info(f"✓ Table {fully_qualified} exists and is accessible")
        return True, None
        
    except snowflake.connector.errors.ProgrammingError as e:
        error_msg = f"Table validation failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during table validation: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


