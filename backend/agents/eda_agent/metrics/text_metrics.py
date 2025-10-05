"""
Text metrics calculation for text/string columns
"""
from typing import Dict, Any, Optional
from ..database.connection import SnowflakeConnection
from ..utils.logger import get_logger
from ..utils.helpers import safe_divide, format_number

logger = get_logger(__name__)


class TextMetricsCalculator:
    """
    Calculates statistics for text/string columns
    """
    
    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize text metrics calculator
        
        Args:
            connection: SnowflakeConnection instance
        """
        self.conn = connection
    
    def calculate(
        self,
        table_name: str,
        column_name: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate all text metrics for a column
        
        Args:
            table_name: Fully qualified table name
            column_name: Column name
            total_rows: Total number of rows
            
        Returns:
            dict: Text metrics (currently empty as all text metrics have been removed)
        """
        try:
            logger.debug(f"Text metrics calculation skipped for {column_name} (no text metrics configured)")
            return {}
            
        except Exception as e:
            logger.error(f"Error in text metrics calculation for {column_name}: {str(e)}")
            return {}
    
    def _calculate_length_stats(
        self,
        table_name: str,
        column_name: str
    ) -> Dict[str, Any]:
        """
        Calculate string length statistics
        
        Args:
            table_name: Table name
            column_name: Column name
            
        Returns:
            dict: Length statistics
        """
        try:
            query = f"""
                SELECT 
                    AVG(LENGTH("{column_name}")) as avg_len
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            if not result:
                return {
                    'avg_length': None
                }
            
            return {
                'avg_length': format_number(result[0])
            }
            
        except Exception as e:
            logger.error(f"Error calculating length stats: {str(e)}")
            return {
                'avg_length': None
            }
    
    def _calculate_empty_stats(
        self,
        table_name: str,
        column_name: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate empty string statistics
        
        Args:
            table_name: Table name
            column_name: Column name
            total_rows: Total rows
            
        Returns:
            dict: Empty string statistics
        """
        try:
            query = f"""
                SELECT COUNT(*) as empty_count
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                  AND LENGTH(TRIM("{column_name}")) = 0
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            empty_count = result[0] if result else 0
            
            return {
                'empty_string_count': empty_count,
                'empty_string_percentage': format_number(safe_divide(empty_count * 100, total_rows, 0.0))
            }
            
        except Exception as e:
            logger.error(f"Error calculating empty stats: {str(e)}")
            return {
                'empty_string_count': None,
                'empty_string_percentage': None
            }
