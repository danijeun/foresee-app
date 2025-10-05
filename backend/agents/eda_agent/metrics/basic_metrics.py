"""
Basic metrics calculation for all column types
"""
from typing import Dict, Any, Optional
from ..database.connection import SnowflakeConnection
from ..utils.logger import get_logger
from ..utils.helpers import safe_divide

logger = get_logger(__name__)


class BasicMetricsCalculator:
    """
    Calculates basic statistics applicable to all column types
    """
    
    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize basic metrics calculator
        
        Args:
            connection: SnowflakeConnection instance
        """
        self.conn = connection
    
    def calculate(
        self,
        table_name: str,
        column_name: str,
        data_type: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate basic metrics for a column
        
        Args:
            table_name: Fully qualified table name
            column_name: Column name
            data_type: Column data type
            total_rows: Total number of rows in table
            
        Returns:
            dict: Basic metrics
        """
        try:
            logger.debug(f"Calculating basic metrics for {column_name}")
            
            metrics = {
                'column_name': column_name,
                'data_type': data_type,
                'total_count': total_rows
            }
            
            # Calculate null counts
            null_stats = self._calculate_null_stats(table_name, column_name, total_rows)
            metrics.update(null_stats)
            
            # Calculate unique counts
            unique_stats = self._calculate_unique_stats(table_name, column_name, total_rows)
            metrics.update(unique_stats)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics for {column_name}: {str(e)}")
            return {
                'column_name': column_name,
                'data_type': data_type,
                'total_count': total_rows,
                'error': str(e)
            }
    
    def _calculate_null_stats(
        self,
        table_name: str,
        column_name: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate null-related statistics
        
        Args:
            table_name: Table name
            column_name: Column name
            total_rows: Total rows
            
        Returns:
            dict: Null statistics
        """
        try:
            query = f"""
                SELECT 
                    COUNT("{column_name}") as non_null_count,
                    COUNT(*) - COUNT("{column_name}") as null_count
                FROM {table_name}
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            null_count = result[1] if result else total_rows
            
            return {
                'null_percentage': safe_divide(null_count * 100, total_rows, 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating null stats: {str(e)}")
            return {
                'null_percentage': 100.0
            }
    
    def _calculate_unique_stats(
        self,
        table_name: str,
        column_name: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate unique value statistics
        
        Args:
            table_name: Table name
            column_name: Column name
            total_rows: Total rows
            
        Returns:
            dict: Unique value statistics
        """
        try:
            query = f"""
                SELECT COUNT(DISTINCT "{column_name}") as unique_count
                FROM {table_name}
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            unique_count = result[0] if result else 0
            
            return {
                'unique_count': unique_count,
                'cardinality_ratio': safe_divide(unique_count, total_rows, 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating unique stats: {str(e)}")
            return {
                'unique_count': 0,
                'cardinality_ratio': 0.0
            }
