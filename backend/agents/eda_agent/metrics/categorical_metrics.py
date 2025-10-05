"""
Categorical metrics calculation for categorical/text columns
"""
from typing import Dict, Any, List, Optional
from ..database.connection import SnowflakeConnection
from ..utils.logger import get_logger
from ..utils.helpers import safe_divide, format_number, create_value_frequency_dict
from ..config import EDAConfig
import math

logger = get_logger(__name__)


class CategoricalMetricsCalculator:
    """
    Calculates statistics for categorical columns
    """
    
    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize categorical metrics calculator
        
        Args:
            connection: SnowflakeConnection instance
        """
        self.conn = connection
        self.config = EDAConfig
    
    def calculate(
        self,
        table_name: str,
        column_name: str,
        total_rows: int,
        unique_count: int
    ) -> Dict[str, Any]:
        """
        Calculate all categorical metrics for a column
        
        Args:
            table_name: Fully qualified table name
            column_name: Column name
            total_rows: Total number of rows
            unique_count: Number of unique values
            
        Returns:
            dict: Categorical metrics
        """
        try:
            logger.debug(f"Calculating categorical metrics for {column_name}")
            
            metrics = {}
            
            # Mode (most frequent value)
            mode_stats = self._calculate_mode(table_name, column_name, total_rows)
            metrics.update(mode_stats)
            
            # Entropy (measure of randomness)
            entropy = self._calculate_entropy(table_name, column_name, total_rows)
            metrics['entropy'] = entropy
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating categorical metrics for {column_name}: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_mode(
        self,
        table_name: str,
        column_name: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate mode (most frequent value) and its statistics
        
        Args:
            table_name: Table name
            column_name: Column name
            total_rows: Total rows
            
        Returns:
            dict: Mode statistics
        """
        try:
            query = f"""
                SELECT "{column_name}", COUNT(*) as freq
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                GROUP BY "{column_name}"
                ORDER BY freq DESC
                LIMIT 1
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            if not result:
                return {
                    'mode_value': None
                }
            
            mode_value = result[0]
            
            return {
                'mode_value': str(mode_value) if mode_value is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating mode: {str(e)}")
            return {
                'mode_value': None
            }
    
    def _get_top_values(
        self,
        table_name: str,
        column_name: str,
        total_rows: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Get top N most frequent values
        
        Args:
            table_name: Table name
            column_name: Column name
            total_rows: Total rows
            limit: Number of top values to retrieve
            
        Returns:
            list: Top values with counts and percentages
        """
        try:
            query = f"""
                SELECT "{column_name}", COUNT(*) as freq
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                GROUP BY "{column_name}"
                ORDER BY freq DESC
                LIMIT {limit}
            """
            
            self.conn.execute(query)
            results = self.conn.fetch_all()
            
            return create_value_frequency_dict(results, total_rows)
            
        except Exception as e:
            logger.error(f"Error getting top values: {str(e)}")
            return []
    
    def _get_bottom_values(
        self,
        table_name: str,
        column_name: str,
        total_rows: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Get bottom N least frequent values
        
        Args:
            table_name: Table name
            column_name: Column name
            total_rows: Total rows
            limit: Number of bottom values to retrieve
            
        Returns:
            list: Bottom values with counts and percentages
        """
        try:
            query = f"""
                SELECT "{column_name}", COUNT(*) as freq
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                GROUP BY "{column_name}"
                ORDER BY freq ASC
                LIMIT {limit}
            """
            
            self.conn.execute(query)
            results = self.conn.fetch_all()
            
            return create_value_frequency_dict(results, total_rows)
            
        except Exception as e:
            logger.error(f"Error getting bottom values: {str(e)}")
            return []
    
    def _calculate_entropy(
        self,
        table_name: str,
        column_name: str,
        total_rows: int
    ) -> Optional[float]:
        """
        Calculate Shannon entropy for the column
        
        Args:
            table_name: Table name
            column_name: Column name
            total_rows: Total rows
            
        Returns:
            float: Entropy value
        """
        try:
            query = f"""
                SELECT COUNT(*) as freq
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                GROUP BY "{column_name}"
            """
            
            self.conn.execute(query)
            results = self.conn.fetch_all()
            
            if not results:
                return None
            
            # Calculate entropy
            entropy = 0.0
            for (freq,) in results:
                if freq > 0:
                    probability = freq / total_rows
                    entropy -= probability * math.log2(probability)
            
            return format_number(entropy)
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            return None
    
    def _calculate_concentration_ratio(
        self,
        table_name: str,
        column_name: str,
        total_rows: int
    ) -> Optional[float]:
        """
        Calculate concentration ratio (percentage of data in top 3 categories)
        
        Args:
            table_name: Table name
            column_name: Column name
            total_rows: Total rows
            
        Returns:
            float: Concentration ratio percentage
        """
        try:
            query = f"""
                SELECT SUM(freq) as total_top3
                FROM (
                    SELECT COUNT(*) as freq
                    FROM {table_name}
                    WHERE "{column_name}" IS NOT NULL
                    GROUP BY "{column_name}"
                    ORDER BY freq DESC
                    LIMIT 3
                ) t
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            if not result or result[0] is None:
                return None
            
            top3_count = result[0]
            concentration = safe_divide(top3_count * 100, total_rows, 0.0)
            
            return format_number(concentration)
            
        except Exception as e:
            logger.error(f"Error calculating concentration ratio: {str(e)}")
            return None
