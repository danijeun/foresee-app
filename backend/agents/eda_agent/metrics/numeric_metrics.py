"""
Numeric metrics calculation for numeric columns
"""
from typing import Dict, Any, Optional
from ..database.connection import SnowflakeConnection
from ..utils.logger import get_logger
from ..utils.helpers import safe_divide, format_number
from ..config import EDAConfig

logger = get_logger(__name__)


class NumericMetricsCalculator:
    """
    Calculates comprehensive statistics for numeric columns
    """
    
    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize numeric metrics calculator
        
        Args:
            connection: SnowflakeConnection instance
        """
        self.conn = connection
        self.config = EDAConfig
    
    def calculate(
        self,
        table_name: str,
        column_name: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate all numeric metrics for a column
        
        Args:
            table_name: Fully qualified table name
            column_name: Column name
            total_rows: Total number of rows
            
        Returns:
            dict: Numeric metrics
        """
        try:
            logger.debug(f"Calculating numeric metrics for {column_name}")
            
            metrics = {}
            
            # Basic statistics
            basic_stats = self._calculate_basic_stats(table_name, column_name)
            metrics.update(basic_stats)
            
            # Percentiles
            percentile_stats = self._calculate_percentiles(table_name, column_name)
            metrics.update(percentile_stats)
            
            # Outlier detection (calculate IQR internally but don't store it)
            if metrics.get('q1') is not None and metrics.get('q3') is not None:
                iqr = metrics['q3'] - metrics['q1']
                outlier_stats = self._calculate_outliers(
                    table_name, 
                    column_name, 
                    metrics.get('q1'), 
                    metrics.get('q3'),
                    iqr,
                    total_rows
                )
                metrics.update(outlier_stats)
            
            # Skewness (approximation using Pearson's formula)
            if all(k in metrics for k in ['mean_value', 'median_value', 'std_dev']):
                if metrics['std_dev'] and metrics['std_dev'] > 0:
                    skewness = safe_divide(
                        3 * (metrics['mean_value'] - metrics['median_value']),
                        metrics['std_dev'],
                        0.0
                    )
                    metrics['skewness'] = format_number(skewness)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating numeric metrics for {column_name}: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_basic_stats(
        self,
        table_name: str,
        column_name: str
    ) -> Dict[str, Any]:
        """
        Calculate basic statistical measures
        
        Args:
            table_name: Table name
            column_name: Column name
            
        Returns:
            dict: Basic statistics
        """
        try:
            query = f"""
                SELECT 
                    AVG("{column_name}") as mean_val,
                    STDDEV("{column_name}") as std_val,
                    VARIANCE("{column_name}") as var_val,
                    MIN("{column_name}") as min_val,
                    MAX("{column_name}") as max_val
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            if not result:
                return {}
            
            mean_val = format_number(result[0])
            std_val = format_number(result[1])
            var_val = format_number(result[2])
            min_val = result[3]
            max_val = result[4]
            
            stats = {
                'mean_value': mean_val,
                'std_dev': std_val,
                'min_value': min_val,
                'max_value': max_val
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating basic stats: {str(e)}")
            return {}
    
    def _calculate_percentiles(
        self,
        table_name: str,
        column_name: str
    ) -> Dict[str, Any]:
        """
        Calculate percentiles including median, Q1, Q3
        
        Args:
            table_name: Table name
            column_name: Column name
            
        Returns:
            dict: Percentile statistics
        """
        try:
            query = f"""
                SELECT 
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{column_name}") as p25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY "{column_name}") as p50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{column_name}") as p75
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            if not result:
                return {}
            
            return {
                'q1': format_number(result[0]),
                'median_value': format_number(result[1]),
                'q3': format_number(result[2])
            }
            
        except Exception as e:
            logger.error(f"Error calculating percentiles: {str(e)}")
            return {}
    
    def _calculate_outliers(
        self,
        table_name: str,
        column_name: str,
        q1: Optional[float],
        q3: Optional[float],
        iqr: Optional[float],
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Calculate outlier counts using IQR method
        
        Args:
            table_name: Table name
            column_name: Column name
            q1: First quartile
            q3: Third quartile
            iqr: Interquartile range
            total_rows: Total rows
            
        Returns:
            dict: Outlier statistics
        """
        try:
            if q1 is None or q3 is None or iqr is None:
                return {
                    'outlier_count': None,
                    'outlier_percentage': None
                }
            
            # Convert to float to avoid Decimal multiplication issues
            lower_bound = float(q1) - (self.config.IQR_MULTIPLIER * float(iqr))
            upper_bound = float(q3) + (self.config.IQR_MULTIPLIER * float(iqr))
            
            query = f"""
                SELECT COUNT(*) as outlier_count
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                  AND ("{column_name}" < {lower_bound} OR "{column_name}" > {upper_bound})
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            outlier_count = result[0] if result else 0
            
            return {
                'outlier_percentage': format_number(safe_divide(outlier_count * 100, total_rows, 0.0))
            }
            
        except Exception as e:
            logger.error(f"Error calculating outliers: {str(e)}")
            return {
                'outlier_percentage': None
            }
    
    def calculate_mode(self, table_name: str, column_name: str) -> Optional[Any]:
        """
        Calculate mode (most common value) for numeric column
        
        Args:
            table_name: Table name
            column_name: Column name
            
        Returns:
            Mode value or None
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
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error calculating mode: {str(e)}")
            return None
