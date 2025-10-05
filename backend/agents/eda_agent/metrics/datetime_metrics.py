"""
Datetime metrics calculation for date/time columns
"""
from typing import Dict, Any, Optional
from ..database.connection import SnowflakeConnection
from ..utils.logger import get_logger
from ..utils.helpers import format_number

logger = get_logger(__name__)


class DatetimeMetricsCalculator:
    """
    Calculates statistics for datetime columns
    """
    
    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize datetime metrics calculator
        
        Args:
            connection: SnowflakeConnection instance
        """
        self.conn = connection
    
    def calculate(
        self,
        table_name: str,
        column_name: str
    ) -> Dict[str, Any]:
        """
        Calculate all datetime metrics for a column
        
        Args:
            table_name: Fully qualified table name
            column_name: Column name
            
        Returns:
            dict: Datetime metrics (currently empty as all datetime metrics have been removed)
        """
        try:
            logger.debug(f"Datetime metrics calculation skipped for {column_name} (no datetime metrics configured)")
            return {}
            
        except Exception as e:
            logger.error(f"Error in datetime metrics calculation for {column_name}: {str(e)}")
            return {}
    
    def _calculate_date_range(
        self,
        table_name: str,
        column_name: str
    ) -> Dict[str, Any]:
        """
        Calculate min date, max date, and range
        
        Args:
            table_name: Table name
            column_name: Column name
            
        Returns:
            dict: Date range statistics
        """
        try:
            query = f"""
                SELECT 
                    MIN("{column_name}") as min_date,
                    MAX("{column_name}") as max_date,
                    DATEDIFF(day, MIN("{column_name}"), MAX("{column_name}")) as range_days
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            if not result:
                return {
                    'min_date': None,
                    'max_date': None,
                    'date_range_days': None
                }
            
            return {
                'min_date': result[0],
                'max_date': result[1],
                'date_range_days': result[2]
            }
            
        except Exception as e:
            logger.error(f"Error calculating date range: {str(e)}")
            return {
                'min_date': None,
                'max_date': None,
                'date_range_days': None
            }
    
    def _calculate_temporal_distribution(
        self,
        table_name: str,
        column_name: str
    ) -> Dict[str, Any]:
        """
        Calculate temporal distribution (by year, month, day of week)
        
        Args:
            table_name: Table name
            column_name: Column name
            
        Returns:
            dict: Temporal distribution
        """
        try:
            # Get top 5 years
            year_query = f"""
                SELECT YEAR("{column_name}") as year, COUNT(*) as count
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                GROUP BY YEAR("{column_name}")
                ORDER BY count DESC
                LIMIT 5
            """
            
            self.conn.execute(year_query)
            year_results = self.conn.fetch_all()
            years = [{'year': r[0], 'count': r[1]} for r in year_results]
            
            # Get top 5 months
            month_query = f"""
                SELECT MONTH("{column_name}") as month, COUNT(*) as count
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                GROUP BY MONTH("{column_name}")
                ORDER BY count DESC
                LIMIT 5
            """
            
            self.conn.execute(month_query)
            month_results = self.conn.fetch_all()
            months = [{'month': r[0], 'count': r[1]} for r in month_results]
            
            # Get day of week distribution
            dow_query = f"""
                SELECT DAYOFWEEK("{column_name}") as dow, COUNT(*) as count
                FROM {table_name}
                WHERE "{column_name}" IS NOT NULL
                GROUP BY DAYOFWEEK("{column_name}")
                ORDER BY dow
            """
            
            self.conn.execute(dow_query)
            dow_results = self.conn.fetch_all()
            day_of_week = [{'day': r[0], 'count': r[1]} for r in dow_results]
            
            return {
                'by_year': years,
                'by_month': months,
                'by_day_of_week': day_of_week
            }
            
        except Exception as e:
            logger.error(f"Error calculating temporal distribution: {str(e)}")
            return {}
    
    def _calculate_largest_gap(
        self,
        table_name: str,
        column_name: str
    ) -> Optional[int]:
        """
        Calculate largest gap between consecutive dates
        
        Args:
            table_name: Table name
            column_name: Column name
            
        Returns:
            int: Largest gap in days
        """
        try:
            # This is an approximation using LAG function
            query = f"""
                WITH date_gaps AS (
                    SELECT 
                        "{column_name}",
                        LAG("{column_name}") OVER (ORDER BY "{column_name}") as prev_date
                    FROM {table_name}
                    WHERE "{column_name}" IS NOT NULL
                )
                SELECT MAX(DATEDIFF(day, prev_date, "{column_name}")) as max_gap
                FROM date_gaps
                WHERE prev_date IS NOT NULL
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            return result[0] if result and result[0] is not None else None
            
        except Exception as e:
            logger.error(f"Error calculating largest gap: {str(e)}")
            return None
