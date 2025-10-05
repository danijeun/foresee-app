"""
Target variable analysis metrics
"""
from typing import Dict, Any, Optional
from ..database.connection import SnowflakeConnection
from ..utils.logger import get_logger
from ..utils.helpers import safe_divide, format_number
from ..config import EDAConfig

logger = get_logger(__name__)


class TargetMetricsCalculator:
    """
    Calculates target variable analysis metrics for supervised EDA
    """
    
    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize target metrics calculator
        
        Args:
            connection: SnowflakeConnection instance
        """
        self.conn = connection
        self.config = EDAConfig
    
    def analyze_target(
        self,
        table_name: str,
        target_column: str,
        data_type: str,
        unique_count: int,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Analyze target variable and determine if classification or regression
        
        Args:
            table_name: Table name
            target_column: Target column name
            data_type: Data type of target
            unique_count: Number of unique values
            total_rows: Total rows
            
        Returns:
            dict: Target analysis results
        """
        try:
            logger.info(f"Analyzing target variable: {target_column}")
            
            # Determine problem type
            is_numeric = self.config.is_numeric_type(data_type)
            
            if is_numeric and unique_count > 20:
                # Regression problem
                problem_type = 'regression'
                target_stats = self._analyze_regression_target(table_name, target_column, total_rows)
            else:
                # Classification problem
                if unique_count == 2:
                    problem_type = 'binary_classification'
                else:
                    problem_type = 'multiclass_classification'
                target_stats = self._analyze_classification_target(table_name, target_column, total_rows)
            
            return {
                'problem_type': problem_type,
                'target_stats': target_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing target: {str(e)}")
            return {
                'problem_type': 'unknown',
                'target_stats': {},
                'error': str(e)
            }
    
    def _analyze_classification_target(
        self,
        table_name: str,
        target_column: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Analyze classification target variable
        
        Args:
            table_name: Table name
            target_column: Target column name
            total_rows: Total rows
            
        Returns:
            dict: Classification target statistics
        """
        try:
            # Get class distribution
            query = f"""
                SELECT 
                    "{target_column}",
                    COUNT(*) as count,
                    COUNT(*) * 100.0 / {total_rows} as percentage
                FROM {table_name}
                WHERE "{target_column}" IS NOT NULL
                GROUP BY "{target_column}"
                ORDER BY count DESC
            """
            
            self.conn.execute(query)
            results = self.conn.fetch_all()
            
            class_distribution = []
            for row in results:
                class_distribution.append({
                    'class': str(row[0]),
                    'count': int(row[1]),
                    'percentage': format_number(row[2])
                })
            
            # Calculate balance metrics
            num_classes = len(class_distribution)
            if num_classes > 0:
                counts = [c['count'] for c in class_distribution]
                majority_class_pct = max(counts) / total_rows * 100
                minority_class_pct = min(counts) / total_rows * 100
                balance_ratio = safe_divide(min(counts), max(counts), 0.0)
            else:
                majority_class_pct = None
                minority_class_pct = None
                balance_ratio = None
            
            return {
                'num_classes': num_classes,
                'class_distribution': class_distribution,
                'majority_class_percentage': format_number(majority_class_pct),
                'minority_class_percentage': format_number(minority_class_pct),
                'balance_ratio': format_number(balance_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing classification target: {str(e)}")
            return {}
    
    def _analyze_regression_target(
        self,
        table_name: str,
        target_column: str,
        total_rows: int
    ) -> Dict[str, Any]:
        """
        Analyze regression target variable
        
        Args:
            table_name: Table name
            target_column: Target column name
            total_rows: Total rows
            
        Returns:
            dict: Regression target statistics
        """
        try:
            # Use numeric metrics
            query = f"""
                SELECT 
                    AVG("{target_column}") as mean_val,
                    STDDEV("{target_column}") as std_val,
                    MIN("{target_column}") as min_val,
                    MAX("{target_column}") as max_val,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{target_column}") as q1,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY "{target_column}") as median,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{target_column}") as q3
                FROM {table_name}
                WHERE "{target_column}" IS NOT NULL
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            if not result:
                return {}
            
            mean_val = format_number(result[0])
            std_val = format_number(result[1])
            median_val = format_number(result[5])
            
            # Calculate skewness approximation
            skewness = None
            if std_val and std_val > 0 and mean_val is not None and median_val is not None:
                skewness = format_number(safe_divide(3 * (mean_val - median_val), std_val, 0.0))
            
            return {
                'mean': mean_val,
                'median': median_val,
                'std_dev': std_val,
                'min': format_number(result[2]),
                'max': format_number(result[3]),
                'q1': format_number(result[4]),
                'q3': format_number(result[6]),
                'skewness': skewness,
                'distribution_shape': self._determine_distribution_shape(skewness)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regression target: {str(e)}")
            return {}
    
    def _determine_distribution_shape(self, skewness: Optional[float]) -> str:
        """
        Determine distribution shape based on skewness
        
        Args:
            skewness: Skewness value
            
        Returns:
            str: Distribution shape description
        """
        if skewness is None:
            return 'unknown'
        
        if abs(skewness) < 0.5:
            return 'approximately_normal'
        elif skewness > 0:
            return 'right_skewed' if skewness > 1 else 'moderately_right_skewed'
        else:
            return 'left_skewed' if skewness < -1 else 'moderately_left_skewed'
