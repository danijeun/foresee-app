"""
Storage module for saving EDA results to Snowflake
"""
from typing import Dict, List, Any, Optional
import json
from ..utils.logger import get_logger
from ..utils.helpers import format_variant_value, convert_to_sql_null
from .connection import SnowflakeConnection

logger = get_logger(__name__)


class ResultsStorage:
    """
    Handles storage of EDA results to Snowflake tables
    """
    
    def __init__(self, connection: SnowflakeConnection, schema_name: Optional[str] = None):
        """
        Initialize results storage
        
        Args:
            connection: SnowflakeConnection instance
            schema_name: Schema where EDA tables are located
        """
        self.conn = connection
        self.schema = schema_name or connection.schema
    
    def save_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Save EDA summary to workflow_eda_summary table
        
        Args:
            summary_data: Dictionary with summary data
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Saving EDA summary for analysis: {summary_data.get('analysis_id')}")
            
            # Build INSERT statement
            columns = [
                'analysis_id', 'table_name', 'schema_name', 'database_name',
                'total_rows', 'total_columns', 'duplicate_rows', 'duplicate_percentage',
                'target_column', 'analysis_type'
            ]
            
            values = [
                f"'{summary_data.get('analysis_id')}'",
                f"'{summary_data.get('table_name')}'",
                f"'{summary_data.get('schema_name', '')}'" if summary_data.get('schema_name') else 'NULL',
                f"'{summary_data.get('database_name', '')}'" if summary_data.get('database_name') else 'NULL',
                str(summary_data.get('total_rows', 0)),
                str(summary_data.get('total_columns', 0)),
                str(summary_data.get('duplicate_rows', 0)),
                str(summary_data.get('duplicate_percentage', 0.0)),
                f"'{summary_data.get('target_column')}'" if summary_data.get('target_column') else 'NULL',
                f"'{summary_data.get('analysis_type', 'unsupervised')}'"
            ]
            
            insert_sql = f"""
                INSERT INTO {self.schema}.workflow_eda_summary 
                ({', '.join(columns)})
                VALUES ({', '.join(values)})
            """
            
            self.conn.execute(insert_sql)
            self.conn.commit()
            
            logger.info("✓ Summary saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save summary: {str(e)}")
            self.conn.rollback()
            raise
    
    def save_column_stats(self, column_stats: List[Dict[str, Any]], batch_size: int = 10) -> bool:
        """
        Save column statistics to workflow_eda_column_stats table
        
        Args:
            column_stats: List of dictionaries with column statistics
            batch_size: Number of rows to insert per batch
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Saving statistics for {len(column_stats)} columns")
            
            # Process in batches
            for i in range(0, len(column_stats), batch_size):
                batch = column_stats[i:i + batch_size]
                self._insert_column_stats_batch(batch)
                logger.info(f"  Progress: {min(i + batch_size, len(column_stats))}/{len(column_stats)} columns")
            
            self.conn.commit()
            logger.info("✓ All column statistics saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save column stats: {str(e)}")
            self.conn.rollback()
            raise
    
    def _insert_column_stats_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Insert a batch of column statistics
        
        Args:
            batch: List of column stat dictionaries
        """
        # All possible columns in the table
        all_columns = [
            'analysis_id', 'column_name', 'data_type',
            'total_count', 'null_percentage',
            'unique_count', 'cardinality_ratio',
            'mean_value', 'median_value', 'mode_value', 'std_dev',
            'min_value', 'max_value',
            'q1', 'q3',
            'skewness', 'outlier_percentage',
            'entropy'
        ]
        
        # Build multi-row INSERT using SELECT with PARSE_JSON
        select_statements = []
        for stats in batch:
            row_values = []
            for col in all_columns:
                value = stats.get(col)
                
                # Handle VARIANT types (JSON objects/arrays)
                if col in ['mode_value', 'min_value', 'max_value']:
                    if value is None:
                        row_values.append('NULL')
                    elif isinstance(value, (dict, list)):
                        # Complex types: use PARSE_JSON in SELECT
                        json_str = json.dumps(value)
                        escaped = json_str.replace("'", "''")
                        row_values.append(f"PARSE_JSON('{escaped}')")
                    else:
                        # Scalar values for VARIANT: use TO_VARIANT to ensure proper type handling
                        if isinstance(value, str):
                            escaped = value.replace("'", "''")
                            row_values.append(f"TO_VARIANT('{escaped}')")
                        elif isinstance(value, (int, float)):
                            row_values.append(f"TO_VARIANT({value})")
                        else:
                            row_values.append(convert_to_sql_null(value))
                else:
                    row_values.append(convert_to_sql_null(value))
            
            select_statements.append(f"SELECT {', '.join(row_values)}")
        
        # Use UNION ALL to combine multiple rows
        insert_sql = f"""
            INSERT INTO {self.schema}.workflow_eda_column_stats 
            ({', '.join(all_columns)})
            {' UNION ALL '.join(select_statements)}
        """
        
        self.conn.execute(insert_sql)
    
    
    def get_analysis_summary(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve summary for a specific analysis
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            dict or None: Analysis summary data
        """
        try:
            query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    schema_name,
                    database_name,
                    total_rows,
                    total_columns,
                    duplicate_rows,
                    duplicate_percentage,
                    target_column,
                    analysis_type
                FROM {self.schema}.workflow_eda_summary
                WHERE analysis_id = '{analysis_id}'
            """
            
            self.conn.execute(query)
            result = self.conn.fetch_one()
            
            if not result:
                return None
            
            # Return with explicit keys
            return {
                'analysis_id': result[0],
                'table_name': result[1],
                'schema_name': result[2],
                'database_name': result[3],
                'total_rows': result[4],
                'total_columns': result[5],
                'duplicate_rows': result[6],
                'duplicate_percentage': result[7],
                'target_column': result[8],
                'analysis_type': result[9]
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve analysis summary: {str(e)}")
            return None
    
    def get_column_stats(self, analysis_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve column statistics for a specific analysis
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            list: List of column statistics
        """
        try:
            query = f"""
                SELECT *
                FROM {self.schema}.workflow_eda_column_stats
                WHERE analysis_id = '{analysis_id}'
                ORDER BY column_name
            """
            
            self.conn.execute(query)
            results = self.conn.fetch_all()
            
            if not results:
                return []
            
            columns = [desc[0] for desc in self.conn.cursor.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve column stats: {str(e)}")
            return []
