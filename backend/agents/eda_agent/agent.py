"""
Main Snowflake EDA Agent class
Production-ready exploratory data analysis for Snowflake tables
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import json

from .database.connection import SnowflakeConnection
from .database.schema import SchemaManager
from .database.storage import ResultsStorage
from .metrics import (
    BasicMetricsCalculator,
    NumericMetricsCalculator,
    CategoricalMetricsCalculator,
    DatetimeMetricsCalculator,
    TextMetricsCalculator,
    TargetMetricsCalculator
)
from .utils.logger import setup_logger, get_logger
from .utils.validators import validate_table_exists
from .utils.helpers import generate_analysis_id, merge_dicts
from .config import EDAConfig

logger = get_logger(__name__)


class SnowflakeEDAAgent:
    """
    Production-ready EDA Agent for Snowflake tables
    
    Performs comprehensive exploratory data analysis on any Snowflake table
    and saves all results to workflow schema tables.
    
    Example:
        >>> agent = SnowflakeEDAAgent(
        ...     account='your_account',
        ...     user='your_user',
        ...     password='your_password',
        ...     warehouse='COMPUTE_WH',
        ...     database='ANALYTICS',
        ...     workflow_schema='WORKFLOW_EDA'
        ... )
        >>> 
        >>> # Create EDA tables if they don't exist
        >>> agent.create_eda_tables()
        >>> 
        >>> # Perform unsupervised EDA
        >>> analysis_id = agent.analyze_table(
        ...     table_name='CUSTOMERS',
        ...     schema='PUBLIC'
        ... )
        >>> 
        >>> # Perform supervised EDA with target
        >>> analysis_id = agent.analyze_table(
        ...     table_name='CUSTOMERS',
        ...     schema='PUBLIC',
        ...     target_column='CHURNED'
        ... )
        >>> 
        >>> # Get results
        >>> summary = agent.get_analysis_summary(analysis_id)
        >>> print(summary)
    """
    
    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        workflow_schema: Optional[str] = None,
        log_level: str = 'INFO',
        log_file: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Snowflake EDA Agent
        
        Args:
            account: Snowflake account identifier
            user: Username
            password: Password
            warehouse: Warehouse name
            database: Database name
            schema: Default schema name
            workflow_schema: Schema for storing EDA results
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            log_file: Path to log file (optional)
            **kwargs: Additional connection parameters
        """
        # Setup logging
        import logging
        level = getattr(logging, log_level.upper(), logging.INFO)
        global logger
        logger = setup_logger('eda_agent', level=level, log_file=log_file)
        
        logger.info("=" * 70)
        logger.info("Initializing Snowflake EDA Agent")
        logger.info("=" * 70)
        
        # Store configuration
        self.workflow_schema = workflow_schema or EDAConfig.WORKFLOW_SCHEMA
        
        # Establish connection
        self.conn = SnowflakeConnection(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema,
            **kwargs
        )
        
        # Initialize managers
        self.schema_manager = SchemaManager(self.conn)
        self.storage = ResultsStorage(self.conn, self.workflow_schema)
        
        # Initialize metric calculators
        self.basic_calc = BasicMetricsCalculator(self.conn)
        self.numeric_calc = NumericMetricsCalculator(self.conn)
        self.categorical_calc = CategoricalMetricsCalculator(self.conn)
        self.datetime_calc = DatetimeMetricsCalculator(self.conn)
        self.text_calc = TextMetricsCalculator(self.conn)
        self.target_calc = TargetMetricsCalculator(self.conn)
        
        logger.info("✓ EDA Agent initialized successfully")
    
    def create_eda_tables(self) -> bool:
        """
        Create EDA results tables in the workflow schema
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Creating EDA tables...")
            
            # Create workflow schema if it doesn't exist
            self.schema_manager.create_workflow_schema(self.workflow_schema)
            
            # Create EDA tables
            self.schema_manager.create_eda_tables(self.workflow_schema)
            
            logger.info("✓ EDA tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create EDA tables: {str(e)}")
            raise
    
    def analyze_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        target_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> str:
        """
        Perform comprehensive EDA on a Snowflake table
        
        Args:
            table_name: Name of the table to analyze
            schema: Schema name (uses connection default if not provided)
            database: Database name (uses connection default if not provided)
            target_column: Target variable for supervised EDA (optional)
            exclude_columns: List of columns to exclude from analysis
            
        Returns:
            str: Analysis ID for retrieving results
        """
        analysis_id = generate_analysis_id()
        
        try:
            logger.info("=" * 70)
            logger.info(f"Starting EDA Analysis")
            logger.info(f"Analysis ID: {analysis_id}")
            logger.info(f"Table: {table_name}")
            if target_column:
                logger.info(f"Target Column: {target_column} (Supervised EDA)")
            else:
                logger.info("Mode: Unsupervised EDA")
            logger.info("=" * 70)
            
            # Build fully qualified table name
            schema = schema or self.conn.schema
            database = database or self.conn.database
            qualified_table = self._build_qualified_name(database, schema, table_name)
            
            # Validate table exists
            exists, error_msg = validate_table_exists(
                self.conn.cursor, table_name, schema, database
            )
            if not exists:
                raise ValueError(f"Table validation failed: {error_msg}")
            
            # Get table metadata
            logger.info("Retrieving table metadata...")
            column_info = self.conn.get_column_info(table_name, schema)
            total_rows = self._get_row_count(qualified_table)
            
            logger.info(f"  Total rows: {total_rows:,}")
            logger.info(f"  Total columns: {len(column_info)}")
            
            # Validate target column if provided
            if target_column:
                if target_column not in column_info:
                    raise ValueError(f"Target column '{target_column}' not found in table")
                logger.info(f"  Target column validated: {target_column}")
            
            # Filter out excluded columns
            if exclude_columns:
                column_info = {k: v for k, v in column_info.items() if k not in exclude_columns}
                logger.info(f"  Excluded {len(exclude_columns)} columns")
            
            # Calculate duplicate statistics
            duplicate_stats = self._calculate_duplicates(qualified_table, total_rows)
            
            # Save initial summary
            summary_data = {
                'analysis_id': analysis_id,
                'table_name': table_name,
                'schema_name': schema,
                'database_name': database,
                'total_rows': total_rows,
                'total_columns': len(column_info),
                'duplicate_rows': duplicate_stats['duplicate_rows'],
                'duplicate_percentage': duplicate_stats['duplicate_percentage'],
                'target_column': target_column,
                'analysis_type': 'supervised' if target_column else 'unsupervised'
            }
            
            self.storage.save_summary(summary_data)
            
            # Analyze target variable if provided
            target_info = None
            if target_column:
                logger.info(f"\n📊 Analyzing target variable: {target_column}")
                target_data_type = column_info[target_column]
                target_unique_count = self._get_unique_count(qualified_table, target_column)
                target_info = self.target_calc.analyze_target(
                    qualified_table,
                    target_column,
                    target_data_type,
                    target_unique_count,
                    total_rows
                )
                logger.info(f"  Problem type: {target_info['problem_type']}")
            
            # Analyze each column
            logger.info(f"\n📈 Analyzing {len(column_info)} columns...")
            column_stats_list = []
            
            for idx, (col_name, data_type) in enumerate(column_info.items(), 1):
                try:
                    logger.info(f"  [{idx}/{len(column_info)}] {col_name} ({data_type})")
                    
                    # Calculate column statistics
                    col_stats = self._analyze_column(
                        qualified_table,
                        col_name,
                        data_type,
                        total_rows,
                        target_column,
                        target_info
                    )
                    
                    # Add metadata
                    col_stats['analysis_id'] = analysis_id
                    
                    column_stats_list.append(col_stats)
                    
                except Exception as e:
                    logger.error(f"    ⚠️ Error analyzing column {col_name}: {str(e)}")
                    # Add error entry
                    column_stats_list.append({
                        'analysis_id': analysis_id,
                        'column_name': col_name,
                        'data_type': data_type,
                        'quality_issues': [{'issue': 'analysis_failed', 'error': str(e)}]
                    })
            
            # Save all column statistics
            logger.info("\n💾 Saving analysis results...")
            self.storage.save_column_stats(column_stats_list)
            
            logger.info("=" * 70)
            logger.info(f"✓ EDA Analysis Completed Successfully")
            logger.info(f"  Analysis ID: {analysis_id}")
            logger.info(f"  Columns analyzed: {len(column_stats_list)}")
            logger.info("=" * 70)
            
            return analysis_id
            
        except Exception as e:
            error_msg = str(e)
            
            logger.error("=" * 70)
            logger.error(f"✗ EDA Analysis Failed")
            logger.error(f"  Error: {error_msg}")
            logger.error("=" * 70)
            
            raise
    
    def _analyze_column(
        self,
        table_name: str,
        column_name: str,
        data_type: str,
        total_rows: int,
        target_column: Optional[str],
        target_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a single column with appropriate metrics
        
        Args:
            table_name: Qualified table name
            column_name: Column name
            data_type: Column data type
            total_rows: Total rows in table
            target_column: Target column name (if supervised)
            target_info: Target analysis info
            
        Returns:
            dict: Column statistics
        """
        # Basic metrics (all columns)
        stats = self.basic_calc.calculate(table_name, column_name, data_type, total_rows)
        
        # Type-specific metrics
        if EDAConfig.is_numeric_type(data_type):
            numeric_stats = self.numeric_calc.calculate(table_name, column_name, total_rows)
            stats = merge_dicts(stats, numeric_stats)
            
            # Mode for numeric
            mode_value = self.numeric_calc.calculate_mode(table_name, column_name)
            if mode_value is not None:
                stats['mode_value'] = mode_value
        
        if EDAConfig.is_categorical_type(data_type) or stats.get('unique_count', 0) <= EDAConfig.MAX_CATEGORICAL_CARDINALITY:
            categorical_stats = self.categorical_calc.calculate(
                table_name, column_name, total_rows, stats.get('unique_count', 0)
            )
            stats = merge_dicts(stats, categorical_stats)
        
        if EDAConfig.is_datetime_type(data_type):
            datetime_stats = self.datetime_calc.calculate(table_name, column_name)
            stats = merge_dicts(stats, datetime_stats)
        
        if EDAConfig.is_text_type(data_type):
            text_stats = self.text_calc.calculate(table_name, column_name, total_rows)
            stats = merge_dicts(stats, text_stats)
        
        return stats
    
    def _build_qualified_name(
        self,
        database: Optional[str],
        schema: Optional[str],
        table: str
    ) -> str:
        """Build fully qualified table name"""
        parts = []
        if database:
            parts.append(database)
        if schema:
            parts.append(schema)
        parts.append(table)
        return '.'.join(parts)
    
    def _get_row_count(self, table_name: str) -> int:
        """Get total row count for a table"""
        try:
            self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = self.conn.fetch_one()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting row count: {str(e)}")
            return 0
    
    def _get_unique_count(self, table_name: str, column_name: str) -> int:
        """Get unique value count for a column"""
        try:
            self.conn.execute(f'SELECT COUNT(DISTINCT "{column_name}") FROM {table_name}')
            result = self.conn.fetch_one()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting unique count: {str(e)}")
            return 0
    
    def _calculate_duplicates(self, table_name: str, total_rows: int) -> Dict[str, Any]:
        """Calculate duplicate row statistics"""
        try:
            # Use a different approach: compare total rows with distinct rows
            # Get all column names
            column_info = self.conn.get_column_info(table_name.split('.')[-1], 
                                                    table_name.split('.')[-2] if '.' in table_name else None)
            columns = list(column_info.keys())
            
            if not columns:
                return {'duplicate_rows': 0, 'duplicate_percentage': 0.0}
            
            # Build DISTINCT query with all columns
            columns_quoted = ', '.join([f'"{col}"' for col in columns])
            query = f"""
                WITH distinct_rows AS (
                    SELECT DISTINCT {columns_quoted}
                    FROM {table_name}
                )
                SELECT {total_rows} - COUNT(*) as duplicate_count
                FROM distinct_rows
            """
            self.conn.execute(query)
            result = self.conn.fetch_one()
            duplicate_rows = result[0] if result and result[0] else 0
            
            return {
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': round((duplicate_rows / total_rows * 100) if total_rows > 0 else 0, 2)
            }
        except Exception as e:
            logger.warning(f"Could not calculate duplicates: {str(e)}")
            return {'duplicate_rows': 0, 'duplicate_percentage': 0.0}
    
    def get_analysis_summary(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary for a specific analysis
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            dict: Analysis summary
        """
        return self.storage.get_analysis_summary(analysis_id)
    
    def get_column_stats(self, analysis_id: str) -> List[Dict[str, Any]]:
        """
        Get column statistics for a specific analysis
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            list: Column statistics
        """
        return self.storage.get_column_stats(analysis_id)
    
    def export_results(
        self,
        analysis_id: str,
        format: str = 'json',
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export analysis results to file
        
        Args:
            analysis_id: Analysis ID
            format: Output format ('json' or 'csv')
            output_path: Output file path
            
        Returns:
            str: Output file path
        """
        try:
            logger.info(f"Exporting results for analysis {analysis_id}...")
            
            # Get data
            summary = self.get_analysis_summary(analysis_id)
            column_stats = self.get_column_stats(analysis_id)
            
            if not summary:
                logger.error(f"Analysis {analysis_id} not found")
                return None
            
            results = {
                'summary': summary,
                'column_statistics': column_stats
            }
            
            # Determine output path
            if not output_path:
                output_path = f"eda_results_{analysis_id}.{format}"
            
            # Export based on format
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"✓ Results exported to: {output_path}")
                return output_path
            
            elif format == 'csv':
                import pandas as pd
                df = pd.DataFrame(column_stats)
                df.to_csv(output_path, index=False)
                logger.info(f"✓ Column stats exported to: {output_path}")
                return output_path
            
            else:
                logger.error(f"Unsupported format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return None
    
    def close(self) -> None:
        """Close connection"""
        self.conn.close()
        logger.info("EDA Agent connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False
