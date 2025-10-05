"""
Schema and table management for EDA results
"""
from typing import Optional
from ..utils.logger import get_logger
from ..config import EDA_SUMMARY_TABLE_SQL, EDA_COLUMN_STATS_TABLE_SQL
from .connection import SnowflakeConnection

logger = get_logger(__name__)


class SchemaManager:
    """
    Manages database schema and table creation for EDA results
    """
    
    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize schema manager
        
        Args:
            connection: SnowflakeConnection instance
        """
        self.conn = connection
    
    def create_workflow_schema(self, schema_name: str) -> bool:
        """
        Create workflow schema if it doesn't exist
        
        Args:
            schema_name: Name of the schema to create
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Creating schema: {schema_name}")
            self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            logger.info(f"✓ Schema {schema_name} ready")
            return True
        except Exception as e:
            logger.error(f"Failed to create schema {schema_name}: {str(e)}")
            raise
    
    def create_eda_tables(self, schema_name: Optional[str] = None) -> bool:
        """
        Create EDA results tables in the specified schema
        
        Args:
            schema_name: Target schema name (uses connection default if not provided)
            
        Returns:
            bool: True if successful
        """
        try:
            if schema_name:
                # Temporarily switch to target schema
                original_schema = self.conn.schema
                self.conn.use_schema(schema_name)
            
            logger.info(f"Creating EDA tables in schema: {schema_name or self.conn.schema}")
            
            # Create summary table
            summary_sql = EDA_SUMMARY_TABLE_SQL.format(schema=schema_name or self.conn.schema)
            self.conn.execute(summary_sql)
            logger.info("  ✓ workflow_eda_summary table created")
            
            # Create column stats table
            stats_sql = EDA_COLUMN_STATS_TABLE_SQL.format(schema=schema_name or self.conn.schema)
            self.conn.execute(stats_sql)
            logger.info("  ✓ workflow_eda_column_stats table created")
            
            # Restore original schema if changed
            if schema_name and original_schema:
                self.conn.use_schema(original_schema)
            
            logger.info("✓ EDA tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create EDA tables: {str(e)}")
            raise
    
    def table_exists(self, table_name: str, schema_name: Optional[str] = None) -> bool:
        """
        Check if a table exists
        
        Args:
            table_name: Name of the table
            schema_name: Schema name (optional)
            
        Returns:
            bool: True if table exists
        """
        try:
            if schema_name:
                qualified_name = f"{schema_name}.{table_name}"
            else:
                qualified_name = table_name
            
            self.conn.execute(f"SELECT 1 FROM {qualified_name} LIMIT 1")
            return True
        except:
            return False
    
    def drop_eda_tables(self, schema_name: Optional[str] = None, confirm: bool = False) -> bool:
        """
        Drop EDA tables (use with caution!)
        
        Args:
            schema_name: Schema name
            confirm: Must be True to actually drop tables
            
        Returns:
            bool: True if successful
        """
        if not confirm:
            logger.warning("Drop operation not confirmed. Set confirm=True to proceed.")
            return False
        
        try:
            if schema_name:
                original_schema = self.conn.schema
                self.conn.use_schema(schema_name)
            
            logger.warning(f"Dropping EDA tables from schema: {schema_name or self.conn.schema}")
            
            self.conn.execute(f"DROP TABLE IF EXISTS {schema_name or self.conn.schema}.workflow_eda_column_stats")
            logger.info("  ✓ workflow_eda_column_stats dropped")
            
            self.conn.execute(f"DROP TABLE IF EXISTS {schema_name or self.conn.schema}.workflow_eda_summary")
            logger.info("  ✓ workflow_eda_summary dropped")
            
            if schema_name and original_schema:
                self.conn.use_schema(original_schema)
            
            logger.info("✓ EDA tables dropped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop EDA tables: {str(e)}")
            raise
    
    def get_existing_analyses(
        self,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        limit: int = 10
    ) -> list:
        """
        Get list of existing analyses
        
        Args:
            schema_name: Filter by schema name
            table_name: Filter by table name
            limit: Maximum number of results
            
        Returns:
            list: List of analysis records
        """
        try:
            schema = schema_name or self.conn.schema
            query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    schema_name,
                    database_name,
                    total_rows,
                    total_columns,
                    target_column,
                    analysis_type,
                    analysis_timestamp,
                    status
                FROM {schema}.workflow_eda_summary
            """
            
            if table_name:
                query += f" WHERE table_name = '{table_name}'"
            
            query += f" ORDER BY analysis_timestamp DESC LIMIT {limit}"
            
            self.conn.execute(query)
            results = self.conn.fetch_all()
            
            analyses = []
            for row in results:
                analyses.append({
                    'analysis_id': row[0],
                    'table_name': row[1],
                    'schema_name': row[2],
                    'database_name': row[3],
                    'total_rows': row[4],
                    'total_columns': row[5],
                    'target_column': row[6],
                    'analysis_type': row[7],
                    'analysis_timestamp': row[8],
                    'status': row[9]
                })
            
            return analyses
            
        except Exception as e:
            logger.error(f"Failed to retrieve existing analyses: {str(e)}")
            return []
    
    def delete_analysis(self, analysis_id: str, schema_name: Optional[str] = None) -> bool:
        """
        Delete an analysis and its associated data
        
        Args:
            analysis_id: Analysis ID to delete
            schema_name: Schema name
            
        Returns:
            bool: True if successful
        """
        try:
            schema = schema_name or self.conn.schema
            
            logger.info(f"Deleting analysis: {analysis_id}")
            
            # Delete column stats first (foreign key constraint)
            self.conn.execute(
                f"DELETE FROM {schema}.workflow_eda_column_stats WHERE analysis_id = '{analysis_id}'"
            )
            
            # Delete summary
            self.conn.execute(
                f"DELETE FROM {schema}.workflow_eda_summary WHERE analysis_id = '{analysis_id}'"
            )
            
            self.conn.commit()
            logger.info(f"✓ Analysis {analysis_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete analysis {analysis_id}: {str(e)}")
            self.conn.rollback()
            return False
