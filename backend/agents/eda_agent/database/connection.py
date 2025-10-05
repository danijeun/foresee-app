"""
Snowflake connection management
"""
import snowflake.connector
from typing import Optional, Dict, Any
from ..utils.logger import get_logger
from ..config import EDAConfig

logger = get_logger(__name__)


class SnowflakeConnection:
    """
    Manages Snowflake database connections with proper error handling
    """
    
    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Snowflake connection
        
        Args:
            account: Snowflake account identifier
            user: Username
            password: Password
            warehouse: Warehouse name
            database: Database name
            schema: Schema name
            role: Role name
            **kwargs: Additional connection parameters
        """
        self.account = account or EDAConfig.SNOWFLAKE_ACCOUNT
        self.user = user or EDAConfig.SNOWFLAKE_USER
        self.password = password or EDAConfig.SNOWFLAKE_PASSWORD
        self.warehouse = warehouse or EDAConfig.SNOWFLAKE_WAREHOUSE
        self.database = database or EDAConfig.SNOWFLAKE_DATABASE
        self.schema = schema or EDAConfig.SNOWFLAKE_SCHEMA
        self.role = role
        
        self.conn: Optional[snowflake.connector.SnowflakeConnection] = None
        self.cursor: Optional[snowflake.connector.cursor.SnowflakeCursor] = None
        
        # Validate required parameters
        if not all([self.account, self.user, self.password]):
            raise ValueError(
                "Missing required connection parameters. "
                "Provide account, user, and password either as arguments or environment variables."
            )
        
        # Additional connection parameters
        self.extra_params = kwargs
        
        # Connect
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Snowflake"""
        try:
            logger.info(f"Connecting to Snowflake account: {self.account}")
            
            conn_params = {
                'account': self.account,
                'user': self.user,
                'password': self.password,
                'warehouse': self.warehouse
            }
            
            # Add optional parameters
            if self.database:
                conn_params['database'] = self.database
            if self.schema:
                conn_params['schema'] = self.schema
            if self.role:
                conn_params['role'] = self.role
            
            # Add extra parameters
            conn_params.update(self.extra_params)
            
            self.conn = snowflake.connector.connect(**conn_params)
            self.cursor = self.conn.cursor()
            
            logger.info("✓ Successfully connected to Snowflake")
            
            # Log connection details
            if self.database:
                logger.info(f"  Database: {self.database}")
            if self.schema:
                logger.info(f"  Schema: {self.schema}")
            if self.warehouse:
                logger.info(f"  Warehouse: {self.warehouse}")
                
        except snowflake.connector.errors.ProgrammingError as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during connection: {str(e)}")
            raise
    
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a query
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
            
        Returns:
            Cursor execution result
        """
        try:
            if params:
                return self.cursor.execute(query, params)
            return self.cursor.execute(query)
        except snowflake.connector.errors.ProgrammingError as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query[:200]}...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query execution: {str(e)}")
            raise
    
    def fetch_one(self) -> Optional[tuple]:
        """Fetch one result from the cursor"""
        try:
            return self.cursor.fetchone()
        except Exception as e:
            logger.error(f"Error fetching result: {str(e)}")
            return None
    
    def fetch_all(self) -> list:
        """Fetch all results from the cursor"""
        try:
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching results: {str(e)}")
            return []
    
    def fetch_df(self, query: str):
        """
        Execute query and return results as pandas DataFrame
        
        Args:
            query: SQL query
            
        Returns:
            pandas.DataFrame: Query results
        """
        try:
            import pandas as pd
            self.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            results = self.fetch_all()
            return pd.DataFrame(results, columns=columns)
        except ImportError:
            logger.error("pandas is required for fetch_df()")
            raise
        except Exception as e:
            logger.error(f"Error fetching DataFrame: {str(e)}")
            raise
    
    def commit(self) -> None:
        """Commit the transaction"""
        try:
            if self.conn:
                self.conn.commit()
        except Exception as e:
            logger.error(f"Error committing transaction: {str(e)}")
            raise
    
    def rollback(self) -> None:
        """Rollback the transaction"""
        try:
            if self.conn:
                self.conn.rollback()
        except Exception as e:
            logger.error(f"Error rolling back transaction: {str(e)}")
            raise
    
    def use_database(self, database: str) -> None:
        """Switch to a different database"""
        try:
            self.execute(f"USE DATABASE {database}")
            self.database = database
            logger.info(f"Switched to database: {database}")
        except Exception as e:
            logger.error(f"Failed to switch database: {str(e)}")
            raise
    
    def use_schema(self, schema: str) -> None:
        """Switch to a different schema"""
        try:
            self.execute(f"USE SCHEMA {schema}")
            self.schema = schema
            logger.info(f"Switched to schema: {schema}")
        except Exception as e:
            logger.error(f"Failed to switch schema: {str(e)}")
            raise
    
    def use_warehouse(self, warehouse: str) -> None:
        """Switch to a different warehouse"""
        try:
            self.execute(f"USE WAREHOUSE {warehouse}")
            self.warehouse = warehouse
            logger.info(f"Switched to warehouse: {warehouse}")
        except Exception as e:
            logger.error(f"Failed to switch warehouse: {str(e)}")
            raise
    
    def get_column_info(self, table_name: str, schema: Optional[str] = None) -> Dict[str, str]:
        """
        Get column names and data types for a table
        
        Args:
            table_name: Table name
            schema: Schema name (optional, uses current schema if not provided)
            
        Returns:
            dict: Dictionary mapping column names to data types
        """
        try:
            if schema:
                qualified_table = f"{schema}.{table_name}"
            else:
                qualified_table = table_name
            
            self.execute(f"DESC TABLE {qualified_table}")
            results = self.fetch_all()
            
            # Results format: (name, type, kind, null?, default, primary key, unique key, check, expression, comment)
            column_info = {row[0]: row[1] for row in results}
            
            logger.info(f"Retrieved {len(column_info)} columns from {qualified_table}")
            return column_info
            
        except Exception as e:
            logger.error(f"Failed to get column info for {table_name}: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the connection"""
        try:
            if self.cursor:
                self.cursor.close()
                logger.debug("Cursor closed")
            if self.conn:
                self.conn.close()
                logger.info("✓ Snowflake connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False  # Don't suppress exceptions
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self.close()
        except:
            pass
