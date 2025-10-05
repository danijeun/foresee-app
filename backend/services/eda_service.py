"""
EDA Service - Automatic exploratory data analysis after upload
Integrates the EDA Agent with the upload workflow
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from eda_agent package
from agents.eda_agent import SnowflakeEDAAgent
from agents.eda_agent.database import SnowflakeConnection, SchemaManager, ResultsStorage
from services.config import Config


class WorkflowEDAService:
    """
    Service that automatically runs EDA on uploaded datasets
    Stores results in the same workflow schema
    """
    
    def __init__(self, workflow_schema: str):
        """
        Initialize EDA service for a specific workflow
        
        Args:
            workflow_schema: Workflow schema name (e.g., "WORKFLOW_12345")
        """
        self.workflow_schema = workflow_schema
        self.conn = None
        
    def run_eda_after_upload(
        self,
        table_name: str,
        workflow_id: str,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive EDA on a newly uploaded table
        
        Args:
            table_name: Name of the uploaded table
            workflow_id: Workflow ID
            target_column: Optional target variable for supervised EDA
            
        Returns:
            dict: EDA results summary
        """
        try:
            print(f"\n📊 Starting EDA Analysis for {table_name}...")
            
            # Initialize connection
            self.conn = SnowflakeConnection(
                account=Config.SNOWFLAKE_ACCOUNT,
                user=Config.SNOWFLAKE_USER,
                password=Config.SNOWFLAKE_PASSWORD,
                warehouse=Config.INGESTION_WAREHOUSE,
                database=Config.SNOWFLAKE_DATABASE,
                schema=self.workflow_schema
            )
            
            # Create EDA tables in the workflow schema
            schema_manager = SchemaManager(self.conn)
            schema_manager.create_eda_tables(self.workflow_schema)
            print(f"  ✓ EDA tables ready in {self.workflow_schema}")
            
            # Initialize EDA agent
            agent = SnowflakeEDAAgent(
                account=Config.SNOWFLAKE_ACCOUNT,
                user=Config.SNOWFLAKE_USER,
                password=Config.SNOWFLAKE_PASSWORD,
                warehouse=Config.INGESTION_WAREHOUSE,
                database=Config.SNOWFLAKE_DATABASE,
                schema=self.workflow_schema,
                workflow_schema=self.workflow_schema,  # Store results in same schema
                log_level='INFO'
            )
            
            # Run EDA analysis
            print(f"  🔍 Analyzing table structure and statistics...")
            analysis_id = agent.analyze_table(
                table_name=table_name,
                schema=self.workflow_schema,
                target_column=target_column
            )
            
            print(f"  ✓ EDA completed! Analysis ID: {analysis_id}")
            
            # Get summary
            summary = agent.get_analysis_summary(analysis_id)
            
            # Get column statistics
            column_stats = agent.get_column_stats(analysis_id)
            
            # Close connections
            agent.close()
            self.conn.close()
            
            # Prepare response
            result = {
                'success': True,
                'analysis_id': analysis_id,
                'workflow_id': workflow_id,
                'summary': {
                    'table_name': summary['table_name'],
                    'total_rows': summary['total_rows'],
                    'total_columns': summary['total_columns'],
                    'duplicate_rows': summary['duplicate_rows'],
                    'duplicate_percentage': summary['duplicate_percentage'],
                    'analysis_type': summary['analysis_type'],
                    'target_column': summary.get('target_column')
                },
                'column_stats_count': len(column_stats),
                'results_location': {
                    'schema': self.workflow_schema,
                    'summary_table': f"{self.workflow_schema}.workflow_eda_summary",
                    'stats_table': f"{self.workflow_schema}.workflow_eda_column_stats"
                }
            }
            
            print(f"\n✅ EDA Analysis Complete!")
            print(f"   - Rows analyzed: {summary['total_rows']:,}")
            print(f"   - Columns analyzed: {summary['total_columns']}")
            print(f"   - Duplicates found: {summary['duplicate_rows']:,}")
            
            return result
            
        except Exception as e:
            print(f"❌ EDA Analysis Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'workflow_id': workflow_id,
                'table_name': table_name
            }
        
        finally:
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
    
    def get_eda_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve EDA results for a specific analysis
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            dict: Complete EDA results including summary and column stats
        """
        try:
            # Initialize connection
            conn = SnowflakeConnection(
                account=Config.SNOWFLAKE_ACCOUNT,
                user=Config.SNOWFLAKE_USER,
                password=Config.SNOWFLAKE_PASSWORD,
                warehouse=Config.INGESTION_WAREHOUSE,
                database=Config.SNOWFLAKE_DATABASE,
                schema=self.workflow_schema
            )
            
            # Get results
            storage = ResultsStorage(conn, self.workflow_schema)
            summary = storage.get_analysis_summary(analysis_id)
            column_stats = storage.get_column_stats(analysis_id)
            
            conn.close()
            
            if not summary:
                return None
            
            return {
                'analysis_id': analysis_id,
                'summary': summary,
                'column_statistics': column_stats
            }
            
        except Exception as e:
            print(f"❌ Error retrieving EDA results: {str(e)}")
            return None
    
    def get_latest_eda_for_table(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent EDA analysis for a table
        
        Args:
            table_name: Table name
            
        Returns:
            dict: Latest EDA results or None
        """
        try:
            conn = SnowflakeConnection(
                account=Config.SNOWFLAKE_ACCOUNT,
                user=Config.SNOWFLAKE_USER,
                password=Config.SNOWFLAKE_PASSWORD,
                warehouse=Config.INGESTION_WAREHOUSE,
                database=Config.SNOWFLAKE_DATABASE,
                schema=self.workflow_schema
            )
            
            # Query for latest analysis
            query = f"""
                SELECT analysis_id
                FROM {self.workflow_schema}.workflow_eda_summary
                WHERE table_name = '{table_name}'
                ORDER BY analysis_timestamp DESC
                LIMIT 1
            """
            
            conn.execute(query)
            result = conn.fetch_one()
            
            if not result:
                conn.close()
                return None
            
            analysis_id = result[0]
            conn.close()
            
            # Get full results
            return self.get_eda_results(analysis_id)
            
        except Exception as e:
            print(f"❌ Error retrieving latest EDA: {str(e)}")
            return None
