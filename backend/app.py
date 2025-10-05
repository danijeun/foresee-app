"""
Flask API for Foresee App
Connects the frontend with backend services for ML analysis
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import traceback
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))

from services.workflow_manager import WorkflowManager
from services.snowflake_ingestion import SnowflakeCSVUploader
from services.eda_service import WorkflowEDAService
from agents.target_variable_agent import TargetVariableAgent
from agents.logistic_regression_agent import LogisticRegressionAgent
from agents.decision_tree_agent import DecisionTreeAgent
from agents.xgboost_agent import XGBoostAgent
from agents.natural_language_agent import NaturalLanguageAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
ALLOWED_EXTENSIONS = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_workflow_reports(workflow_id: str):
    """
    Clean up old PDF and JSON reports for a specific workflow
    Called when starting a new workflow to remove previous analysis files
    
    Args:
        workflow_id: UUID of the workflow to clean up
    """
    try:
        backend_dir = Path(__file__).parent
        
        # Clean up PDFs
        pdf_dir = backend_dir / "pdf"
        pdf_files = []
        if pdf_dir.exists():
            pdf_pattern = f"insights_{workflow_id}_*.pdf"
            pdf_files = list(pdf_dir.glob(pdf_pattern))
            for pdf_file in pdf_files:
                try:
                    pdf_file.unlink()
                    print(f"   🗑️  Deleted old PDF: {pdf_file.name}")
                except Exception as e:
                    print(f"   ⚠️  Could not delete PDF {pdf_file.name}: {e}")
        
        # Clean up JSONs
        insights_dir = backend_dir / "insights"
        json_files = []
        if insights_dir.exists():
            json_pattern = f"insights_{workflow_id}_*.json"
            json_files = list(insights_dir.glob(json_pattern))
            for json_file in json_files:
                try:
                    json_file.unlink()
                    print(f"   🗑️  Deleted old JSON: {json_file.name}")
                except Exception as e:
                    print(f"   ⚠️  Could not delete JSON {json_file.name}: {e}")
        
        if pdf_files or json_files:
            print(f"   ✓ Cleaned up {len(pdf_files)} PDFs and {len(json_files)} JSONs for workflow {workflow_id}")
    
    except Exception as e:
        print(f"   ⚠️  Error during cleanup: {e}")


def run_eda_agent(schema_name, table_name, workflow_id):
    """
    Run EDA Agent in a separate thread
    
    Returns:
        dict: EDA results or error
    """
    try:
        print(f"🤖 [EDA Agent] Starting analysis for {table_name}...")
        start_time = time.time()
        
        eda_service = WorkflowEDAService(schema_name)
        eda_result = eda_service.run_eda_after_upload(
            table_name=table_name,
            workflow_id=workflow_id,
            target_column=None
        )
        
        elapsed = time.time() - start_time
        print(f"✅ [EDA Agent] Completed in {elapsed:.2f}s")
        
        return {
            'success': True,
            'data': eda_result,
            'elapsed_time': round(elapsed, 2)
        }
    except Exception as e:
        print(f"❌ [EDA Agent] Error: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }


def run_target_variable_agent(workflow_id, table_name, sample_size=100):
    """
    Run Target Variable Agent in a separate thread
    
    Returns:
        dict: Target variable suggestions or error
    """
    try:
        print(f"🎯 [Target Agent] Analyzing target variables for {table_name}...")
        start_time = time.time()
        
        agent = TargetVariableAgent()
        result = agent.analyze_workflow(
            workflow_id=workflow_id,
            table_name=table_name,
            sample_size=sample_size
        )
        
        elapsed = time.time() - start_time
        print(f"✅ [Target Agent] Completed in {elapsed:.2f}s")
        
        return {
            'success': True,
            'data': {
                'recommendations': result.get('suggestions', {}).get('recommendations', []),
                'ranking_rationale': result.get('suggestions', {}).get('ranking_rationale', ''),
                'total_columns': len(result.get('columns', [])),
                'row_count': result.get('row_count', 0)
            },
            'elapsed_time': round(elapsed, 2)
        }
    except Exception as e:
        print(f"❌ [Target Agent] Error: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Foresee API',
        'version': '1.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """
    Upload CSV file and create a workflow for analysis
    Uses system temporary directory - no local folder created
    
    Returns:
        JSON with workflow info and upload results
    """
    temp_file = None
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Get optional workflow name
        workflow_name = request.form.get('workflow_name', 'Unnamed Analysis')
        
        # Create temporary file in system temp directory
        filename = secure_filename(file.filename)
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False)
        filepath = temp_file.name
        
        # Save uploaded file to temp location
        file.save(filepath)
        temp_file.close()
        
        print(f"📁 File saved to system temp: {filepath}")
        
        # Initialize workflow manager
        print("🔌 Connecting to Snowflake...")
        manager = WorkflowManager()
        
        # Create a new workflow
        print("📦 Creating workflow...")
        workflow = manager.create_workflow(workflow_name=workflow_name)
        
        # Clean up old reports for this workflow
        print("🧹 Cleaning up old reports...")
        cleanup_workflow_reports(workflow['workflow_id'])
        
        # Get uploader for this workflow
        print("📤 Uploading CSV to Snowflake...")
        uploader = manager.get_workflow_uploader(
            schema_name=workflow['schema_name']
        )
        
        # Upload CSV to Snowflake
        table_name = filename.rsplit('.', 1)[0].replace('-', '_').replace(' ', '_')
        result = uploader.upload_csv(filepath, table_name)
        
        print("\n" + "=" * 70)
        print("🚀 Running EDA & Target Analysis in PARALLEL...")
        print("=" * 70)
        
        # Start timer
        start_time = time.time()
        
        # START Target Variable Agent FIRST (it's faster - should finish in ~45s)
        executor = ThreadPoolExecutor(max_workers=2)
        future_target = executor.submit(
            run_target_variable_agent,
            workflow['workflow_id'],
            table_name,
            100  # sample_size
        )
        print("🎯 Target Agent launched!")
        
        # Fetch preview data
        print("🔍 Fetching preview data...")
        preview_data = uploader.query(f"SELECT * FROM {table_name} LIMIT 5")
        column_names = [desc[0] for desc in uploader.cursor.description]
        
        # Clean up upload connections
        uploader.close()
        manager.close()
        
        # NOW start EDA Agent (runs in parallel with Target Agent)
        print("🤖 Starting EDA Agent...")
        future_eda = executor.submit(
            run_eda_agent,
            workflow['schema_name'],
            table_name,
            workflow['workflow_id']
        )
        
        # Wait for BOTH to complete
        print("\n⏳ Waiting for both agents to complete...")
        eda_result = None
        target_result = None
        
        for future in as_completed([future_eda, future_target]):
            if future == future_target:
                target_result = future.result()
                if target_result['success']:
                    print(f"✅ Target Agent finished! ({target_result['elapsed_time']}s)")
            elif future == future_eda:
                eda_result = future.result()
                if eda_result['success']:
                    print(f"✅ EDA Agent finished! ({eda_result['elapsed_time']}s)")
        
        # Clean up executor
        executor.shutdown(wait=False)
        
        total_time = time.time() - start_time
        print(f"\n⚡ Total parallel execution time: {total_time:.2f}s")
        
        # Calculate time saved
        sequential_time = (eda_result.get('elapsed_time', 0) if eda_result else 0) + \
                         (target_result.get('elapsed_time', 0) if target_result else 0)
        time_saved = sequential_time - total_time
        if time_saved > 0:
            print(f"💨 Time saved: {time_saved:.2f}s ({(time_saved/sequential_time*100):.1f}% faster)")
        
        print("=" * 70)
        
        # Prepare response
        response = {
            'success': True,
            'message': 'File uploaded - EDA and Target Analysis completed in parallel',
            'workflow': {
                'id': workflow['workflow_id'],
                'schema': workflow['schema_name'],
                'name': workflow['workflow_name'],
                'created_at': workflow['created_at']
            },
            'upload': {
                'filename': filename,
                'table_name': table_name,
                'file_size_mb': result['file_size_mb'],
                'rows_loaded': result['rows_loaded'],
                'method': result['method']
            },
            'preview': {
                'columns': column_names,
                'data': [list(row) for row in preview_data]
            },
            'eda': eda_result['data'] if eda_result and eda_result['success'] else {'error': eda_result.get('error', 'EDA failed') if eda_result else 'EDA not run'},
            'target_suggestions': target_result['data'] if target_result and target_result['success'] else {'error': target_result.get('error', 'Target analysis failed') if target_result else 'Target analysis not run'},
            'performance': {
                'eda_time': eda_result.get('elapsed_time', 0) if eda_result else 0,
                'target_time': target_result.get('elapsed_time', 0) if target_result else 0,
                'total_time': round(total_time, 2),
                'time_saved': round(time_saved, 2) if time_saved > 0 else 0
            }
        }
        
        print("✅ Upload and parallel analysis completed")
        print("=" * 70)
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500
    
    finally:
        # Always clean up the temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
                print(f"🗑️ Temporary file deleted from system temp")
            except Exception as e:
                print(f"⚠️ Could not delete temporary file: {e}")


@app.route('/api/workflows', methods=['GET'])
def list_workflows():
    """
    List all active workflows
    
    Returns:
        JSON with list of workflows
    """
    try:
        manager = WorkflowManager()
        workflows = manager.list_workflows()
        manager.close()
        
        return jsonify({
            'success': True,
            'count': len(workflows),
            'workflows': workflows
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/workflow/<workflow_id>', methods=['DELETE'])
def delete_workflow(workflow_id):
    """
    Delete a workflow
    
    Args:
        workflow_id: UUID of the workflow to delete
        
    Returns:
        JSON with success message
    """
    try:
        manager = WorkflowManager()
        manager.delete_workflow(workflow_id=workflow_id)
        manager.close()
        
        return jsonify({
            'success': True,
            'message': f'Workflow {workflow_id} deleted successfully'
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def execute_query():
    """
    Execute a SQL query on a workflow
    
    Body:
        {
            "workflow_id": "uuid",
            "query": "SELECT * FROM table_name LIMIT 10"
        }
        
    Returns:
        JSON with query results
    """
    try:
        data = request.get_json()
        workflow_id = data.get('workflow_id')
        query = data.get('query')
        
        if not workflow_id or not query:
            return jsonify({'error': 'workflow_id and query are required'}), 400
        
        manager = WorkflowManager()
        uploader = manager.get_workflow_uploader(workflow_id=workflow_id)
        
        # Execute query
        results = uploader.query(query)
        column_names = [desc[0] for desc in uploader.cursor.description]
        
        uploader.close()
        manager.close()
        
        return jsonify({
            'success': True,
            'columns': column_names,
            'data': [list(row) for row in results],
            'row_count': len(results)
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/target-suggestions/<workflow_id>/<table_name>', methods=['GET'])
def get_target_suggestions(workflow_id, table_name):
    """
    Get target variable suggestions using the Target Variable Agent
    
    Args:
        workflow_id: Workflow UUID
        table_name: Table name to analyze
        
    Query Parameters:
        sample_size: Number of rows to sample (default: 100)
        
    Returns:
        JSON with ranked target variable suggestions
    """
    try:
        sample_size = request.args.get('sample_size', 100, type=int)
        
        print(f"\n🎯 Getting target suggestions for workflow {workflow_id}, table {table_name}")
        
        # Initialize Target Variable Agent
        agent = TargetVariableAgent()
        
        # Analyze the workflow
        result = agent.analyze_workflow(
            workflow_id=workflow_id,
            table_name=table_name,
            sample_size=sample_size
        )
        
        # Extract recommendations
        recommendations = result.get('suggestions', {}).get('recommendations', [])
        ranking_rationale = result.get('suggestions', {}).get('ranking_rationale', '')
        
        print(f"  ✓ Found {len(recommendations)} target variable suggestions")
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'table_name': table_name,
            'recommendations': recommendations,
            'ranking_rationale': ranking_rationale,
            'total_columns': len(result.get('columns', [])),
            'row_count': result.get('row_count', 0)
        }), 200
        
    except Exception as e:
        print(f"❌ Error getting target suggestions: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/analyze-target-variable', methods=['POST'])
def analyze_target_variable():
    """
    Analyze a workflow's dataset and extract the most important target variables
    ranked by importance (1-100 score) using Gemini AI
    
    Body:
        {
            "workflow_id": "uuid",
            "table_name": "optional_table_name",
            "sample_size": 100  // optional, default 100
        }
        
    Returns:
        JSON with top 5 target variables ranked by importance, including:
        - Importance scores (1-100)
        - Predictability assessment (HIGH/MEDIUM/LOW)
        - Problem type (regression/classification)
        - Suggested features
        - Ranking rationale
    """
    try:
        data = request.get_json()
        workflow_id = data.get('workflow_id')
        table_name = data.get('table_name')
        sample_size = data.get('sample_size', 100)
        
        if not workflow_id:
            return jsonify({'error': 'workflow_id is required'}), 400
        
        print(f"🤖 Starting target variable analysis for workflow: {workflow_id}")
        
        # Initialize the agent
        agent = TargetVariableAgent()
        
        # Analyze the workflow
        result = agent.analyze_workflow(
            workflow_id=workflow_id,
            table_name=table_name,
            sample_size=sample_size
        )
        
        print(f"✅ Analysis completed successfully")
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/select-target', methods=['POST'])
def select_target_variable(workflow_id):
    """
    Save the user's selected target variable to the workflow_eda_summary table
    and automatically trigger ML model training (Logistic Regression + Decision Tree + XGBoost)
    
    Args:
        workflow_id: Workflow UUID
        
    Body:
        {
            "target_variable": "column_name",
            "table_name": "table_name",
            "problem_type": "regression/classification",
            "importance_score": 95
        }
        
    Returns:
        JSON with success confirmation and all three ML models training results
    """
    try:
        data = request.get_json()
        target_variable = data.get('target_variable')
        table_name = data.get('table_name')
        problem_type = data.get('problem_type', 'unknown')
        importance_score = data.get('importance_score')
        
        if not target_variable or not table_name:
            return jsonify({'error': 'target_variable and table_name are required'}), 400
        
        print(f"\n🎯 Saving target variable selection for workflow {workflow_id}")
        print(f"   Target: {target_variable}")
        print(f"   Table: {table_name}")
        print(f"   Type: {problem_type}")
        
        schema_name = f"WORKFLOW_{workflow_id}"
        
        # Initialize workflow manager
        manager = WorkflowManager()
        uploader = manager.get_workflow_uploader(schema_name=schema_name)
        
        # Update the workflow_eda_summary table with the selected target
        update_query = f"""
            UPDATE {schema_name}.workflow_eda_summary
            SET 
                target_column = '{target_variable}',
                analysis_type = '{problem_type}'
            WHERE table_name = '{table_name}'
        """
        
        uploader.cursor.execute(update_query)
        rows_updated = uploader.cursor.rowcount
        uploader.conn.commit()
        
        if rows_updated > 0:
            print(f"  ✓ Updated workflow_eda_summary with target variable")
        else:
            print(f"  ⚠️  No EDA summary found for table {table_name}")
        
        # Also save to workflow metadata for easy retrieval
        metadata_insert = f"""
            MERGE INTO {schema_name}.workflow_metadata AS target
            USING (
                SELECT 
                    'target_variable' as key,
                    PARSE_JSON('{{"variable": "{target_variable}", "table": "{table_name}", "type": "{problem_type}", "score": {importance_score or "null"}}}') as value,
                    CURRENT_TIMESTAMP() as updated_at
            ) AS source
            ON target.key = source.key
            WHEN MATCHED THEN
                UPDATE SET value = source.value, updated_at = source.updated_at
            WHEN NOT MATCHED THEN
                INSERT (key, value, updated_at) VALUES (source.key, source.value, source.updated_at)
        """
        uploader.cursor.execute(metadata_insert)
        uploader.conn.commit()
        
        print(f"  ✓ Target variable saved to workflow metadata")
        
        uploader.close()
        manager.close()
        
        # Automatically start ML model training after target selection
        print("\n" + "=" * 70)
        print("🤖 AUTO-STARTING ML MODEL TRAINING")
        print("=" * 70)
        
        lr_result = None
        lr_error = None
        dt_result = None
        dt_error = None
        
        # Train Logistic Regression
        try:
            print("\n" + "=" * 70)
            print("🤖 LOGISTIC REGRESSION TRAINING")
            print("=" * 70)
            
            lr_agent = LogisticRegressionAgent()
            lr_result = lr_agent.train_and_evaluate(
                workflow_id=workflow_id,
                table_name=table_name,
                test_size=0.2,
                random_state=42,
                max_iter=1000
            )
            
            print("=" * 70)
            print("✅ Logistic Regression Training Completed!")
            print("=" * 70)
            
        except Exception as lr_e:
            lr_error = str(lr_e)
            print(f"❌ Logistic Regression Training Failed: {lr_error}")
            traceback.print_exc()
        
        # Train Decision Tree
        try:
            print("\n" + "=" * 70)
            print("🌳 DECISION TREE TRAINING")
            print("=" * 70)
            
            dt_agent = DecisionTreeAgent()
            dt_result = dt_agent.train_and_evaluate(
                workflow_id=workflow_id,
                table_name=table_name,
                test_size=0.2,
                random_state=42,
                max_depth=10  # Limit depth to prevent overfitting
            )
            
            print("=" * 70)
            print("✅ Decision Tree Training Completed!")
            print("=" * 70)
            
        except Exception as dt_e:
            dt_error = str(dt_e)
            print(f"❌ Decision Tree Training Failed: {dt_error}")
            traceback.print_exc()
        
        # Train XGBoost
        xgb_result = None
        xgb_error = None
        
        try:
            print("\n" + "=" * 70)
            print("🚀 XGBOOST TRAINING")
            print("=" * 70)
            
            xgb_agent = XGBoostAgent()
            xgb_result = xgb_agent.train_and_evaluate(
                workflow_id=workflow_id,
                table_name=table_name,
                test_size=0.2,
                random_state=42,
                n_estimators=100,
                max_depth=6
            )
            
            print("=" * 70)
            print("✅ XGBoost Training Completed!")
            print("=" * 70)
            
        except Exception as xgb_e:
            xgb_error = str(xgb_e)
            print(f"❌ XGBoost Training Failed: {xgb_error}")
            traceback.print_exc()
        
        # Prepare response with both target selection and ML results
        response = {
            'success': True,
            'message': 'Target variable saved and ML models training completed',
            'workflow_id': workflow_id,
            'target_variable': target_variable,
            'table_name': table_name,
            'problem_type': problem_type,
            'importance_score': importance_score
        }
        
        # Add Logistic Regression results if available
        if lr_result and lr_result.get('success'):
            response['logistic_regression'] = {
                'success': True,
                'model_type': 'Logistic Regression',
                'insights': {
                    'test_accuracy': lr_result['insights']['test_accuracy'],
                    'test_precision': lr_result['insights']['test_precision'],
                    'test_recall': lr_result['insights']['test_recall'],
                    'test_f1': lr_result['insights']['test_f1'],
                    'test_roc_auc': lr_result['insights']['test_roc_auc'],
                    'total_samples': lr_result['insights']['total_samples'],
                    'total_features': lr_result['insights']['total_features'],
                    'n_classes': lr_result['insights']['n_classes'],
                    'performance_summary': lr_result['insights']['performance_summary'],
                    'top_features': lr_result['insights']['top_features'][:5],
                    'recommendations': lr_result['insights']['recommendations']
                },
                'results_location': f"WORKFLOW_{workflow_id}.LOGISTIC_REGRESSION_SUMMARY"
            }
        elif lr_error:
            response['logistic_regression'] = {
                'success': False,
                'error': lr_error,
                'message': 'Logistic Regression training failed'
            }
        
        # Add Decision Tree results if available
        if dt_result and dt_result.get('success'):
            response['decision_tree'] = {
                'success': True,
                'model_type': 'Decision Tree',
                'insights': {
                    'test_accuracy': dt_result['insights']['test_accuracy'],
                    'test_precision': dt_result['insights']['test_precision'],
                    'test_recall': dt_result['insights']['test_recall'],
                    'test_f1': dt_result['insights']['test_f1'],
                    'test_roc_auc': dt_result['insights']['test_roc_auc'],
                    'tree_depth': dt_result['insights']['tree_depth'],
                    'n_leaves': dt_result['insights']['n_leaves'],
                    'total_samples': dt_result['insights']['total_samples'],
                    'total_features': dt_result['insights']['total_features'],
                    'n_classes': dt_result['insights']['n_classes'],
                    'performance_summary': dt_result['insights']['performance_summary'],
                    'top_features': dt_result['insights']['top_features'][:5],
                    'recommendations': dt_result['insights']['recommendations']
                },
                'results_location': f"WORKFLOW_{workflow_id}.DECISION_TREE_SUMMARY"
            }
        elif dt_error:
            response['decision_tree'] = {
                'success': False,
                'error': dt_error,
                'message': 'Decision Tree training failed'
            }
        
        # Add XGBoost results if available
        if xgb_result and xgb_result.get('success'):
            response['xgboost'] = {
                'success': True,
                'model_type': 'XGBoost',
                'insights': {
                    'test_accuracy': xgb_result['insights']['test_accuracy'],
                    'test_precision': xgb_result['insights']['test_precision'],
                    'test_recall': xgb_result['insights']['test_recall'],
                    'test_f1': xgb_result['insights']['test_f1'],
                    'test_roc_auc': xgb_result['insights']['test_roc_auc'],
                    'n_estimators': xgb_result['insights']['n_estimators'],
                    'max_depth': xgb_result['insights']['max_depth'],
                    'total_samples': xgb_result['insights']['total_samples'],
                    'total_features': xgb_result['insights']['total_features'],
                    'n_classes': xgb_result['insights']['n_classes'],
                    'performance_summary': xgb_result['insights']['performance_summary'],
                    'top_features': xgb_result['insights']['top_features'][:5],
                    'recommendations': xgb_result['insights']['recommendations']
                },
                'results_location': f"WORKFLOW_{workflow_id}.XGBOOST_SUMMARY"
            }
        elif xgb_error:
            response['xgboost'] = {
                'success': False,
                'error': xgb_error,
                'message': 'XGBoost training failed'
            }
        
        # Automatically generate Natural Language Insights after ML models complete
        nl_result = None
        nl_error = None
        
        try:
            print("\n" + "=" * 70)
            print("📝 GENERATING NATURAL LANGUAGE INSIGHTS")
            print("=" * 70)
            
            # Generate output path for JSON file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backend_dir = Path(__file__).parent
            insights_dir = backend_dir / "insights"
            output_path = insights_dir / f"insights_{workflow_id}_{timestamp}.json"
            output_path = str(output_path)
            
            nl_agent = NaturalLanguageAgent()
            nl_result = nl_agent.generate_insights(
                workflow_id=workflow_id,
                table_name=table_name,
                output_path=output_path
            )
            
            print("=" * 70)
            print("✅ Natural Language Insights Generated!")
            print("=" * 70)
            
        except Exception as nl_e:
            nl_error = str(nl_e)
            print(f"❌ Natural Language Insights Generation Failed: {nl_error}")
            traceback.print_exc()
        
        # Add Natural Language Insights to response
        if nl_result and nl_result.get('success'):
            response['natural_language_insights'] = {
                'success': True,
                'json_path': nl_result.get('output_path'),
                'pdf_path': nl_result.get('pdf_path'),
                'insights': nl_result.get('insights'),
                'message': 'Natural language insights and PDF report generated successfully'
            }
        elif nl_error:
            response['natural_language_insights'] = {
                'success': False,
                'error': nl_error,
                'message': 'Natural language insights generation failed'
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error saving target variable: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/train-logistic-regression', methods=['POST'])
def train_logistic_regression(workflow_id):
    """
    Train a logistic regression model on the selected target variable
    
    Args:
        workflow_id: Workflow UUID
        
    Body:
        {
            "table_name": "table_name",
            "test_size": 0.2,  // optional, default 0.2
            "random_state": 42,  // optional, default 42
            "max_iter": 1000  // optional, default 1000
        }
        
    Returns:
        JSON with model insights and performance metrics
    """
    try:
        data = request.get_json()
        table_name = data.get('table_name')
        test_size = data.get('test_size', 0.2)
        random_state = data.get('random_state', 42)
        max_iter = data.get('max_iter', 1000)
        
        if not table_name:
            return jsonify({'error': 'table_name is required'}), 400
        
        print("\n" + "=" * 70)
        print(f"🤖 Starting Logistic Regression Training")
        print(f"   Workflow: {workflow_id}")
        print(f"   Table: {table_name}")
        print("=" * 70)
        
        # Initialize Logistic Regression Agent
        agent = LogisticRegressionAgent()
        
        # Train and evaluate model
        result = agent.train_and_evaluate(
            workflow_id=workflow_id,
            table_name=table_name,
            test_size=test_size,
            random_state=random_state,
            max_iter=max_iter
        )
        
        print("=" * 70)
        print("✅ Logistic Regression Training Completed")
        print("=" * 70)
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'table_name': table_name,
            'model_type': 'Logistic Regression',
            'insights': {
                'target_variable': result['insights']['target_variable'],
                'test_accuracy': result['insights']['test_accuracy'],
                'test_precision': result['insights']['test_precision'],
                'test_recall': result['insights']['test_recall'],
                'test_f1': result['insights']['test_f1'],
                'test_roc_auc': result['insights']['test_roc_auc'],
                'total_samples': result['insights']['total_samples'],
                'total_features': result['insights']['total_features'],
                'n_classes': result['insights']['n_classes'],
                'performance_summary': result['insights']['performance_summary'],
                'top_features': result['insights']['top_features'][:5],
                'top_features_summary': result['insights']['top_features_summary'],
                'data_quality_summary': result['insights']['data_quality_summary'],
                'recommendations': result['insights']['recommendations']
            },
            'results_location': f"WORKFLOW_{workflow_id}.LOGISTIC_REGRESSION_SUMMARY"
        }), 200
        
    except Exception as e:
        print(f"❌ Error training logistic regression: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/train-decision-tree', methods=['POST'])
def train_decision_tree(workflow_id):
    """
    Train a decision tree model on the selected target variable
    
    Args:
        workflow_id: Workflow UUID
        
    Body:
        {
            "table_name": "table_name",
            "test_size": 0.2,  // optional, default 0.2
            "random_state": 42,  // optional, default 42
            "max_depth": 10,  // optional, default None (unlimited)
            "min_samples_split": 2,  // optional, default 2
            "min_samples_leaf": 1  // optional, default 1
        }
        
    Returns:
        JSON with model insights and performance metrics
    """
    try:
        data = request.get_json()
        table_name = data.get('table_name')
        test_size = data.get('test_size', 0.2)
        random_state = data.get('random_state', 42)
        max_depth = data.get('max_depth')  # None = unlimited
        min_samples_split = data.get('min_samples_split', 2)
        min_samples_leaf = data.get('min_samples_leaf', 1)
        
        if not table_name:
            return jsonify({'error': 'table_name is required'}), 400
        
        print("\n" + "=" * 70)
        print(f"🌳 Starting Decision Tree Training")
        print(f"   Workflow: {workflow_id}")
        print(f"   Table: {table_name}")
        print("=" * 70)
        
        # Initialize Decision Tree Agent
        agent = DecisionTreeAgent()
        
        # Train and evaluate model
        result = agent.train_and_evaluate(
            workflow_id=workflow_id,
            table_name=table_name,
            test_size=test_size,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        
        print("=" * 70)
        print("✅ Decision Tree Training Completed")
        print("=" * 70)
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'table_name': table_name,
            'model_type': 'Decision Tree',
            'insights': {
                'target_variable': result['insights']['target_variable'],
                'test_accuracy': result['insights']['test_accuracy'],
                'test_precision': result['insights']['test_precision'],
                'test_recall': result['insights']['test_recall'],
                'test_f1': result['insights']['test_f1'],
                'test_roc_auc': result['insights']['test_roc_auc'],
                'tree_depth': result['insights']['tree_depth'],
                'n_leaves': result['insights']['n_leaves'],
                'total_samples': result['insights']['total_samples'],
                'total_features': result['insights']['total_features'],
                'n_classes': result['insights']['n_classes'],
                'performance_summary': result['insights']['performance_summary'],
                'top_features': result['insights']['top_features'][:5],
                'top_features_summary': result['insights']['top_features_summary'],
                'data_quality_summary': result['insights']['data_quality_summary'],
                'model_config_summary': result['insights']['model_config_summary'],
                'recommendations': result['insights']['recommendations']
            },
            'results_location': f"WORKFLOW_{workflow_id}.DECISION_TREE_SUMMARY"
        }), 200
        
    except Exception as e:
        print(f"❌ Error training decision tree: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/decision-tree-results', methods=['GET'])
def get_decision_tree_results(workflow_id):
    """
    Get decision tree results from the DECISION_TREE_SUMMARY table
    
    Args:
        workflow_id: Workflow UUID
        
    Query Parameters:
        table_name: Optional table name to filter results
        
    Returns:
        JSON with decision tree analysis results
    """
    try:
        table_name = request.args.get('table_name')
        schema_name = f"WORKFLOW_{workflow_id}"
        
        print(f"\n🔍 Retrieving decision tree results for workflow {workflow_id}")
        
        # Initialize workflow manager
        manager = WorkflowManager()
        uploader = manager.get_workflow_uploader(schema_name=schema_name)
        
        # Build query
        if table_name:
            query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    target_variable,
                    model_type,
                    problem_type,
                    test_accuracy,
                    test_precision,
                    test_recall,
                    test_f1_score,
                    test_roc_auc,
                    train_accuracy,
                    tree_depth,
                    n_leaves,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    total_samples,
                    total_features,
                    n_classes,
                    performance_summary,
                    top_features_summary,
                    data_quality_summary,
                    model_config_summary,
                    confusion_matrix,
                    top_features,
                    recommendations,
                    created_at
                FROM {schema_name}.DECISION_TREE_SUMMARY
                WHERE table_name = '{table_name}'
                ORDER BY created_at DESC
            """
        else:
            query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    target_variable,
                    model_type,
                    problem_type,
                    test_accuracy,
                    test_precision,
                    test_recall,
                    test_f1_score,
                    test_roc_auc,
                    train_accuracy,
                    tree_depth,
                    n_leaves,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    total_samples,
                    total_features,
                    n_classes,
                    performance_summary,
                    top_features_summary,
                    data_quality_summary,
                    model_config_summary,
                    confusion_matrix,
                    top_features,
                    recommendations,
                    created_at
                FROM {schema_name}.DECISION_TREE_SUMMARY
                ORDER BY created_at DESC
            """
        
        uploader.cursor.execute(query)
        results = uploader.cursor.fetchall()
        
        uploader.close()
        manager.close()
        
        if not results:
            return jsonify({
                'success': True,
                'message': 'No decision tree results found',
                'results': []
            }), 200
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                'analysis_id': row[0],
                'table_name': row[1],
                'target_variable': row[2],
                'model_type': row[3],
                'problem_type': row[4],
                'test_accuracy': row[5],
                'test_precision': row[6],
                'test_recall': row[7],
                'test_f1_score': row[8],
                'test_roc_auc': row[9],
                'train_accuracy': row[10],
                'tree_depth': row[11],
                'n_leaves': row[12],
                'max_depth': row[13],
                'min_samples_split': row[14],
                'min_samples_leaf': row[15],
                'total_samples': row[16],
                'total_features': row[17],
                'n_classes': row[18],
                'performance_summary': row[19],
                'top_features_summary': row[20],
                'data_quality_summary': row[21],
                'model_config_summary': row[22],
                'confusion_matrix': row[23],
                'top_features': row[24],
                'recommendations': row[25],
                'created_at': str(row[26]) if row[26] else None
            })
        
        print(f"  ✓ Found {len(formatted_results)} result(s)")
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'count': len(formatted_results),
            'results': formatted_results
        }), 200
        
    except Exception as e:
        print(f"❌ Error retrieving decision tree results: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/train-xgboost', methods=['POST'])
def train_xgboost(workflow_id):
    """
    Train an XGBoost model on the selected target variable
    
    Args:
        workflow_id: Workflow UUID
        
    Body:
        {
            "table_name": "table_name",
            "test_size": 0.2,  // optional, default 0.2
            "random_state": 42,  // optional, default 42
            "n_estimators": 100,  // optional, default 100
            "max_depth": 6,  // optional, default 6
            "learning_rate": 0.3,  // optional, default 0.3
            "subsample": 1.0,  // optional, default 1.0
            "colsample_bytree": 1.0  // optional, default 1.0
        }
        
    Returns:
        JSON with model insights and performance metrics
    """
    try:
        data = request.get_json()
        table_name = data.get('table_name')
        test_size = data.get('test_size', 0.2)
        random_state = data.get('random_state', 42)
        n_estimators = data.get('n_estimators', 100)
        max_depth = data.get('max_depth', 6)
        learning_rate = data.get('learning_rate', 0.3)
        subsample = data.get('subsample', 1.0)
        colsample_bytree = data.get('colsample_bytree', 1.0)
        
        if not table_name:
            return jsonify({'error': 'table_name is required'}), 400
        
        print("\n" + "=" * 70)
        print(f"🚀 Starting XGBoost Training")
        print(f"   Workflow: {workflow_id}")
        print(f"   Table: {table_name}")
        print("=" * 70)
        
        # Initialize XGBoost Agent
        agent = XGBoostAgent()
        
        # Train and evaluate model
        result = agent.train_and_evaluate(
            workflow_id=workflow_id,
            table_name=table_name,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )
        
        print("=" * 70)
        print("✅ XGBoost Training Completed")
        print("=" * 70)
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'table_name': table_name,
            'model_type': 'XGBoost',
            'insights': {
                'target_variable': result['insights']['target_variable'],
                'test_accuracy': result['insights']['test_accuracy'],
                'test_precision': result['insights']['test_precision'],
                'test_recall': result['insights']['test_recall'],
                'test_f1': result['insights']['test_f1'],
                'test_roc_auc': result['insights']['test_roc_auc'],
                'n_estimators': result['insights']['n_estimators'],
                'max_depth': result['insights']['max_depth'],
                'learning_rate': result['insights']['learning_rate'],
                'total_samples': result['insights']['total_samples'],
                'total_features': result['insights']['total_features'],
                'n_classes': result['insights']['n_classes'],
                'performance_summary': result['insights']['performance_summary'],
                'top_features': result['insights']['top_features'][:5],
                'top_features_summary': result['insights']['top_features_summary'],
                'data_quality_summary': result['insights']['data_quality_summary'],
                'model_config_summary': result['insights']['model_config_summary'],
                'recommendations': result['insights']['recommendations']
            },
            'results_location': f"WORKFLOW_{workflow_id}.XGBOOST_SUMMARY"
        }), 200
        
    except Exception as e:
        print(f"❌ Error training XGBoost: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/xgboost-results', methods=['GET'])
def get_xgboost_results(workflow_id):
    """
    Get XGBoost results from the XGBOOST_SUMMARY table
    
    Args:
        workflow_id: Workflow UUID
        
    Query Parameters:
        table_name: Optional table name to filter results
        
    Returns:
        JSON with XGBoost analysis results
    """
    try:
        table_name = request.args.get('table_name')
        schema_name = f"WORKFLOW_{workflow_id}"
        
        print(f"\n🔍 Retrieving XGBoost results for workflow {workflow_id}")
        
        # Initialize workflow manager
        manager = WorkflowManager()
        uploader = manager.get_workflow_uploader(schema_name=schema_name)
        
        # Build query
        if table_name:
            query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    target_variable,
                    model_type,
                    problem_type,
                    test_accuracy,
                    test_precision,
                    test_recall,
                    test_f1_score,
                    test_roc_auc,
                    train_accuracy,
                    n_estimators,
                    max_depth,
                    learning_rate,
                    subsample,
                    colsample_bytree,
                    total_samples,
                    total_features,
                    n_classes,
                    performance_summary,
                    top_features_summary,
                    data_quality_summary,
                    model_config_summary,
                    confusion_matrix,
                    top_features,
                    recommendations,
                    created_at
                FROM {schema_name}.XGBOOST_SUMMARY
                WHERE table_name = '{table_name}'
                ORDER BY created_at DESC
            """
        else:
            query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    target_variable,
                    model_type,
                    problem_type,
                    test_accuracy,
                    test_precision,
                    test_recall,
                    test_f1_score,
                    test_roc_auc,
                    train_accuracy,
                    n_estimators,
                    max_depth,
                    learning_rate,
                    subsample,
                    colsample_bytree,
                    total_samples,
                    total_features,
                    n_classes,
                    performance_summary,
                    top_features_summary,
                    data_quality_summary,
                    model_config_summary,
                    confusion_matrix,
                    top_features,
                    recommendations,
                    created_at
                FROM {schema_name}.XGBOOST_SUMMARY
                ORDER BY created_at DESC
            """
        
        uploader.cursor.execute(query)
        results = uploader.cursor.fetchall()
        
        uploader.close()
        manager.close()
        
        if not results:
            return jsonify({
                'success': True,
                'message': 'No XGBoost results found',
                'results': []
            }), 200
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                'analysis_id': row[0],
                'table_name': row[1],
                'target_variable': row[2],
                'model_type': row[3],
                'problem_type': row[4],
                'test_accuracy': row[5],
                'test_precision': row[6],
                'test_recall': row[7],
                'test_f1_score': row[8],
                'test_roc_auc': row[9],
                'train_accuracy': row[10],
                'n_estimators': row[11],
                'max_depth': row[12],
                'learning_rate': row[13],
                'subsample': row[14],
                'colsample_bytree': row[15],
                'total_samples': row[16],
                'total_features': row[17],
                'n_classes': row[18],
                'performance_summary': row[19],
                'top_features_summary': row[20],
                'data_quality_summary': row[21],
                'model_config_summary': row[22],
                'confusion_matrix': row[23],
                'top_features': row[24],
                'recommendations': row[25],
                'created_at': str(row[26]) if row[26] else None
            })
        
        print(f"  ✓ Found {len(formatted_results)} result(s)")
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'count': len(formatted_results),
            'results': formatted_results
        }), 200
        
    except Exception as e:
        print(f"❌ Error retrieving XGBoost results: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/logistic-regression-results', methods=['GET'])
def get_logistic_regression_results(workflow_id):
    """
    Get logistic regression results from the LOGISTIC_REGRESSION_SUMMARY table
    
    Args:
        workflow_id: Workflow UUID
        
    Query Parameters:
        table_name: Optional table name to filter results
        
    Returns:
        JSON with logistic regression analysis results
    """
    try:
        table_name = request.args.get('table_name')
        schema_name = f"WORKFLOW_{workflow_id}"
        
        print(f"\n🔍 Retrieving logistic regression results for workflow {workflow_id}")
        
        # Initialize workflow manager
        manager = WorkflowManager()
        uploader = manager.get_workflow_uploader(schema_name=schema_name)
        
        # Build query
        if table_name:
            query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    target_variable,
                    model_type,
                    problem_type,
                    test_accuracy,
                    test_precision,
                    test_recall,
                    test_f1_score,
                    test_roc_auc,
                    train_accuracy,
                    total_samples,
                    total_features,
                    n_classes,
                    performance_summary,
                    top_features_summary,
                    data_quality_summary,
                    confusion_matrix,
                    top_features,
                    recommendations,
                    created_at
                FROM {schema_name}.LOGISTIC_REGRESSION_SUMMARY
                WHERE table_name = '{table_name}'
                ORDER BY created_at DESC
            """
        else:
            query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    target_variable,
                    model_type,
                    problem_type,
                    test_accuracy,
                    test_precision,
                    test_recall,
                    test_f1_score,
                    test_roc_auc,
                    train_accuracy,
                    total_samples,
                    total_features,
                    n_classes,
                    performance_summary,
                    top_features_summary,
                    data_quality_summary,
                    confusion_matrix,
                    top_features,
                    recommendations,
                    created_at
                FROM {schema_name}.LOGISTIC_REGRESSION_SUMMARY
                ORDER BY created_at DESC
            """
        
        uploader.cursor.execute(query)
        results = uploader.cursor.fetchall()
        
        uploader.close()
        manager.close()
        
        if not results:
            return jsonify({
                'success': True,
                'message': 'No logistic regression results found',
                'results': []
            }), 200
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                'analysis_id': row[0],
                'table_name': row[1],
                'target_variable': row[2],
                'model_type': row[3],
                'problem_type': row[4],
                'test_accuracy': row[5],
                'test_precision': row[6],
                'test_recall': row[7],
                'test_f1_score': row[8],
                'test_roc_auc': row[9],
                'train_accuracy': row[10],
                'total_samples': row[11],
                'total_features': row[12],
                'n_classes': row[13],
                'performance_summary': row[14],
                'top_features_summary': row[15],
                'data_quality_summary': row[16],
                'confusion_matrix': row[17],
                'top_features': row[18],
                'recommendations': row[19],
                'created_at': str(row[20]) if row[20] else None
            })
        
        print(f"  ✓ Found {len(formatted_results)} result(s)")
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'count': len(formatted_results),
            'results': formatted_results
        }), 200
        
    except Exception as e:
        print(f"❌ Error retrieving logistic regression results: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/generate-insights', methods=['POST'])
def generate_natural_language_insights(workflow_id):
    """
    Generate natural language insights from EDA and ML models using Gemini
    
    Args:
        workflow_id: Workflow UUID
        
    Body:
        {
            "table_name": "table_name",
            "save_to_file": false  // optional, default false
        }
        
    Returns:
        JSON with comprehensive natural language insights
    """
    try:
        data = request.get_json()
        table_name = data.get('table_name')
        save_to_file = data.get('save_to_file', False)
        
        if not table_name:
            return jsonify({'error': 'table_name is required'}), 400
        
        print("\n" + "=" * 70)
        print(f"📝 Generating Natural Language Insights")
        print(f"   Workflow: {workflow_id}")
        print(f"   Table: {table_name}")
        print("=" * 70)
        
        # Initialize Natural Language Agent
        agent = NaturalLanguageAgent()
        
        # Determine output path
        output_path = None
        if save_to_file:
            output_path = f"insights_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Generate insights
        result = agent.generate_insights(
            workflow_id=workflow_id,
            table_name=table_name,
            output_path=output_path
        )
        
        print("=" * 70)
        print("✅ Natural Language Insights Generated")
        print("=" * 70)
        
        return jsonify({
            'success': True,
            'workflow_id': workflow_id,
            'table_name': table_name,
            'insights': result['insights'],
            'output_path': output_path
        }), 200
        
    except Exception as e:
        print(f"❌ Error generating insights: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/insights-summary', methods=['GET'])
def get_insights_summary(workflow_id):
    """
    Get a quick summary of available insights
    
    Args:
        workflow_id: Workflow UUID
        
    Query Parameters:
        table_name: Table name to check
        
    Returns:
        JSON with summary of available data for insights generation
    """
    try:
        table_name = request.args.get('table_name')
        
        if not table_name:
            return jsonify({'error': 'table_name query parameter is required'}), 400
        
        schema_name = f"WORKFLOW_{workflow_id}"
        
        print(f"\n📊 Checking available insights data for workflow {workflow_id}")
        
        # Initialize workflow manager
        manager = WorkflowManager()
        uploader = manager.get_workflow_uploader(schema_name=schema_name)
        
        summary = {
            'workflow_id': workflow_id,
            'table_name': table_name,
            'eda_available': False,
            'ml_models_available': []
        }
        
        # Check EDA data
        try:
            uploader.cursor.execute(f"""
                SELECT COUNT(*) FROM {schema_name}.workflow_eda_summary
                WHERE table_name = '{table_name}'
            """)
            eda_count = uploader.cursor.fetchone()[0]
            summary['eda_available'] = eda_count > 0
        except:
            pass
        
        # Check ML models
        model_tables = [
            ('LOGISTIC_REGRESSION_SUMMARY', 'Logistic Regression'),
            ('DECISION_TREE_SUMMARY', 'Decision Tree'),
            ('XGBOOST_SUMMARY', 'XGBoost')
        ]
        
        for table, model_type in model_tables:
            try:
                uploader.cursor.execute(f"""
                    SELECT test_accuracy FROM {schema_name}.{table}
                    WHERE table_name = '{table_name}'
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                result = uploader.cursor.fetchone()
                if result:
                    summary['ml_models_available'].append({
                        'model_type': model_type,
                        'accuracy': float(result[0])
                    })
            except:
                pass
        
        uploader.close()
        manager.close()
        
        print(f"  ✓ EDA Available: {summary['eda_available']}")
        print(f"  ✓ ML Models Available: {len(summary['ml_models_available'])}")
        
        return jsonify({
            'success': True,
            'summary': summary,
            'ready_for_insights': summary['eda_available'] or len(summary['ml_models_available']) > 0
        }), 200
        
    except Exception as e:
        print(f"❌ Error checking insights data: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/report/view', methods=['GET'])
def view_report(workflow_id):
    """
    View the latest generated PDF report for a workflow in browser
    
    Args:
        workflow_id: Workflow UUID
        
    Returns:
        PDF file for viewing
    """
    try:
        # Find the latest PDF for this workflow
        backend_dir = Path(__file__).parent
        pdf_dir = backend_dir / "pdf"
        
        if not pdf_dir.exists():
            return jsonify({'error': 'No PDF reports found'}), 404
        
        # Find all PDFs for this workflow
        pdf_pattern = f"insights_{workflow_id}_*.pdf"
        pdf_files = list(pdf_dir.glob(pdf_pattern))
        
        if not pdf_files:
            return jsonify({'error': f'No PDF report found for workflow {workflow_id}'}), 404
        
        # Get the most recent PDF
        latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
        
        print(f"📄 Viewing PDF: {latest_pdf.name}")
        
        from flask import send_file
        return send_file(
            latest_pdf,
            mimetype='application/pdf',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"❌ Error viewing PDF: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/report/download', methods=['GET'])
def download_report(workflow_id):
    """
    Download the latest generated PDF report for a workflow
    
    Args:
        workflow_id: Workflow UUID
        
    Returns:
        PDF file as download
    """
    try:
        # Find the latest PDF for this workflow
        backend_dir = Path(__file__).parent
        pdf_dir = backend_dir / "pdf"
        
        if not pdf_dir.exists():
            return jsonify({'error': 'No PDF reports found'}), 404
        
        # Find all PDFs for this workflow
        pdf_pattern = f"insights_{workflow_id}_*.pdf"
        pdf_files = list(pdf_dir.glob(pdf_pattern))
        
        if not pdf_files:
            return jsonify({'error': f'No PDF report found for workflow {workflow_id}'}), 404
        
        # Get the most recent PDF
        latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
        
        print(f"📥 Downloading PDF: {latest_pdf.name}")
        
        from flask import send_file
        return send_file(
            latest_pdf,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"ML_Report_{workflow_id}.pdf"
        )
        
    except Exception as e:
        print(f"❌ Error downloading PDF: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/download-pdf/<filename>', methods=['GET'])
def download_pdf(workflow_id, filename):
    """
    Download a specific PDF report by filename
    
    Args:
        workflow_id: Workflow UUID
        filename: PDF filename
        
    Returns:
        PDF file as download
    """
    try:
        # Construct PDF path
        backend_dir = Path(__file__).parent
        pdf_path = backend_dir / "pdf" / filename
        
        if not pdf_path.exists():
            return jsonify({'error': 'PDF file not found'}), 404
        
        from flask import send_file
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"❌ Error downloading PDF: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/workflow/<workflow_id>/view-pdf/<filename>', methods=['GET'])
def view_pdf(workflow_id, filename):
    """
    View a specific PDF report by filename in browser
    
    Args:
        workflow_id: Workflow UUID
        filename: PDF filename
        
    Returns:
        PDF file for viewing
    """
    try:
        # Construct PDF path
        backend_dir = Path(__file__).parent
        pdf_path = backend_dir / "pdf" / filename
        
        if not pdf_path.exists():
            return jsonify({'error': 'PDF file not found'}), 404
        
        from flask import send_file
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"❌ Error viewing PDF: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Foresee API Server Starting")
    print("=" * 70)
    print("📍 API will be available at: http://localhost:5000")
    print("📍 Frontend should connect to: http://localhost:5000/api/")
    print("=" * 70)
    
    # Disable reloader to prevent OSError with parallel threads on Windows
    # The reloader causes socket errors when multiple threads are running
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Prevents Windows socket errors with ThreadPoolExecutor
    )
