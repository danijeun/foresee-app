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

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))

from services.workflow_manager import WorkflowManager
from services.snowflake_ingestion import SnowflakeCSVUploader
from services.eda_service import WorkflowEDAService
from agents.target_variable_agent import TargetVariableAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
ALLOWED_EXTENSIONS = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        JSON with success confirmation
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
        
        return jsonify({
            'success': True,
            'message': 'Target variable saved successfully',
            'workflow_id': workflow_id,
            'target_variable': target_variable,
            'table_name': table_name,
            'problem_type': problem_type,
            'importance_score': importance_score
        }), 200
        
    except Exception as e:
        print(f"❌ Error saving target variable: {str(e)}")
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
