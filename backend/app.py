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

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))

from services.workflow_manager import WorkflowManager
from services.snowflake_ingestion import SnowflakeCSVUploader
from services.eda_service import WorkflowEDAService

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
ALLOWED_EXTENSIONS = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        
        # Get preview data
        print("🔍 Fetching preview data...")
        preview_data = uploader.query(f"SELECT * FROM {table_name} LIMIT 5")
        column_names = [desc[0] for desc in uploader.cursor.description]
        
        # Clean up upload connections
        uploader.close()
        manager.close()
        
        # Run EDA Analysis automatically
        print("\n" + "=" * 70)
        print("🤖 Running Automatic EDA Analysis...")
        print("=" * 70)
        
        eda_service = WorkflowEDAService(workflow['schema_name'])
        eda_result = eda_service.run_eda_after_upload(
            table_name=table_name,
            workflow_id=workflow['workflow_id'],
            target_column=None  # Auto-detect or specify via request
        )
        
        # Prepare response with EDA results
        response = {
            'success': True,
            'message': 'File uploaded successfully and EDA completed',
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
            'eda': eda_result  # Include EDA analysis results
        }
        
        print("✅ Upload and EDA completed successfully")
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


@app.route('/api/eda/<workflow_id>/<analysis_id>', methods=['GET'])
def get_eda_results(workflow_id, analysis_id):
    """
    Get EDA results for a specific analysis
    
    Args:
        workflow_id: Workflow UUID
        analysis_id: EDA analysis ID
        
    Returns:
        JSON with complete EDA results
    """
    try:
        schema_name = f"WORKFLOW_{workflow_id}"
        eda_service = WorkflowEDAService(schema_name)
        
        results = eda_service.get_eda_results(analysis_id)
        
        if not results:
            return jsonify({'error': 'EDA analysis not found'}), 404
        
        return jsonify({
            'success': True,
            'data': results
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/eda/<workflow_id>/latest/<table_name>', methods=['GET'])
def get_latest_eda(workflow_id, table_name):
    """
    Get the latest EDA analysis for a specific table
    
    Args:
        workflow_id: Workflow UUID
        table_name: Table name
        
    Returns:
        JSON with latest EDA results
    """
    try:
        schema_name = f"WORKFLOW_{workflow_id}"
        eda_service = WorkflowEDAService(schema_name)
        
        results = eda_service.get_latest_eda_for_table(table_name)
        
        if not results:
            return jsonify({'error': 'No EDA analysis found for this table'}), 404
        
        return jsonify({
            'success': True,
            'data': results
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/eda/<workflow_id>/summary', methods=['GET'])
def get_workflow_eda_summary(workflow_id):
    """
    Get summary of all EDA analyses in a workflow
    
    Args:
        workflow_id: Workflow UUID
        
    Returns:
        JSON with list of all EDA analyses
    """
    try:
        schema_name = f"WORKFLOW_{workflow_id}"
        
        manager = WorkflowManager()
        uploader = manager.get_workflow_uploader(schema_name=schema_name)
        
        # Query all EDA summaries
        query = f"""
            SELECT 
                analysis_id,
                table_name,
                total_rows,
                total_columns,
                duplicate_percentage,
                analysis_type,
                target_column
            FROM {schema_name}.workflow_eda_summary
        """
        
        results = uploader.query(query)
        
        analyses = []
        for row in results:
            analyses.append({
                'analysis_id': row[0],
                'table_name': row[1],
                'total_rows': row[2],
                'total_columns': row[3],
                'duplicate_percentage': row[4],
                'analysis_type': row[5],
                'target_column': row[6]
            })
        
        uploader.close()
        manager.close()
        
        return jsonify({
            'success': True,
            'count': len(analyses),
            'analyses': analyses
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Foresee API Server Starting")
    print("=" * 70)
    print("📍 API will be available at: http://localhost:5000")
    print("📍 Frontend should connect to: http://localhost:5000/api/")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
