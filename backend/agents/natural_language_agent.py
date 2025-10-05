"""
Natural Language Agent for Foresee App
Generates natural language insights from EDA and ML models using Gemini LLM
"""
import sys
from pathlib import Path
import json
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.workflow_manager import WorkflowManager
from services.config import Config


class NaturalLanguageAgent:
    """
    Agent that generates natural language insights from EDA and ML model results
    """
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize the Natural Language Agent
        
        Args:
            gemini_api_key: Google Gemini API key (uses env variable if not provided)
        """
        self.workflow_manager = None
        
        # Configure Gemini API
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment")
        
        genai.configure(api_key=api_key)
        
        # Use Gemini model
        model_name = 'models/gemini-2.0-flash-exp'
        
        try:
            self.model = genai.GenerativeModel(model_name)
            print(f"✅ Using Gemini model: {model_name}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini model {model_name}: {str(e)}")
    
    def generate_insights(
        self,
        workflow_id: str,
        table_name: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate natural language insights from EDA and ML models
        
        Args:
            workflow_id: UUID of the workflow
            table_name: Name of the table analyzed
            output_path: Optional path to save JSON output
            
        Returns:
            dict: Complete insights in structured JSON format
        """
        try:
            print("=" * 70)
            print("📝 NATURAL LANGUAGE INSIGHTS GENERATOR")
            print("=" * 70)
            
            # Initialize workflow manager
            self.workflow_manager = WorkflowManager()
            schema_name = f"WORKFLOW_{workflow_id}"
            uploader = self.workflow_manager.get_workflow_uploader(schema_name=schema_name)
            
            # Step 1: Collect EDA data
            print(f"\n📊 Step 1: Collecting EDA insights...")
            eda_data = self._collect_eda_data(uploader, schema_name, table_name)
            print(f"   ✓ EDA data collected")
            
            # Step 2: Collect ML model results
            print(f"\n🤖 Step 2: Collecting ML model results...")
            ml_data = self._collect_ml_data(uploader, schema_name, table_name)
            print(f"   ✓ Found {len(ml_data)} ML model(s)")
            
            # Step 3: Generate natural language insights using Gemini
            print(f"\n💡 Step 3: Generating natural language insights...")
            insights = self._generate_nl_insights(eda_data, ml_data, table_name)
            print(f"   ✓ Insights generated")
            
            # Step 4: Structure the output
            print(f"\n📋 Step 4: Structuring output...")
            output = self._structure_output(
                workflow_id,
                table_name,
                eda_data,
                ml_data,
                insights
            )
            
            # Step 5: Save to JSON if path provided
            if output_path:
                print(f"\n💾 Step 5: Saving to file...")
                self._save_to_json(output, output_path)
                print(f"   ✓ Saved to: {output_path}")
            
            # Clean up
            uploader.close()
            self.workflow_manager.close()
            
            print("\n" + "=" * 70)
            print("✅ NATURAL LANGUAGE INSIGHTS GENERATION COMPLETED!")
            print("=" * 70)
            
            return {
                'success': True,
                'workflow_id': workflow_id,
                'table_name': table_name,
                'insights': output,
                'output_path': output_path
            }
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            raise
        finally:
            if self.workflow_manager:
                self.workflow_manager.close()
    
    def _collect_eda_data(
        self,
        uploader,
        schema_name: str,
        table_name: str
    ) -> Dict[str, Any]:
        """
        Collect EDA data from workflow_eda_summary and workflow_eda_column_stats
        """
        eda_data = {
            'summary': None,
            'columns': []
        }
        
        # Get EDA summary
        try:
            summary_query = f"""
                SELECT 
                    analysis_id,
                    table_name,
                    total_rows,
                    total_columns,
                    duplicate_rows,
                    duplicate_percentage,
                    target_column,
                    analysis_type
                FROM {schema_name}.workflow_eda_summary
                WHERE table_name = '{table_name}'
                ORDER BY analysis_id DESC
                LIMIT 1
            """
            
            uploader.cursor.execute(summary_query)
            result = uploader.cursor.fetchone()
            
            if result:
                eda_data['summary'] = {
                    'analysis_id': result[0],
                    'table_name': result[1],
                    'total_rows': result[2],
                    'total_columns': result[3],
                    'duplicate_rows': result[4],
                    'duplicate_percentage': result[5],
                    'target_column': result[6],
                    'analysis_type': result[7]
                }
        except Exception as e:
            print(f"   ⚠️  Could not fetch EDA summary: {e}")
        
        # Get column statistics (top 20 most important)
        try:
            if eda_data['summary']:
                columns_query = f"""
                    SELECT 
                        column_name,
                        data_type,
                        null_percentage,
                        unique_count,
                        cardinality_ratio,
                        mean_value,
                        median_value,
                        std_dev,
                        min_value,
                        max_value
                    FROM {schema_name}.workflow_eda_column_stats
                    WHERE analysis_id = '{eda_data['summary']['analysis_id']}'
                    ORDER BY null_percentage ASC, unique_count DESC
                    LIMIT 20
                """
                
                uploader.cursor.execute(columns_query)
                results = uploader.cursor.fetchall()
                
                for row in results:
                    eda_data['columns'].append({
                        'column_name': row[0],
                        'data_type': row[1],
                        'null_percentage': row[2],
                        'unique_count': row[3],
                        'cardinality_ratio': row[4],
                        'mean_value': row[5],
                        'median_value': row[6],
                        'std_dev': row[7],
                        'min_value': str(row[8]) if row[8] is not None else None,
                        'max_value': str(row[9]) if row[9] is not None else None
                    })
        except Exception as e:
            print(f"   ⚠️  Could not fetch column stats: {e}")
        
        return eda_data
    
    def _collect_ml_data(
        self,
        uploader,
        schema_name: str,
        table_name: str
    ) -> List[Dict[str, Any]]:
        """
        Collect ML model results from all model summary tables
        """
        ml_models = []
        
        # Tables to check
        model_tables = [
            ('LOGISTIC_REGRESSION_SUMMARY', 'Logistic Regression'),
            ('DECISION_TREE_SUMMARY', 'Decision Tree'),
            ('XGBOOST_SUMMARY', 'XGBoost')
        ]
        
        for table, model_type in model_tables:
            try:
                query = f"""
                    SELECT 
                        analysis_id,
                        target_variable,
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
                        recommendations
                    FROM {schema_name}.{table}
                    WHERE table_name = '{table_name}'
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                
                uploader.cursor.execute(query)
                result = uploader.cursor.fetchone()
                
                if result:
                    model_data = {
                        'model_type': model_type,
                        'analysis_id': result[0],
                        'target_variable': result[1],
                        'test_accuracy': float(result[2]) if result[2] is not None else None,
                        'test_precision': float(result[3]) if result[3] is not None else None,
                        'test_recall': float(result[4]) if result[4] is not None else None,
                        'test_f1_score': float(result[5]) if result[5] is not None else None,
                        'test_roc_auc': float(result[6]) if result[6] is not None else None,
                        'train_accuracy': float(result[7]) if result[7] is not None else None,
                        'total_samples': result[8],
                        'total_features': result[9],
                        'n_classes': result[10],
                        'performance_summary': result[11],
                        'top_features_summary': result[12],
                        'recommendations': result[13]
                    }
                    
                    ml_models.append(model_data)
                    print(f"   ✓ {model_type}: {model_data['test_accuracy']:.2%} accuracy")
                    
            except Exception as e:
                print(f"   ℹ️  {model_type} not found or error: {e}")
        
        return ml_models
    
    def _generate_nl_insights(
        self,
        eda_data: Dict[str, Any],
        ml_data: List[Dict[str, Any]],
        table_name: str
    ) -> Dict[str, Any]:
        """
        Use Gemini to generate natural language insights
        """
        # Prepare data summary for prompt
        prompt = self._create_insights_prompt(eda_data, ml_data, table_name)
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Parse the response
            insights = self._parse_gemini_response(response.text)
            
            return insights
            
        except Exception as e:
            print(f"⚠️ Error calling Gemini API: {e}")
            # Return fallback insights if LLM fails
            return self._generate_fallback_insights(eda_data, ml_data)
    
    def _create_insights_prompt(
        self,
        eda_data: Dict[str, Any],
        ml_data: List[Dict[str, Any]],
        table_name: str
    ) -> str:
        """
        Create a detailed prompt for Gemini to generate insights
        """
        # EDA summary
        eda_summary = ""
        if eda_data['summary']:
            s = eda_data['summary']
            eda_summary = f"""
EXPLORATORY DATA ANALYSIS (EDA):
- Dataset: {s['table_name']}
- Total Rows: {s['total_rows']:,}
- Total Columns: {s['total_columns']}
- Duplicate Rows: {s['duplicate_rows']:,} ({s['duplicate_percentage']:.1f}%)
- Target Variable: {s['target_column']}
- Analysis Type: {s['analysis_type']}

Top Columns:
"""
            for col in eda_data['columns'][:10]:
                eda_summary += f"\n- {col['column_name']} ({col['data_type']}): "
                eda_summary += f"{col['null_percentage']:.1f}% null, {col['unique_count']} unique values"
        
        # ML models summary
        ml_summary = "\nMACHINE LEARNING MODELS:\n"
        if ml_data:
            for model in ml_data:
                ml_summary += f"\n{model['model_type']}:"
                ml_summary += f"\n- Target: {model['target_variable']}"
                ml_summary += f"\n- Accuracy: {model['test_accuracy']:.2%}"
                ml_summary += f"\n- Precision: {model['test_precision']:.2%}"
                ml_summary += f"\n- Recall: {model['test_recall']:.2%}"
                ml_summary += f"\n- F1-Score: {model['test_f1_score']:.2%}"
                ml_summary += f"\n- Classes: {model['n_classes']}"
                ml_summary += f"\n{model['top_features_summary']}\n"
        else:
            ml_summary += "No ML models trained yet."
        
        prompt = f"""
You are a data science expert analyzing results from an automated machine learning pipeline.

{eda_summary}

{ml_summary}

Please generate a comprehensive, professional analysis report in JSON format with the following sections:

1. **executive_summary**: A 2-3 paragraph high-level overview of the dataset and key findings.

2. **data_quality_insights**: Detailed analysis of data quality including:
   - Overall data completeness
   - Key data quality issues
   - Recommendations for data improvement

3. **eda_insights**: Key exploratory data analysis findings including:
   - Dataset characteristics
   - Important patterns or anomalies
   - Variable relationships
   - Data distribution insights

4. **ml_performance_comparison**: Compare all ML models (if multiple exist):
   - Best performing model and why
   - Performance metrics comparison
   - Trade-offs between models
   - Overfitting/underfitting analysis

5. **feature_importance_analysis**: Analysis of most important features:
   - Top predictive features
   - Feature impact on predictions
   - Feature engineering suggestions

6. **business_recommendations**: Actionable business recommendations:
   - How to use these models
   - Deployment recommendations
   - Risk considerations
   - Next steps

7. **technical_recommendations**: Technical next steps:
   - Model improvements
   - Data collection recommendations
   - Monitoring suggestions

Format your response as a valid JSON object with these keys. Each section should contain well-written, professional prose.
Be specific, quantitative, and actionable. Use actual numbers from the data.
"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's response into structured insights
        """
        try:
            # Try to extract JSON from response
            # Sometimes LLM adds markdown code blocks
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            insights = json.loads(json_text)
            return insights
            
        except Exception as e:
            print(f"⚠️ Could not parse JSON response: {e}")
            # Return raw text in a structure
            return {
                'executive_summary': response_text[:500],
                'raw_response': response_text
            }
    
    def _generate_fallback_insights(
        self,
        eda_data: Dict[str, Any],
        ml_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate basic insights if LLM fails
        """
        insights = {
            'executive_summary': '',
            'data_quality_insights': '',
            'eda_insights': '',
            'ml_performance_comparison': '',
            'feature_importance_analysis': '',
            'business_recommendations': [],
            'technical_recommendations': []
        }
        
        # Executive summary
        if eda_data['summary']:
            s = eda_data['summary']
            insights['executive_summary'] = f"Analysis of {s['table_name']} dataset with {s['total_rows']:,} rows and {s['total_columns']} columns. "
            
            if ml_data:
                best_model = max(ml_data, key=lambda x: x['test_accuracy'])
                insights['executive_summary'] += f"Best performing model: {best_model['model_type']} with {best_model['test_accuracy']:.1%} accuracy."
        
        # Data quality
        if eda_data['summary']:
            insights['data_quality_insights'] = f"Dataset has {eda_data['summary']['duplicate_percentage']:.1f}% duplicate rows. "
            high_null = [c for c in eda_data['columns'] if c['null_percentage'] > 10]
            if high_null:
                insights['data_quality_insights'] += f"Found {len(high_null)} columns with >10% missing values."
        
        # ML comparison
        if ml_data:
            comparison = "Model Performance Comparison:\n"
            for model in sorted(ml_data, key=lambda x: x['test_accuracy'], reverse=True):
                comparison += f"- {model['model_type']}: {model['test_accuracy']:.1%} accuracy\n"
            insights['ml_performance_comparison'] = comparison
        
        return insights
    
    def _structure_output(
        self,
        workflow_id: str,
        table_name: str,
        eda_data: Dict[str, Any],
        ml_data: List[Dict[str, Any]],
        insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Structure the final output JSON
        """
        output = {
            'metadata': {
                'workflow_id': workflow_id,
                'table_name': table_name,
                'generated_at': datetime.now().isoformat(),
                'generator': 'Natural Language Agent v1.0'
            },
            'dataset_overview': {
                'total_rows': eda_data['summary']['total_rows'] if eda_data['summary'] else None,
                'total_columns': eda_data['summary']['total_columns'] if eda_data['summary'] else None,
                'target_variable': eda_data['summary']['target_column'] if eda_data['summary'] else None,
                'duplicate_percentage': eda_data['summary']['duplicate_percentage'] if eda_data['summary'] else None
            },
            'ml_models_summary': [
                {
                    'model_type': m['model_type'],
                    'accuracy': m['test_accuracy'],
                    'precision': m['test_precision'],
                    'recall': m['test_recall'],
                    'f1_score': m['test_f1_score'],
                    'roc_auc': m['test_roc_auc']
                }
                for m in ml_data
            ],
            'natural_language_insights': insights,
            'raw_data': {
                'eda': eda_data,
                'ml_models': ml_data
            }
        }
        
        return output
    
    def _save_to_json(self, output: Dict[str, Any], filepath: str):
        """
        Save output to JSON file
        """
        # Ensure the directory exists
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    """
    Example usage of the Natural Language Agent
    """
    import traceback
    
    # Initialize agent
    agent = NaturalLanguageAgent()
    
    # Replace with your actual workflow_id and table_name
    workflow_id = "your_workflow_id_here"
    table_name = "your_table_name"
    
    try:
        print("📝 Starting Natural Language Insights Generation...")
        
        # Generate insights
        result = agent.generate_insights(
            workflow_id=workflow_id,
            table_name=table_name,
            output_path=f"insights_{workflow_id}.json"
        )
        
        if result['success']:
            print("\n✅ Insights Generated Successfully!")
            print(f"\n📊 Executive Summary:")
            print(result['insights']['natural_language_insights'].get('executive_summary', 'N/A'))
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
