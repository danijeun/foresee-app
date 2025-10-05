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
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.units import inch

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
            pdf_path = None
            if output_path:
                print(f"\n💾 Step 5: Saving to JSON file...")
                self._save_to_json(output, output_path)
                print(f"   ✓ Saved to: {output_path}")
                
                # Step 6: Generate PDF report
                print(f"\n📄 Step 6: Generating PDF report...")
                pdf_path = self._generate_pdf_report(output, output_path)
                if pdf_path:
                    print(f"   ✓ PDF saved to: {pdf_path}")
                else:
                    print(f"   ⚠️  PDF generation skipped or failed")
            
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
                'output_path': output_path,
                'pdf_path': pdf_path
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
                        recommendations,
                        confusion_matrix,
                        top_features,
                        feature_importance,
                        preprocessing_details
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
                        'recommendations': result[13],
                        'confusion_matrix': result[14],
                        'top_features': result[15],
                        'feature_importance': result[16],
                        'preprocessing_details': result[17]
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
        Structure the final output JSON with 5 comprehensive sections
        """
        print(f"\n📊 Generating visualizations and tables...")
        
        # Generate visualizations
        model_comparison_chart = self._generate_model_comparison_chart(ml_data)
        
        # Generate confusion matrices for each model
        confusion_matrices = {}
        feature_importance_charts = {}
        
        for model in ml_data:
            if model.get('confusion_matrix'):
                cm_chart = self._generate_confusion_matrix_chart(
                    model['confusion_matrix'],
                    model['model_type']
                )
                if cm_chart:
                    confusion_matrices[model['model_type']] = cm_chart
            
            if model.get('top_features'):
                fi_chart = self._generate_feature_importance_chart(
                    model['top_features'],
                    model['model_type']
                )
                if fi_chart:
                    feature_importance_charts[model['model_type']] = fi_chart
        
        # Generate comparison table
        metrics_table = self._generate_metrics_comparison_table(ml_data)
        
        # Best model
        best_model = max(ml_data, key=lambda x: x['test_accuracy']) if ml_data else None
        
        # SECTION 1: DATASET
        dataset_section = {
            'title': 'Dataset Overview',
            'description': self._generate_dataset_description(eda_data, table_name),
            'statistics': {
                'total_rows': eda_data['summary']['total_rows'] if eda_data['summary'] else 0,
                'total_columns': eda_data['summary']['total_columns'] if eda_data['summary'] else 0,
                'target_variable': eda_data['summary']['target_column'] if eda_data['summary'] else 'N/A',
                'duplicate_rows': eda_data['summary'].get('duplicate_rows', 0) if eda_data['summary'] else 0,
                'duplicate_percentage': f"{eda_data['summary'].get('duplicate_percentage', 0):.2f}%" if eda_data['summary'] else '0%',
                'memory_usage_mb': f"{eda_data['summary']['memory_usage_mb']:.2f}" if (eda_data['summary'] and eda_data['summary'].get('memory_usage_mb')) else 'N/A'
            },
            'column_summary_table': [
                {
                    'Column': col.get('column_name', 'Unknown'),
                    'Type': col.get('data_type', 'Unknown'),
                    'Null %': f"{col.get('null_percentage', 0):.2f}%",
                    'Unique Values': col.get('unique_count', 0),
                    'Mean': f"{col['mean']:.4f}" if col.get('mean') is not None else 'N/A'
                }
                for col in (eda_data['columns'][:15] if eda_data['columns'] else [])
            ]
        }
        
        # SECTION 2: DATA PREPROCESSING
        preprocessing_section = {
            'title': 'Data Preprocessing',
            'description': self._generate_preprocessing_description(ml_data),
            'preprocessing_steps': []
        }
        
        if ml_data and ml_data[0].get('preprocessing_details'):
            prep_details = ml_data[0]['preprocessing_details']
            # Parse preprocessing_details if it's a string
            if isinstance(prep_details, str):
                prep_details = json.loads(prep_details)
            preprocessing_section['preprocessing_steps'] = [
                {
                    'step': 'Missing Values Handling',
                    'description': f"Handled {prep_details.get('missing_values_handled', 0)} missing values through median imputation for numeric columns"
                },
                {
                    'step': 'Categorical Encoding',
                    'description': f"Encoded {len(prep_details.get('encoded_columns', []))} categorical columns using Label Encoding",
                    'columns': prep_details.get('encoded_columns', [])
                },
                {
                    'step': 'Feature Removal',
                    'description': f"Dropped {len(prep_details.get('dropped_columns', []))} columns due to high cardinality, missing values, or data type incompatibility",
                    'columns': prep_details.get('dropped_columns', [])
                },
                {
                    'step': 'Feature Scaling',
                    'description': 'Applied StandardScaler to normalize all numeric features'
                }
            ]
        
        # SECTION 3: MODELS PERFORMANCE SUMMARY
        performance_section = {
            'title': 'Models Performance Summary',
            'description': self._generate_performance_description(ml_data, best_model),
            'metrics_comparison_table': metrics_table,
            'visualizations': {
                'model_comparison_chart': model_comparison_chart,
                'confusion_matrices': confusion_matrices
            },
            'best_model': {
                'name': best_model['model_type'] if best_model else 'N/A',
                'accuracy': f"{best_model['test_accuracy']:.4f}" if best_model else 'N/A',
                'precision': f"{best_model['test_precision']:.4f}" if best_model else 'N/A',
                'recall': f"{best_model['test_recall']:.4f}" if best_model else 'N/A',
                'f1_score': f"{best_model['test_f1_score']:.4f}" if best_model else 'N/A',
                'roc_auc': f"{best_model['test_roc_auc']:.4f}" if best_model else 'N/A'
            } if best_model else None
        }
        
        # SECTION 4: MODEL EXPLAINABILITY
        explainability_section = {
            'title': 'Model Explainability',
            'description': self._generate_explainability_description(ml_data),
            'feature_importance': {},
            'visualizations': feature_importance_charts,
            'interpretation': {}
        }
        
        for model in ml_data:
            if model.get('top_features'):
                top_features = model['top_features']
                # Parse top_features if it's a string
                if isinstance(top_features, str):
                    top_features = json.loads(top_features)
                
                explainability_section['feature_importance'][model['model_type']] = [
                    {
                        'feature': f['feature'],
                        'importance': f"{abs(f.get('coefficient', f.get('importance', 0))):.6f}",
                        'rank': idx + 1
                    }
                    for idx, f in enumerate(top_features[:10])
                ]
        
        # SECTION 5: CONCLUSION
        conclusion_section = {
            'title': 'Conclusion',
            'summary': self._generate_conclusion_summary(eda_data, ml_data, best_model),
            'key_findings': self._generate_key_findings(eda_data, ml_data, best_model),
            'recommendations': self._generate_recommendations(ml_data, best_model),
            'next_steps': [
                'Deploy the best performing model for production use',
                'Monitor model performance with real-world data',
                'Consider ensemble methods to potentially improve accuracy',
                'Investigate feature engineering opportunities for underperforming features',
                'Set up regular model retraining pipeline'
            ]
        }
        
        # Complete output structure
        output = {
            'metadata': {
                'workflow_id': workflow_id,
                'table_name': table_name,
                'generated_at': datetime.now().isoformat(),
                'generator': 'Foresee Natural Language Agent v2.0',
                'sections': ['dataset', 'preprocessing', 'performance', 'explainability', 'conclusion']
            },
            'sections': {
                '1_dataset': dataset_section,
                '2_data_preprocessing': preprocessing_section,
                '3_models_performance': performance_section,
                '4_model_explainability': explainability_section,
                '5_conclusion': conclusion_section
            },
            'natural_language_insights': insights,
            'raw_data': {
                'eda': eda_data,
                'ml_models': ml_data
            }
        }
        
        print(f"   ✓ Generated {len(confusion_matrices)} confusion matrices")
        print(f"   ✓ Generated {len(feature_importance_charts)} feature importance charts")
        print(f"   ✓ Generated comparison table with {len(metrics_table)} models")
        
        return output
    
    def _generate_dataset_description(self, eda_data: Dict[str, Any], table_name: str) -> str:
        """Generate dataset description"""
        if not eda_data['summary']:
            return "Dataset information not available."
        
        s = eda_data['summary']
        desc = f"The dataset '{table_name}' contains {s['total_rows']:,} rows and {s['total_columns']} columns. "
        desc += f"The target variable for prediction is '{s['target_column']}'. "
        
        if s.get('duplicate_rows', 0) > 0:
            desc += f"The dataset contains {s['duplicate_rows']:,} duplicate rows ({s['duplicate_percentage']:.2f}%), which were identified during analysis. "
        
        if s.get('memory_usage_mb'):
            desc += f"The total memory footprint of the dataset is approximately {s['memory_usage_mb']:.2f} MB."
        
        return desc
    
    def _generate_preprocessing_description(self, ml_data: List[Dict[str, Any]]) -> str:
        """Generate preprocessing description"""
        if not ml_data or not ml_data[0].get('preprocessing_details'):
            return "Preprocessing details not available."
        
        prep = ml_data[0]['preprocessing_details']
        # Parse preprocessing_details if it's a string
        if isinstance(prep, str):
            prep = json.loads(prep)
        
        desc = "Data preprocessing involved several key steps to prepare the dataset for machine learning. "
        desc += f"We handled {prep.get('missing_values_handled', 0)} missing values through median imputation. "
        desc += f"A total of {len(prep.get('dropped_columns', []))} columns were removed due to high cardinality, excessive missing values, or incompatible data types. "
        desc += f"{len(prep.get('encoded_columns', []))} categorical columns were encoded using Label Encoding, "
        desc += "and all numeric features were scaled using StandardScaler to ensure consistent feature ranges across the dataset."
        
        return desc
    
    def _generate_performance_description(self, ml_data: List[Dict[str, Any]], best_model: Dict[str, Any]) -> str:
        """Generate performance summary description"""
        if not ml_data:
            return "No model performance data available."
        
        desc = f"We evaluated {len(ml_data)} machine learning models on the dataset: "
        desc += ", ".join([m['model_type'] for m in ml_data]) + ". "
        
        if best_model:
            desc += f"The best performing model was {best_model['model_type']} with a test accuracy of {best_model['test_accuracy']:.2%}. "
            desc += f"This model achieved a precision of {best_model['test_precision']:.2%}, "
            desc += f"recall of {best_model['test_recall']:.2%}, and F1-score of {best_model['test_f1_score']:.2%}. "
            
            if best_model['test_roc_auc']:
                desc += f"The ROC-AUC score of {best_model['test_roc_auc']:.4f} indicates excellent discriminative ability."
        
        return desc
    
    def _generate_explainability_description(self, ml_data: List[Dict[str, Any]]) -> str:
        """Generate explainability description"""
        desc = "Understanding which features drive model predictions is crucial for both model validation and business insights. "
        desc += "Feature importance analysis reveals the relative contribution of each input variable to the model's predictions. "
        desc += "We analyzed feature importance across all models using their native importance metrics: "
        desc += "coefficients for Logistic Regression, Gini importance for Decision Trees, and gain-based importance for XGBoost. "
        desc += "The visualizations and rankings below show the top 10 most influential features for each model, "
        desc += "helping stakeholders understand what factors are most predictive of the target variable."
        
        return desc
    
    def _generate_conclusion_summary(self, eda_data: Dict[str, Any], ml_data: List[Dict[str, Any]], best_model: Dict[str, Any]) -> str:
        """Generate conclusion summary"""
        if not best_model:
            return "Insufficient data to generate conclusion."
        
        summary = f"This analysis successfully developed and evaluated multiple machine learning models for predicting '{best_model['target_variable']}'. "
        summary += f"The {best_model['model_type']} emerged as the best performer with {best_model['test_accuracy']:.2%} accuracy. "
        
        # Check for overfitting
        if best_model['train_accuracy'] and (best_model['train_accuracy'] - best_model['test_accuracy'] > 0.1):
            summary += "However, the model shows signs of overfitting, which should be addressed before deployment. "
        else:
            summary += "The model demonstrates good generalization with minimal overfitting. "
        
        summary += "The analysis provides actionable insights into feature importance and model behavior, "
        summary += "enabling informed decision-making for model deployment and further optimization."
        
        return summary
    
    def _generate_key_findings(self, eda_data: Dict[str, Any], ml_data: List[Dict[str, Any]], best_model: Dict[str, Any]) -> List[str]:
        """Generate key findings"""
        findings = []
        
        if best_model:
            findings.append(f"{best_model['model_type']} achieved the highest accuracy of {best_model['test_accuracy']:.2%}")
            
            if best_model['test_roc_auc'] and best_model['test_roc_auc'] > 0.9:
                findings.append(f"Excellent ROC-AUC score of {best_model['test_roc_auc']:.4f} indicates strong predictive power")
            
            # Check performance across models
            if ml_data and len(ml_data) > 1:
                accuracies = [m['test_accuracy'] for m in ml_data]
                accuracy_range = max(accuracies) - min(accuracies)
                if accuracy_range < 0.05:
                    findings.append("All models performed similarly, suggesting the problem has consistent patterns")
                else:
                    findings.append(f"Model performance varied significantly ({accuracy_range:.2%} accuracy range)")
        
        if eda_data['summary']:
            s = eda_data['summary']
            if s['duplicate_percentage'] > 5:
                findings.append(f"Dataset contains {s['duplicate_percentage']:.1f}% duplicate rows requiring attention")
            
            high_null_cols = [c for c in eda_data['columns'] if c['null_percentage'] > 20]
            if high_null_cols:
                findings.append(f"{len(high_null_cols)} columns have >20% missing values")
        
        if ml_data and ml_data[0].get('top_features'):
            top_features = ml_data[0]['top_features']
            # Parse top_features if it's a string
            if isinstance(top_features, str):
                top_features = json.loads(top_features)
            if top_features and len(top_features) > 0:
                top_feature = top_features[0]['feature']
                findings.append(f"'{top_feature}' identified as the most important predictive feature")
        
        return findings
    
    def _generate_recommendations(self, ml_data: List[Dict[str, Any]], best_model: Dict[str, Any]) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        if not best_model:
            return ["Collect more data and retrain models"]
        
        # Performance-based recommendations
        if best_model['test_accuracy'] < 0.7:
            recommendations.append("Consider feature engineering or collecting additional relevant features to improve model performance")
            recommendations.append("Explore advanced models like Neural Networks or Ensemble methods")
        elif best_model['test_accuracy'] > 0.95:
            recommendations.append("Excellent performance achieved - proceed with deployment planning")
            recommendations.append("Implement A/B testing to validate model performance in production")
        
        # Overfitting check
        if best_model.get('train_accuracy') and (best_model['train_accuracy'] - best_model['test_accuracy'] > 0.1):
            recommendations.append("Address overfitting through regularization techniques or by collecting more training data")
            recommendations.append("Consider using cross-validation for more robust model evaluation")
        
        # Class imbalance
        if best_model.get('n_classes') and best_model['n_classes'] > 2:
            recommendations.append("For multi-class problems, investigate per-class performance to identify weak predictions")
        
        # General recommendations
        recommendations.append("Set up model monitoring to track performance degradation over time")
        recommendations.append("Document model assumptions and limitations for stakeholders")
        recommendations.append("Establish a model retraining schedule based on data drift analysis")
        
        return recommendations
    
    def _generate_model_comparison_chart(self, ml_data: List[Dict[str, Any]]) -> str:
        """
        Generate a bar chart comparing model accuracies (returns base64 encoded image)
        """
        if not ml_data:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = [m['model_type'] for m in ml_data]
            accuracies = [m['test_accuracy'] * 100 for m in ml_data]
            
            bars = ax.bar(models, accuracies, color=['#4CAF50', '#2196F3', '#FF9800'])
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 100])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return image_base64
        except Exception as e:
            print(f"   ⚠️  Error generating model comparison chart: {e}")
            return None
    
    def _generate_confusion_matrix_chart(self, confusion_matrix: Any, model_type: str) -> str:
        """
        Generate confusion matrix heatmap (returns base64 encoded image)
        """
        try:
            # Parse confusion matrix if it's a string
            if isinstance(confusion_matrix, str):
                confusion_matrix = json.loads(confusion_matrix)
            
            cm = np.array(confusion_matrix)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, cmap='Blues')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Set ticks
            n_classes = cm.shape[0]
            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            ax.set_xticklabels(np.arange(n_classes))
            ax.set_yticklabels(np.arange(n_classes))
            
            # Add text annotations
            for i in range(n_classes):
                for j in range(n_classes):
                    text = ax.text(j, i, cm[i, j],
                                 ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
            
            ax.set_title(f'Confusion Matrix - {model_type}', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return image_base64
        except Exception as e:
            print(f"   ⚠️  Error generating confusion matrix: {e}")
            return None
    
    def _generate_feature_importance_chart(self, top_features: Any, model_type: str, top_n: int = 10) -> str:
        """
        Generate feature importance bar chart (returns base64 encoded image)
        """
        try:
            # Parse top_features if it's a string
            if isinstance(top_features, str):
                top_features = json.loads(top_features)
            
            features = top_features[:top_n] if isinstance(top_features, list) else []
            
            if not features:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            feature_names = [f['feature'] for f in features]
            importances = [abs(f.get('coefficient', f.get('importance', 0))) for f in features]
            
            bars = ax.barh(feature_names, importances, color='#4CAF50')
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Top {len(features)} Feature Importance - {model_type}', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.4f}',
                       ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return image_base64
        except Exception as e:
            print(f"   ⚠️  Error generating feature importance chart: {e}")
            return None
    
    def _generate_metrics_comparison_table(self, ml_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate a comparison table of all model metrics
        """
        table = []
        for model in ml_data:
            table.append({
                'Model': model['model_type'],
                'Accuracy': f"{model['test_accuracy']:.4f}" if model['test_accuracy'] else 'N/A',
                'Precision': f"{model['test_precision']:.4f}" if model['test_precision'] else 'N/A',
                'Recall': f"{model['test_recall']:.4f}" if model['test_recall'] else 'N/A',
                'F1-Score': f"{model['test_f1_score']:.4f}" if model['test_f1_score'] else 'N/A',
                'ROC-AUC': f"{model['test_roc_auc']:.4f}" if model['test_roc_auc'] else 'N/A'
            })
        return table
    
    def _generate_pdf_report(self, output: Dict[str, Any], json_path: str) -> Optional[str]:
        """
        Generate a comprehensive PDF report from the JSON output
        """
        try:
            # Create PDF path (same as JSON but in pdf folder and .pdf extension)
            json_file = Path(json_path)
            pdf_dir = json_file.parent.parent / "pdf"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = pdf_dir / json_file.name.replace('.json', '.pdf')
            
            # Create PDF document
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                                   leftMargin=0.75*inch, rightMargin=0.75*inch,
                                   topMargin=0.75*inch, bottomMargin=0.75*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            styles.add(ParagraphStyle(name="CenteredTitle", alignment=TA_CENTER,
                                     fontSize=20, spaceAfter=20, leading=24, textColor=colors.HexColor("#1a1a1a")))
            styles.add(ParagraphStyle(name="Justified", alignment=TA_JUSTIFY,
                                     fontSize=11, leading=16))
            styles.add(ParagraphStyle(name="SectionHeader", fontSize=16, leading=18,
                                     spaceBefore=20, spaceAfter=12, textColor=colors.HexColor("#2c3e50")))
            styles.add(ParagraphStyle(name="SubHeader", fontSize=13, leading=16,
                                     spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#34495e")))
            styles.add(ParagraphStyle(name="Subtle", fontSize=9, textColor=colors.grey,
                                     alignment=TA_CENTER))
            
            metadata = output['metadata']
            sections = output['sections']
            
            # ======================
            # 1️⃣ TITLE PAGE
            # ======================
            story.append(Spacer(1, 100))
            story.append(Paragraph("Machine Learning Analysis Report", styles["CenteredTitle"]))
            story.append(Spacer(1, 20))
            story.append(Paragraph(f"<b>Dataset:</b> {metadata['table_name']}", styles["Normal"]))
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>Workflow ID:</b> {metadata['workflow_id']}", styles["Normal"]))
            story.append(Spacer(1, 150))
            
            gen_date = datetime.fromisoformat(metadata['generated_at']).strftime("%B %d, %Y at %I:%M %p")
            story.append(Paragraph(f"Generated by {metadata['generator']}", styles["Subtle"]))
            story.append(Paragraph(f"Date: {gen_date}", styles["Subtle"]))
            story.append(PageBreak())
            
            # ======================
            # 2️⃣ DATASET OVERVIEW
            # ======================
            dataset = sections['1_dataset']
            story.append(Paragraph("1. Dataset Overview", styles["SectionHeader"]))
            story.append(Paragraph(dataset['description'], styles["Justified"]))
            story.append(Spacer(1, 12))
            
            # Dataset statistics table
            stats_data = [["Metric", "Value"]]
            for key, value in dataset['statistics'].items():
                readable_key = key.replace('_', ' ').title()
                stats_data.append([readable_key, str(value)])
            
            stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
            stats_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 10),
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 12))
            
            # Column summary table (limited to fit)
            if dataset['column_summary_table']:
                story.append(Paragraph("Column Summary (Top 10)", styles["SubHeader"]))
                col_data = [["Column", "Type", "Null %", "Unique", "Mean"]]
                for col in dataset['column_summary_table'][:10]:
                    col_data.append([
                        col['Column'][:20],  # Truncate long names
                        col['Type'],
                        col['Null %'],
                        str(col['Unique Values']),
                        str(col['Mean'])[:8]
                    ])
                
                col_table = Table(col_data, colWidths=[1.5*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch])
                col_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2ecc71")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ]))
                story.append(col_table)
            
            story.append(PageBreak())
            
            # ======================
            # 3️⃣ DATA PREPROCESSING
            # ======================
            preprocessing = sections['2_data_preprocessing']
            story.append(Paragraph("2. Data Preprocessing", styles["SectionHeader"]))
            story.append(Paragraph(preprocessing['description'], styles["Justified"]))
            story.append(Spacer(1, 12))
            
            if preprocessing['preprocessing_steps']:
                story.append(Paragraph("Preprocessing Steps:", styles["SubHeader"]))
                for step in preprocessing['preprocessing_steps']:
                    story.append(Paragraph(f"<b>{step['step']}:</b> {step['description']}", styles["Normal"]))
                    story.append(Spacer(1, 6))
            
            story.append(PageBreak())
            
            # ======================
            # 4️⃣ MODEL PERFORMANCE
            # ======================
            performance = sections['3_models_performance']
            story.append(Paragraph("3. Models Performance Summary", styles["SectionHeader"]))
            story.append(Paragraph(performance['description'], styles["Justified"]))
            story.append(Spacer(1, 12))
            
            # Metrics comparison table
            if performance['metrics_comparison_table']:
                metrics_data = [["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
                for row in performance['metrics_comparison_table']:
                    metrics_data.append([
                        row['Model'],
                        row['Accuracy'],
                        row['Precision'],
                        row['Recall'],
                        row['F1-Score'],
                        row['ROC-AUC']
                    ])
                
                metrics_table = Table(metrics_data, colWidths=[1.2*inch]*6)
                metrics_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e74c3c")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ]))
                story.append(metrics_table)
                story.append(Spacer(1, 12))
            
            # Model comparison chart
            if performance['visualizations'].get('model_comparison_chart'):
                story.append(Paragraph("Model Performance Comparison", styles["SubHeader"]))
                img_data = base64.b64decode(performance['visualizations']['model_comparison_chart'])
                img_buffer = io.BytesIO(img_data)
                story.append(RLImage(img_buffer, width=5*inch, height=3*inch))
                story.append(Spacer(1, 12))
            
            story.append(PageBreak())
            
            # Confusion matrices (one per page if multiple)
            if performance['visualizations'].get('confusion_matrices'):
                story.append(Paragraph("Confusion Matrices", styles["SectionHeader"]))
                for model_name, cm_base64 in performance['visualizations']['confusion_matrices'].items():
                    story.append(Paragraph(f"{model_name}", styles["SubHeader"]))
                    img_data = base64.b64decode(cm_base64)
                    img_buffer = io.BytesIO(img_data)
                    story.append(RLImage(img_buffer, width=4.5*inch, height=3.5*inch))
                    story.append(Spacer(1, 12))
                
                story.append(PageBreak())
            
            # ======================
            # 5️⃣ MODEL EXPLAINABILITY
            # ======================
            explainability = sections['4_model_explainability']
            story.append(Paragraph("4. Model Explainability", styles["SectionHeader"]))
            story.append(Paragraph(explainability['description'], styles["Justified"]))
            story.append(Spacer(1, 12))
            
            # Feature importance visualizations
            if explainability['visualizations']:
                for model_name, fi_base64 in explainability['visualizations'].items():
                    story.append(Paragraph(f"Feature Importance - {model_name}", styles["SubHeader"]))
                    img_data = base64.b64decode(fi_base64)
                    img_buffer = io.BytesIO(img_data)
                    story.append(RLImage(img_buffer, width=5*inch, height=3*inch))
                    story.append(Spacer(1, 12))
                
                story.append(PageBreak())
            
            # Feature importance tables
            if explainability['feature_importance']:
                story.append(Paragraph("Feature Importance Rankings", styles["SubHeader"]))
                for model_name, features in explainability['feature_importance'].items():
                    story.append(Paragraph(f"<b>{model_name}:</b>", styles["Normal"]))
                    fi_data = [["Rank", "Feature", "Importance"]]
                    for feat in features[:5]:  # Top 5
                        fi_data.append([str(feat['rank']), feat['feature'][:30], feat['importance']])
                    
                    fi_table = Table(fi_data, colWidths=[0.8*inch, 3*inch, 1.5*inch])
                    fi_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#9b59b6")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (0, -1), "CENTER"),
                        ("ALIGN", (1, 0), (-1, -1), "LEFT"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ]))
                    story.append(fi_table)
                    story.append(Spacer(1, 10))
                
                story.append(PageBreak())
            
            # ======================
            # 6️⃣ CONCLUSION
            # ======================
            conclusion = sections['5_conclusion']
            story.append(Paragraph("5. Conclusion", styles["SectionHeader"]))
            story.append(Paragraph(conclusion['summary'], styles["Justified"]))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph("Key Findings:", styles["SubHeader"]))
            for finding in conclusion['key_findings']:
                story.append(Paragraph(f"• {finding}", styles["Normal"]))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph("Recommendations:", styles["SubHeader"]))
            for rec in conclusion['recommendations'][:5]:  # Top 5
                story.append(Paragraph(f"• {rec}", styles["Normal"]))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph("Next Steps:", styles["SubHeader"]))
            for step in conclusion['next_steps']:
                story.append(Paragraph(f"• {step}", styles["Normal"]))
                story.append(Spacer(1, 4))
            
            story.append(Spacer(1, 40))
            story.append(Paragraph("─" * 80, styles["Subtle"]))
            story.append(Paragraph("End of Report", styles["Subtle"]))
            
            # Build PDF
            doc.build(story)
            
            return str(pdf_path)
            
        except Exception as e:
            print(f"   ⚠️  Error generating PDF: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
