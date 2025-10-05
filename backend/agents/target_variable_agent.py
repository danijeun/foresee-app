"""
Target Variable Agent for Foresee App
Uses Gemini LLM to suggest optimal target variables for ML analysis
"""
import sys
from pathlib import Path
import pandas as pd
import google.generativeai as genai
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.workflow_manager import WorkflowManager
from services.config import Config
import os


class TargetVariableAgent:
    """
    Agent that analyzes dataset structure and suggests optimal target variables
    using Gemini LLM for intelligent recommendations
    """
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize the Target Variable Agent
        
        Args:
            gemini_api_key: Google Gemini API key (uses env variable if not provided)
        """
        # Configure Gemini API
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment")
        
        genai.configure(api_key=api_key)
        
        # Use the confirmed working model
        model_name = 'models/gemini-2.5-flash-preview-05-20'
        
        try:
            self.model = genai.GenerativeModel(model_name)
            print(f"✅ Using Gemini model: {model_name}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini model {model_name}: {str(e)}. Check your API key.")
        
        # Initialize workflow manager
        self.workflow_manager = None
        
    def analyze_workflow(self, workflow_id: str, table_name: str = None, 
                        sample_size: int = 100) -> Dict[str, Any]:
        """
        Analyze a workflow's data and suggest target variables
        
        Args:
            workflow_id: UUID of the workflow
            table_name: Name of the table to analyze (if None, uses the first table found)
            sample_size: Number of rows to sample for analysis
            
        Returns:
            dict: Analysis results with suggested target variables and reasoning
        """
        try:
            # Initialize workflow manager
            self.workflow_manager = WorkflowManager()
            
            # Get uploader for this workflow
            uploader = self.workflow_manager.get_workflow_uploader(workflow_id=workflow_id)
            schema_name = f"WORKFLOW_{workflow_id}"
            
            # If no table name provided, find the first data table
            if table_name is None:
                table_name = self._find_data_table(uploader, schema_name)
                if not table_name:
                    raise ValueError(f"No data tables found in workflow {workflow_id}")
            
            print(f"🔍 Analyzing table: {table_name}")
            
            # Fetch sample data
            query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
            sample_data = uploader.query(query)
            column_names = [desc[0] for desc in uploader.cursor.description]
            
            # Get data types and statistics
            stats = self._get_column_statistics(uploader, table_name, column_names)
            
            # Create DataFrame for easier analysis
            df = pd.DataFrame(sample_data, columns=column_names)
            
            # Generate data profile
            data_profile = self._create_data_profile(df, stats)
            
            # Use Gemini to suggest target variables
            suggestions = self._get_gemini_suggestions(data_profile)
            
            # Clean up
            uploader.close()
            
            return {
                'success': True,
                'workflow_id': workflow_id,
                'table_name': table_name,
                'schema_name': schema_name,
                'columns': column_names,
                'row_count': stats['total_rows'],
                'data_profile': data_profile,
                'suggestions': suggestions
            }
            
        except Exception as e:
            print(f"❌ Error analyzing workflow: {str(e)}")
            raise
        finally:
            if self.workflow_manager:
                self.workflow_manager.close()
    
    def _find_data_table(self, uploader, schema_name: str) -> str:
        """
        Find the first data table in the workflow schema (excluding metadata tables)
        """
        try:
            uploader.cursor.execute(f"SHOW TABLES IN SCHEMA {schema_name}")
            tables = uploader.cursor.fetchall()
            
            # Filter out metadata tables
            for table in tables:
                table_name = table[1]  # Table name is in column 1
                if table_name != 'WORKFLOW_METADATA':
                    return table_name
            
            return None
        except Exception as e:
            print(f"⚠️ Error finding tables: {e}")
            return None
    
    def _get_column_statistics(self, uploader, table_name: str, 
                               column_names: List[str]) -> Dict[str, Any]:
        """
        Get statistical information about the columns
        """
        stats = {'columns': {}}
        
        # Get total row count
        uploader.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        stats['total_rows'] = uploader.cursor.fetchone()[0]
        
        # Get statistics for each column
        for col in column_names:
            col_stats = {}
            
            # Count nulls
            uploader.cursor.execute(f'SELECT COUNT(*) FROM {table_name} WHERE "{col}" IS NULL')
            col_stats['null_count'] = uploader.cursor.fetchone()[0]
            
            # Count distinct values
            uploader.cursor.execute(f'SELECT COUNT(DISTINCT "{col}") FROM {table_name}')
            col_stats['distinct_count'] = uploader.cursor.fetchone()[0]
            
            stats['columns'][col] = col_stats
        
        return stats
    
    def _create_data_profile(self, df: pd.DataFrame, stats: Dict) -> Dict[str, Any]:
        """
        Create a comprehensive data profile for LLM analysis
        """
        profile = {
            'total_rows': stats['total_rows'],
            'total_columns': len(df.columns),
            'columns': []
        }
        
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': stats['columns'][col]['null_count'],
                'null_percentage': round(stats['columns'][col]['null_count'] / stats['total_rows'] * 100, 2),
                'distinct_count': stats['columns'][col]['distinct_count'],
                'sample_values': df[col].dropna().head(5).tolist()
            }
            
            # Add numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else None
                })
            
            profile['columns'].append(col_info)
        
        return profile
    
    def _get_gemini_suggestions(self, data_profile: Dict) -> Dict[str, Any]:
        """
        Use Gemini LLM to suggest target variables based on data profile
        """
        # Create a detailed prompt for Gemini
        prompt = self._create_analysis_prompt(data_profile)
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Parse the response
            suggestions = self._parse_gemini_response(response.text, data_profile)
            
            return suggestions
            
        except Exception as e:
            print(f"⚠️ Error calling Gemini API: {e}")
            return {
                'error': str(e),
                'fallback_suggestions': self._get_fallback_suggestions(data_profile)
            }
    
    def _create_analysis_prompt(self, data_profile: Dict) -> str:
        """
        Create a detailed prompt for Gemini LLM with emphasis on importance ranking
        """
        columns_info = "\n".join([
            f"- {col['name']}: {col['dtype']}, {col['distinct_count']} unique values, "
            f"{col['null_percentage']}% null, samples: {col['sample_values'][:3]}"
            for col in data_profile['columns']
        ])
        
        prompt = f"""
You are a machine learning expert analyzing a dataset to identify the PRIMARY TARGET VARIABLE (outcome/label) for predictive modeling.

CRITICAL: A target variable is what you want to PREDICT (the outcome/result), NOT what you use to make the prediction (the features/inputs).

Dataset Overview:
- Total rows: {data_profile['total_rows']}
- Total columns: {data_profile['total_columns']}

Columns:
{columns_info}

TASK: Identify the PRIMARY TARGET VARIABLES (outcomes/labels to predict) ranked by importance.

IMPORTANT DISTINCTIONS:
- TARGET variables: Outcomes, diagnoses, results, labels (e.g., "diagnosed_diabetes", "churn", "sales_total")
- FEATURE variables: Inputs used for prediction (e.g., "age", "bmi", "temperature", "date")

For a DIABETES dataset: 
- TARGET: "diagnosed_diabetes" (the outcome we want to predict) ✓
- FEATURES: "age", "bmi", "glucose_level", "family_history" (inputs for prediction) ✗

For a SALES dataset:
- TARGET: "total_revenue", "sales_count" (the outcome we want to predict) ✓
- FEATURES: "date", "product_category", "price" (inputs for prediction) ✗

Consider these factors when ranking:
1. Is this an OUTCOME/RESULT (target) or an INPUT/MEASUREMENT (feature)?
2. Business Value: Which outcome would be most valuable to predict?
3. Data Quality: Low nulls, appropriate cardinality
4. Problem Clarity: Clear regression or classification task

Please provide the TOP 5 TARGET VARIABLES (outcomes to predict), RANKED FROM MOST TO LEAST IMPORTANT.

For each recommendation, provide:
- Importance Score (1-100): How important this target is for ML modeling
- Variable Name: The column name
- Problem Type: regression/binary classification/multi-class classification
- Why Important: Clear explanation of business/analytical value
- Predictability: Assessment of how well this can be predicted
- Suggested Features: Best columns to use as features
- Considerations: Any warnings or challenges

Format your response EXACTLY as follows:

RECOMMENDATION 1 (MOST IMPORTANT):
Variable: [column name]
Importance Score: [1-100]
Problem Type: [regression/binary classification/multi-class classification]
Why Important: [clear explanation of business value and why this is the MOST important target]
Predictability: [HIGH/MEDIUM/LOW - assessment of prediction feasibility]
Suggested Features: [comma-separated list of column names]
Considerations: [any warnings or notes]

RECOMMENDATION 2:
Variable: [column name]
Importance Score: [1-100]
Problem Type: [regression/binary classification/multi-class classification]
Why Important: [explanation]
Predictability: [HIGH/MEDIUM/LOW]
Suggested Features: [comma-separated list]
Considerations: [any warnings]

RECOMMENDATION 3:
[same format]

RECOMMENDATION 4:
[same format]

RECOMMENDATION 5:
[same format]

RANKING RATIONALE:
[Brief explanation of why you ranked these variables in this specific order]
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str, 
                               data_profile: Dict) -> Dict[str, Any]:
        """
        Parse Gemini's response into structured format with importance scoring
        """
        recommendations = []
        ranking_rationale = ""
        
        # Split response into sections
        sections = response_text.split('RECOMMENDATION')
        
        # Parse each recommendation
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            if i > 5:  # Only process top 5
                break
                
            rec = {
                'rank': i,
                'variable': None,
                'importance_score': None,
                'problem_type': None,
                'why_important': None,
                'predictability': None,
                'suggested_features': [],
                'considerations': None
            }
            
            lines = section.strip().split('\n')
            current_field = None
            field_content = []
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check if this is a new field
                if line_stripped.startswith('Variable:'):
                    if current_field and field_content:
                        rec[current_field] = ' '.join(field_content).strip()
                    rec['variable'] = line_stripped.replace('Variable:', '').strip()
                    current_field = None
                    field_content = []
                    
                elif line_stripped.startswith('Importance Score:'):
                    if current_field and field_content:
                        rec[current_field] = ' '.join(field_content).strip()
                    score_text = line_stripped.replace('Importance Score:', '').strip()
                    try:
                        # Extract numeric score
                        import re
                        score_match = re.search(r'\d+', score_text)
                        if score_match:
                            rec['importance_score'] = int(score_match.group())
                    except:
                        rec['importance_score'] = None
                    current_field = None
                    field_content = []
                    
                elif line_stripped.startswith('Problem Type:'):
                    if current_field and field_content:
                        rec[current_field] = ' '.join(field_content).strip()
                    rec['problem_type'] = line_stripped.replace('Problem Type:', '').strip()
                    current_field = None
                    field_content = []
                    
                elif line_stripped.startswith('Why Important:'):
                    if current_field and field_content:
                        rec[current_field] = ' '.join(field_content).strip()
                    content = line_stripped.replace('Why Important:', '').strip()
                    current_field = 'why_important'
                    field_content = [content] if content else []
                    
                elif line_stripped.startswith('Predictability:'):
                    if current_field and field_content:
                        rec[current_field] = ' '.join(field_content).strip()
                    rec['predictability'] = line_stripped.replace('Predictability:', '').strip()
                    current_field = None
                    field_content = []
                    
                elif line_stripped.startswith('Suggested Features:'):
                    if current_field and field_content:
                        rec[current_field] = ' '.join(field_content).strip()
                    features_text = line_stripped.replace('Suggested Features:', '').strip()
                    rec['suggested_features'] = [f.strip() for f in features_text.split(',') if f.strip()]
                    current_field = None
                    field_content = []
                    
                elif line_stripped.startswith('Considerations:'):
                    if current_field and field_content:
                        rec[current_field] = ' '.join(field_content).strip()
                    content = line_stripped.replace('Considerations:', '').strip()
                    current_field = 'considerations'
                    field_content = [content] if content else []
                    
                elif line_stripped and not line_stripped.startswith('RECOMMENDATION') and not line_stripped.startswith('RANKING'):
                    # This is continuation of previous field
                    if current_field:
                        field_content.append(line_stripped)
            
            # Save any remaining field content
            if current_field and field_content:
                rec[current_field] = ' '.join(field_content).strip()
            
            if rec['variable']:  # Only add if we found a variable
                recommendations.append(rec)
        
        # Extract ranking rationale
        if 'RANKING RATIONALE:' in response_text:
            ranking_rationale = response_text.split('RANKING RATIONALE:')[-1].strip()
        
        return {
            'recommendations': recommendations,
            'ranking_rationale': ranking_rationale,
            'raw_response': response_text,
            'total_recommendations': len(recommendations)
        }
    
    def _get_fallback_suggestions(self, data_profile: Dict) -> List[Dict]:
        """
        Simple fallback if LLM fails: Just suggest the LAST few columns
        (Most datasets put target variables at the end by convention)
        """
        suggestions = []
        
        # Get last 5 columns as candidates (reversed so last column is first)
        candidates = list(reversed(data_profile['columns'][-5:]))
        
        for i, col in enumerate(candidates, 1):
            # Skip non-numeric columns for simplicity
            if not ('int' in col['dtype'] or 'float' in col['dtype']):
                continue
            
            # Skip if only 1 unique value (constant)
            if col['distinct_count'] <= 1:
                continue
            
            # Determine problem type based on cardinality
            if 2 <= col['distinct_count'] <= 10:
                problem_type = f'{"binary" if col["distinct_count"] == 2 else "multi-class"} classification'
            else:
                problem_type = 'regression'
            
            suggestions.append({
                'rank': i,
                'variable': col['name'],
                'importance_score': 100 - (i * 5),  # Simple: first=100, second=95, etc.
                'problem_type': problem_type,
                'why_important': f'Located near end of dataset (common target position). {col["distinct_count"]} unique values.',
                'predictability': 'UNKNOWN - LLM analysis recommended',
                'suggested_features': [c['name'] for c in data_profile['columns'] if c['name'] != col['name']][:6],
                'considerations': 'Simple fallback suggestion. Use LLM analysis for accurate recommendations.'
            })
        
        return suggestions[:5]


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    """
    Example usage of the Target Variable Agent
    """
    import json
    
    # Initialize agent
    agent = TargetVariableAgent()
    
    # Example: Analyze a workflow
    # Replace with your actual workflow_id from an upload
    workflow_id = "your_workflow_id_here"
    
    try:
        print("🤖 Starting Target Variable Analysis...")
        print("=" * 70)
        
        # Analyze the workflow
        result = agent.analyze_workflow(
            workflow_id=workflow_id,
            sample_size=100
        )
        
        # Display results
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Dataset: {result['table_name']}")
        print(f"📈 Total Rows: {result['row_count']:,}")
        print(f"📋 Total Columns: {len(result['columns'])}")
        print(f"\nColumns: {', '.join(result['columns'])}")
        
        # Display recommendations
        print("\n🎯 Target Variable Recommendations (Ranked by Importance):")
        print("=" * 70)
        
        for rec in result['suggestions']['recommendations']:
            print(f"\n#{rec['rank']} - {rec['variable']}")
            if rec.get('importance_score'):
                print(f"   ⭐ Importance Score: {rec['importance_score']}/100")
            print(f"   📊 Type: {rec['problem_type']}")
            if rec.get('predictability'):
                print(f"   🎯 Predictability: {rec['predictability']}")
            if rec.get('why_important'):
                print(f"   💡 Why Important: {rec['why_important']}")
            if rec.get('suggested_features'):
                print(f"   🔧 Features: {', '.join(rec['suggested_features'][:5])}")
            if rec.get('considerations'):
                print(f"   ⚠️ Notes: {rec['considerations']}")
        
        print("\n" + "=" * 70)
        print("📋 Ranking Rationale:")
        print("=" * 70)
        if result['suggestions'].get('ranking_rationale'):
            print(result['suggestions']['ranking_rationale'])
        else:
            print("Variables ranked by importance score and data quality.")
        
        # Save full results to JSON
        with open('target_analysis_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n💾 Full results saved to: target_analysis_results.json")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

