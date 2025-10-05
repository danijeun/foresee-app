"""
Decision Tree Agent for Foresee App
Trains decision tree models and outputs key insights to Snowflake
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.workflow_manager import WorkflowManager
from services.config import Config


class DecisionTreeAgent:
    """
    Agent that trains decision tree models and generates insights
    """
    
    def __init__(self):
        """
        Initialize the Decision Tree Agent
        """
        self.workflow_manager = None
        self.label_encoders = {}
        self.model = None
        
    def train_and_evaluate(
        self,
        workflow_id: str,
        table_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ) -> Dict[str, Any]:
        """
        Train decision tree model and save insights to Snowflake
        
        Args:
            workflow_id: UUID of the workflow
            table_name: Name of the table to train on
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
            max_depth: Maximum depth of the tree (None = unlimited)
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required at leaf node
            
        Returns:
            dict: Training results and insights
        """
        try:
            print("=" * 70)
            print("🌳 DECISION TREE AGENT")
            print("=" * 70)
            
            # Initialize workflow manager
            self.workflow_manager = WorkflowManager()
            schema_name = f"WORKFLOW_{workflow_id}"
            uploader = self.workflow_manager.get_workflow_uploader(schema_name=schema_name)
            
            # Step 1: Get target variable from workflow_eda_summary
            print(f"\n📋 Step 1: Retrieving target variable...")
            target_info = self._get_target_variable(uploader, schema_name, table_name)
            
            if not target_info['target_column']:
                raise ValueError("No target variable selected. Please select a target variable first.")
            
            print(f"   ✓ Target Variable: {target_info['target_column']}")
            print(f"   ✓ Problem Type: {target_info['problem_type']}")
            
            # Validate problem type
            if target_info['problem_type'] and 'regression' in target_info['problem_type'].lower():
                print(f"   ⚠️  WARNING: Target appears to be for regression, but Decision Tree is for classification.")
                print(f"   ℹ️  Will attempt to convert to classification by binning values if needed.")
            
            # Step 2: Fetch dataset
            print(f"\n📊 Step 2: Fetching dataset from Snowflake...")
            df = self._fetch_dataset(uploader, table_name)
            print(f"   ✓ Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
            
            # Step 3: Preprocess data
            print(f"\n🔧 Step 3: Preprocessing data...")
            X, y, feature_names, preprocessing_report = self._preprocess_data(
                df, 
                target_info['target_column']
            )
            print(f"   ✓ Features prepared: {len(feature_names)} features")
            print(f"   ✓ Target classes: {np.unique(y)}")
            
            # Step 4: Split data
            print(f"\n✂️  Step 4: Splitting data (train {int((1-test_size)*100)}% / test {int(test_size*100)}%)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print(f"   ✓ Training set: {len(X_train):,} samples")
            print(f"   ✓ Test set: {len(X_test):,} samples")
            
            # Step 5: Train model
            print(f"\n🎓 Step 5: Training Decision Tree model...")
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
            self.model.fit(X_train, y_train)
            print(f"   ✓ Model trained successfully!")
            print(f"   ✓ Tree depth: {self.model.get_depth()}")
            print(f"   ✓ Number of leaves: {self.model.get_n_leaves()}")
            
            # Step 6: Evaluate model
            print(f"\n📈 Step 6: Evaluating model performance...")
            evaluation = self._evaluate_model(
                X_train, X_test, y_train, y_test, feature_names
            )
            
            print(f"   ✓ Test Accuracy: {evaluation['test_accuracy']:.2%}")
            print(f"   ✓ Test Precision: {evaluation['test_precision']:.2%}")
            print(f"   ✓ Test Recall: {evaluation['test_recall']:.2%}")
            print(f"   ✓ Test F1-Score: {evaluation['test_f1']:.2%}")
            
            # Step 7: Generate insights
            print(f"\n💡 Step 7: Generating insights...")
            insights = self._generate_insights(
                evaluation,
                preprocessing_report,
                target_info,
                feature_names,
                max_depth,
                min_samples_split,
                min_samples_leaf
            )
            
            # Step 8: Create summary table if not exists
            print(f"\n🗄️  Step 8: Creating DECISION_TREE_SUMMARY table...")
            self._create_summary_table(uploader, schema_name)
            
            # Step 9: Save insights to Snowflake
            print(f"\n💾 Step 9: Saving insights to Snowflake...")
            self._save_insights_to_snowflake(
                uploader,
                schema_name,
                workflow_id,
                table_name,
                target_info['target_column'],
                insights
            )
            
            # Clean up
            uploader.close()
            self.workflow_manager.close()
            
            print("\n" + "=" * 70)
            print("✅ DECISION TREE ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            
            return {
                'success': True,
                'workflow_id': workflow_id,
                'table_name': table_name,
                'target_variable': target_info['target_column'],
                'insights': insights
            }
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            raise
        finally:
            if self.workflow_manager:
                self.workflow_manager.close()
    
    def _get_target_variable(
        self,
        uploader,
        schema_name: str,
        table_name: str
    ) -> Dict[str, Any]:
        """
        Retrieve target variable from workflow_eda_summary table
        """
        query = f"""
            SELECT 
                target_column,
                analysis_type,
                total_rows,
                total_columns
            FROM {schema_name}.workflow_eda_summary
            WHERE table_name = '{table_name}'
            ORDER BY analysis_id DESC
            LIMIT 1
        """
        
        uploader.cursor.execute(query)
        result = uploader.cursor.fetchone()
        
        if not result:
            raise ValueError(f"No EDA summary found for table {table_name}")
        
        return {
            'target_column': result[0],
            'problem_type': result[1] or 'classification',
            'total_rows': result[2],
            'total_columns': result[3]
        }
    
    def _fetch_dataset(self, uploader, table_name: str) -> pd.DataFrame:
        """
        Fetch dataset from Snowflake
        """
        query = f"SELECT * FROM {table_name}"
        uploader.cursor.execute(query)
        
        # Fetch data
        rows = uploader.cursor.fetchall()
        columns = [desc[0] for desc in uploader.cursor.description]
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        return df
    
    def _preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> tuple:
        """
        Preprocess data for decision tree
        
        Returns:
            tuple: (X, y, feature_names, preprocessing_report)
        """
        preprocessing_report = {
            'original_rows': len(df),
            'original_features': len(df.columns) - 1,
            'dropped_columns': [],
            'encoded_columns': [],
            'missing_values_handled': 0,
            'rows_dropped': 0,
            'target_conversion': None
        }
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
        
        # Handle missing values in target
        missing_target = y.isna().sum()
        if missing_target > 0:
            print(f"   ⚠️  Dropping {missing_target} rows with missing target values")
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            preprocessing_report['rows_dropped'] = missing_target
        
        # Drop columns with too many missing values (>50%)
        missing_threshold = 0.5
        for col in X.columns:
            missing_pct = X[col].isna().sum() / len(X)
            if missing_pct > missing_threshold:
                X = X.drop(columns=[col])
                preprocessing_report['dropped_columns'].append(
                    f"{col} ({missing_pct:.1%} missing)"
                )
        
        # Drop date/time columns (not suitable for decision tree)
        datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            X = X.drop(columns=datetime_cols)
            preprocessing_report['dropped_columns'].extend([f"{col} (datetime)" for col in datetime_cols])
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            # Check if column still exists (might have been dropped earlier)
            if col not in X.columns:
                continue
                
            # Get unique count before dropping
            unique_count = X[col].nunique()
            
            # Drop if too many unique values (high cardinality)
            if unique_count > 100:  # Decision trees can handle more categories than logistic regression
                X = X.drop(columns=[col])
                preprocessing_report['dropped_columns'].append(
                    f"{col} (high cardinality: {unique_count} unique values)"
                )
            else:
                # Encode categorical
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values
                X[col] = X[col].fillna('MISSING')
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                preprocessing_report['encoded_columns'].append(col)
        
        # Handle missing values in numeric columns (impute with median)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            missing_count = X[col].isna().sum()
            if missing_count > 0:
                median_value = X[col].median()
                X[col] = X[col].fillna(median_value)
                preprocessing_report['missing_values_handled'] += missing_count
        
        # Convert to numpy arrays
        feature_names = X.columns.tolist()
        X_array = X.values
        
        # Encode target variable
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            # Categorical target - use label encoding
            self.label_encoders['target'] = LabelEncoder()
            y_array = self.label_encoders['target'].fit_transform(y.astype(str))
            preprocessing_report['target_conversion'] = 'categorical_encoded'
        else:
            # Numeric target - check if it needs conversion to classification
            unique_values = y.nunique()
            
            if unique_values <= 10:
                # Few unique values - treat as categorical
                print(f"   ℹ️  Target has {unique_values} unique values - treating as classification")
                y_array = y.values
                preprocessing_report['target_conversion'] = f'numeric_categorical ({unique_values} classes)'
            elif unique_values > 50:
                # Many unique values (continuous) - bin into classes
                print(f"   ℹ️  Target has {unique_values} unique values - converting to 3 classes using quantiles")
                y_binned = pd.qcut(y, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                self.label_encoders['target'] = LabelEncoder()
                y_array = self.label_encoders['target'].fit_transform(y_binned.astype(str))
                print(f"   ✓ Created classes: Low, Medium, High")
                preprocessing_report['target_conversion'] = 'continuous_to_3_classes (Low, Medium, High)'
            else:
                # Moderate number of unique values - treat as is
                y_array = y.values
                preprocessing_report['target_conversion'] = f'numeric_as_is ({unique_values} classes)'
        
        preprocessing_report['final_rows'] = len(X_array)
        preprocessing_report['final_features'] = len(feature_names)
        
        return X_array, y_array, feature_names, preprocessing_report
    
    def _evaluate_model(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model
        """
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Get prediction probabilities for ROC AUC
        y_test_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        n_classes = len(np.unique(y_train))
        
        # Multi-class handling for metrics
        average_method = 'binary' if n_classes == 2 else 'weighted'
        
        evaluation = {
            # Training metrics
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, average=average_method, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, average=average_method, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, average=average_method, zero_division=0),
            
            # Test metrics
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, average=average_method, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, average=average_method, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, average=average_method, zero_division=0),
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
            
            # Feature importance
            'feature_importance': self._get_feature_importance(feature_names),
            
            # Tree structure info
            'tree_depth': int(self.model.get_depth()),
            'n_leaves': int(self.model.get_n_leaves()),
            
            # Number of classes
            'n_classes': n_classes,
            'class_labels': np.unique(y_train).tolist()
        }
        
        # ROC AUC (for binary or multi-class)
        try:
            if n_classes == 2:
                evaluation['test_roc_auc'] = roc_auc_score(y_test, y_test_proba[:, 1])
            else:
                evaluation['test_roc_auc'] = roc_auc_score(
                    y_test, y_test_proba, multi_class='ovr', average='weighted'
                )
        except:
            evaluation['test_roc_auc'] = None
        
        return evaluation
    
    def _get_feature_importance(self, feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Extract feature importance from decision tree
        """
        importances = self.model.feature_importances_
        
        # Create feature importance list
        importance = []
        for feature, imp in zip(feature_names, importances):
            importance.append({
                'feature': feature,
                'importance': float(imp)
            })
        
        # Sort by importance
        importance = sorted(importance, key=lambda x: x['importance'], reverse=True)
        
        return importance
    
    def _generate_insights(
        self,
        evaluation: Dict[str, Any],
        preprocessing_report: Dict[str, Any],
        target_info: Dict[str, Any],
        feature_names: List[str],
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights from the model
        """
        # Performance summary
        performance_summary = f"""
Model achieved {evaluation['test_accuracy']:.1%} accuracy on test data.
Precision: {evaluation['test_precision']:.1%}, Recall: {evaluation['test_recall']:.1%}, F1-Score: {evaluation['test_f1']:.1%}.
Tree has {evaluation['tree_depth']} levels and {evaluation['n_leaves']} leaf nodes.
"""
        
        # Check for overfitting
        accuracy_diff = evaluation['train_accuracy'] - evaluation['test_accuracy']
        if accuracy_diff > 0.1:
            performance_summary += f"\n⚠️  Potential overfitting detected (train-test accuracy gap: {accuracy_diff:.1%})."
        
        # Top features
        top_features = evaluation['feature_importance'][:5]
        top_features_summary = "Top 5 most important features:\n" + "\n".join([
            f"  {i+1}. {f['feature']}: {f['importance']:.4f}"
            for i, f in enumerate(top_features)
        ])
        
        # Data quality insights
        data_quality_summary = f"""
Dataset: {preprocessing_report['original_rows']:,} rows, {preprocessing_report['final_features']} features used.
Preprocessing: {len(preprocessing_report['dropped_columns'])} columns dropped, {preprocessing_report['missing_values_handled']} missing values handled.
Target conversion: {preprocessing_report.get('target_conversion', 'none')}.
"""
        
        # Model configuration
        model_config_summary = f"""
Max depth: {max_depth if max_depth else 'unlimited'}, Min samples split: {min_samples_split}, Min samples leaf: {min_samples_leaf}.
Actual tree depth: {evaluation['tree_depth']}, Number of leaves: {evaluation['n_leaves']}.
"""
        
        # Recommendations
        recommendations = []
        
        if evaluation['test_accuracy'] < 0.7:
            recommendations.append("Consider feature engineering or trying ensemble models (Random Forest, XGBoost).")
        
        if accuracy_diff > 0.1:
            recommendations.append("Model may be overfitting. Try pruning the tree with max_depth or min_samples_split.")
        
        if evaluation['tree_depth'] > 20:
            recommendations.append("Tree is very deep. Consider limiting max_depth to prevent overfitting.")
        
        if preprocessing_report['missing_values_handled'] > preprocessing_report['original_rows'] * 0.1:
            recommendations.append("High number of missing values. Investigate data quality and collection processes.")
        
        if len(preprocessing_report['dropped_columns']) > preprocessing_report['original_features'] * 0.3:
            recommendations.append("Many features were dropped. Review data preprocessing strategy.")
        
        if not recommendations:
            recommendations.append("Model performance is good. Consider deploying for predictions.")
        
        insights = {
            'model_type': 'Decision Tree',
            'target_variable': target_info['target_column'],
            'problem_type': target_info['problem_type'],
            
            # Performance metrics
            'test_accuracy': evaluation['test_accuracy'],
            'test_precision': evaluation['test_precision'],
            'test_recall': evaluation['test_recall'],
            'test_f1': evaluation['test_f1'],
            'test_roc_auc': evaluation.get('test_roc_auc'),
            
            'train_accuracy': evaluation['train_accuracy'],
            
            # Tree structure
            'tree_depth': evaluation['tree_depth'],
            'n_leaves': evaluation['n_leaves'],
            
            # Model parameters
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            
            # Data info
            'total_samples': preprocessing_report['final_rows'],
            'total_features': preprocessing_report['final_features'],
            'n_classes': evaluation['n_classes'],
            
            # Key insights
            'performance_summary': performance_summary.strip(),
            'top_features': top_features,
            'top_features_summary': top_features_summary.strip(),
            'data_quality_summary': data_quality_summary.strip(),
            'model_config_summary': model_config_summary.strip(),
            
            # Confusion matrix
            'confusion_matrix': evaluation['confusion_matrix'],
            
            # Preprocessing info
            'dropped_columns': preprocessing_report['dropped_columns'],
            'encoded_columns': preprocessing_report['encoded_columns'],
            'missing_values_handled': preprocessing_report['missing_values_handled'],
            
            # Recommendations
            'recommendations': recommendations,
            
            # Full feature importance
            'feature_importance': evaluation['feature_importance']
        }
        
        return insights
    
    def _create_summary_table(self, uploader, schema_name: str):
        """
        Create DECISION_TREE_SUMMARY table if it doesn't exist
        """
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.DECISION_TREE_SUMMARY (
            analysis_id VARCHAR(100) PRIMARY KEY,
            workflow_id VARCHAR(100) NOT NULL,
            table_name VARCHAR(255) NOT NULL,
            target_variable VARCHAR(255) NOT NULL,
            
            -- Model info
            model_type VARCHAR(50),
            problem_type VARCHAR(50),
            
            -- Performance metrics
            test_accuracy FLOAT,
            test_precision FLOAT,
            test_recall FLOAT,
            test_f1_score FLOAT,
            test_roc_auc FLOAT,
            train_accuracy FLOAT,
            
            -- Tree structure
            tree_depth INT,
            n_leaves INT,
            
            -- Model parameters
            max_depth INT,
            min_samples_split INT,
            min_samples_leaf INT,
            
            -- Data info
            total_samples INT,
            total_features INT,
            n_classes INT,
            
            -- Text summaries
            performance_summary VARCHAR(5000),
            top_features_summary VARCHAR(5000),
            data_quality_summary VARCHAR(5000),
            model_config_summary VARCHAR(5000),
            
            -- JSON details
            confusion_matrix VARIANT,
            top_features VARIANT,
            feature_importance VARIANT,
            preprocessing_details VARIANT,
            recommendations VARIANT,
            
            -- Metadata
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        uploader.cursor.execute(create_table_sql)
        uploader.conn.commit()
        print(f"   ✓ Table {schema_name}.DECISION_TREE_SUMMARY ready")
    
    def _save_insights_to_snowflake(
        self,
        uploader,
        schema_name: str,
        workflow_id: str,
        table_name: str,
        target_variable: str,
        insights: Dict[str, Any]
    ):
        """
        Save insights to DECISION_TREE_SUMMARY table
        """
        analysis_id = f"DT_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert all numpy types to native Python types
        insights_clean = convert_numpy_types(insights)
        
        # Prepare JSON strings for VARIANT columns - escape single quotes for SQL
        confusion_matrix_json = json.dumps(insights_clean['confusion_matrix']).replace("'", "''")
        top_features_json = json.dumps(insights_clean['top_features']).replace("'", "''")
        feature_importance_json = json.dumps(insights_clean['feature_importance']).replace("'", "''")
        preprocessing_json = json.dumps({
            'dropped_columns': insights_clean['dropped_columns'],
            'encoded_columns': insights_clean['encoded_columns'],
            'missing_values_handled': insights_clean['missing_values_handled']
        }).replace("'", "''")
        recommendations_json = json.dumps(insights_clean['recommendations']).replace("'", "''")
        
        # Use SELECT with PARSE_JSON instead of VALUES clause (Snowflake doesn't support PARSE_JSON in VALUES)
        insert_sql = f"""
        INSERT INTO {schema_name}.DECISION_TREE_SUMMARY (
            analysis_id,
            workflow_id,
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
            feature_importance,
            preprocessing_details,
            recommendations
        )
        SELECT 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            PARSE_JSON('{confusion_matrix_json}'),
            PARSE_JSON('{top_features_json}'),
            PARSE_JSON('{feature_importance_json}'),
            PARSE_JSON('{preprocessing_json}'),
            PARSE_JSON('{recommendations_json}')
        """
        
        uploader.cursor.execute(insert_sql, (
            analysis_id,
            workflow_id,
            table_name,
            target_variable,
            insights_clean['model_type'],
            insights_clean['problem_type'],
            insights_clean['test_accuracy'],
            insights_clean['test_precision'],
            insights_clean['test_recall'],
            insights_clean['test_f1'],
            insights_clean['test_roc_auc'],
            insights_clean['train_accuracy'],
            insights_clean['tree_depth'],
            insights_clean['n_leaves'],
            insights_clean['max_depth'],
            insights_clean['min_samples_split'],
            insights_clean['min_samples_leaf'],
            insights_clean['total_samples'],
            insights_clean['total_features'],
            insights_clean['n_classes'],
            insights_clean['performance_summary'],
            insights_clean['top_features_summary'],
            insights_clean['data_quality_summary'],
            insights_clean['model_config_summary']
        ))
        
        uploader.conn.commit()
        print(f"   ✓ Insights saved with analysis_id: {analysis_id}")
        print(f"   ✓ Location: {schema_name}.DECISION_TREE_SUMMARY")


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    """
    Example usage of the Decision Tree Agent
    """
    import traceback
    
    # Initialize agent
    agent = DecisionTreeAgent()
    
    # Replace with your actual workflow_id and table_name
    workflow_id = "your_workflow_id_here"
    table_name = "your_table_name"
    
    try:
        print("🌳 Starting Decision Tree Analysis...")
        
        # Train and evaluate model
        result = agent.train_and_evaluate(
            workflow_id=workflow_id,
            table_name=table_name,
            test_size=0.2,
            random_state=42,
            max_depth=10  # Limit tree depth to prevent overfitting
        )
        
        if result['success']:
            print("\n📊 Model Insights:")
            print(f"   Accuracy: {result['insights']['test_accuracy']:.2%}")
            print(f"   Tree Depth: {result['insights']['tree_depth']}")
            print(f"   Top Feature: {result['insights']['top_features'][0]['feature']}")
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(result['insights']['recommendations'], 1):
                print(f"   {i}. {rec}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
