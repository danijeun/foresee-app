"""
Test script for Target Variable Agent
Tests the agent on a workflow with comprehensive output
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.workflow_manager import WorkflowManager
from agents.target_variable_agent import TargetVariableAgent


def create_test_workflow():
    """
    Create a test workflow with sample data if needed
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'test', 'hour.csv')
    csv_path = os.path.abspath(csv_path)
    
    if not os.path.exists(csv_path):
        print("❌ Test CSV file not found at:", csv_path)
        return None
    
    print("\n📦 Creating test workflow...")
    manager = WorkflowManager()
    
    try:
        # Create workflow
        workflow = manager.create_workflow(
            workflow_name="Test Target Variable Analysis"
        )
        
        print(f"✅ Workflow created: {workflow['workflow_id']}")
        
        # Upload CSV
        uploader = manager.get_workflow_uploader(schema_name=workflow['schema_name'])
        result = uploader.upload_csv(csv_path, 'bike_sharing_data')
        
        print(f"✅ Data uploaded: {result['rows_loaded']:,} rows")
        
        uploader.close()
        return workflow['workflow_id']
        
    except Exception as e:
        print(f"❌ Error creating test workflow: {e}")
        return None
    finally:
        manager.close()


def list_available_workflows():
    """
    List all available workflows
    """
    print("\n📋 Available Workflows:")
    print("=" * 70)
    
    manager = WorkflowManager()
    try:
        workflows = manager.list_workflows()
        
        if not workflows:
            print("No workflows found. Create one first!")
            return []
        
        for i, wf in enumerate(workflows, 1):
            print(f"{i}. Workflow ID: {wf['workflow_id']}")
            print(f"   Schema: {wf['schema_name']}")
            print(f"   Created: {wf.get('created_on', 'N/A')}")
            print()
        
        return workflows
        
    except Exception as e:
        print(f"❌ Error listing workflows: {e}")
        return []
    finally:
        manager.close()


def test_agent(workflow_id, sample_size=100):
    """
    Test the Target Variable Agent on a workflow
    """
    print("\n" + "=" * 70)
    print("🤖 TESTING TARGET VARIABLE AGENT")
    print("=" * 70)
    print(f"Workflow ID: {workflow_id}")
    print(f"Sample Size: {sample_size}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    try:
        # Initialize agent
        print("\n1️⃣ Initializing Target Variable Agent...")
        agent = TargetVariableAgent()
        print("✅ Agent initialized")
        
        # Analyze workflow
        print(f"\n2️⃣ Analyzing workflow (this may take 10-30 seconds)...")
        result = agent.analyze_workflow(
            workflow_id=workflow_id,
            sample_size=sample_size
        )
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        
        # Display dataset info
        print(f"\n📊 DATASET INFORMATION:")
        print("-" * 70)
        print(f"Table Name:      {result['table_name']}")
        print(f"Schema:          {result['schema_name']}")
        print(f"Total Rows:      {result['row_count']:,}")
        print(f"Total Columns:   {len(result['columns'])}")
        print(f"\nColumn Names:")
        for i, col in enumerate(result['columns'], 1):
            print(f"  {i:2d}. {col}")
        
        # Display data profile summary
        print(f"\n📈 DATA PROFILE SUMMARY:")
        print("-" * 70)
        for col in result['data_profile']['columns'][:10]:  # First 10 columns
            print(f"\n📌 {col['name']}:")
            print(f"   Type:           {col['dtype']}")
            print(f"   Distinct Values: {col['distinct_count']:,}")
            print(f"   Null %:          {col['null_percentage']}%")
            if col.get('mean') is not None:
                print(f"   Mean:            {col['mean']:.2f}")
                print(f"   Min/Max:         {col.get('min', 'N/A')} / {col.get('max', 'N/A')}")
            print(f"   Sample Values:   {col['sample_values'][:3]}")
        
        if len(result['data_profile']['columns']) > 10:
            print(f"\n... and {len(result['data_profile']['columns']) - 10} more columns")
        
        # Display AI recommendations
        print("\n" + "=" * 70)
        print("🎯 TARGET VARIABLE RECOMMENDATIONS (RANKED BY IMPORTANCE)")
        print("=" * 70)
        
        recommendations = result['suggestions']['recommendations']
        
        if not recommendations:
            print("\n⚠️ No recommendations generated.")
            if 'error' in result['suggestions']:
                print(f"Error: {result['suggestions']['error']}")
                print("\nUsing fallback suggestions:")
                recommendations = result['suggestions'].get('fallback_suggestions', [])
        
        # Display each recommendation
        for rec in recommendations:
            print(f"\n{'=' * 70}")
            print(f"RANK #{rec['rank']} - {rec['variable'].upper()}")
            print(f"{'=' * 70}")
            
            if rec.get('importance_score'):
                score = rec['importance_score']
                # Visual score bar
                filled = int(score / 5)
                empty = 20 - filled
                bar = '█' * filled + '░' * empty
                print(f"⭐ Importance Score: {score}/100 [{bar}]")
            
            print(f"📊 Problem Type:    {rec['problem_type']}")
            
            if rec.get('predictability'):
                pred_emoji = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(rec['predictability'], '⚪')
                print(f"🎯 Predictability:  {pred_emoji} {rec['predictability']}")
            
            if rec.get('why_important'):
                print(f"\n💡 Why Important:")
                print(f"   {rec['why_important']}")
            
            if rec.get('suggested_features'):
                print(f"\n🔧 Suggested Features ({len(rec['suggested_features'])} total):")
                features = rec['suggested_features'][:8]  # Show first 8
                for i, feat in enumerate(features, 1):
                    print(f"   {i}. {feat}")
                if len(rec['suggested_features']) > 8:
                    print(f"   ... and {len(rec['suggested_features']) - 8} more")
            
            if rec.get('considerations'):
                print(f"\n⚠️ Considerations:")
                print(f"   {rec['considerations']}")
        
        # Display ranking rationale
        print("\n" + "=" * 70)
        print("📋 RANKING RATIONALE")
        print("=" * 70)
        if result['suggestions'].get('ranking_rationale'):
            print(result['suggestions']['ranking_rationale'])
        else:
            print("Variables ranked by importance score and data quality.")
        
        # Display top recommendation summary
        if recommendations:
            top = recommendations[0]
            print("\n" + "=" * 70)
            print("🏆 RECOMMENDED TARGET VARIABLE")
            print("=" * 70)
            print(f"Variable:       {top['variable']}")
            if top.get('importance_score'):
                print(f"Importance:     {top['importance_score']}/100")
            print(f"Problem Type:   {top['problem_type']}")
            if top.get('predictability'):
                print(f"Predictability: {top['predictability']}")
        
        # Save results to file
        output_file = f"test_results_{workflow_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Full results saved to: {output_file}")
        
        # Next steps
        print("\n" + "=" * 70)
        print("🚀 NEXT STEPS")
        print("=" * 70)
        print("1. Review the recommendations above")
        print("2. Choose a target variable that aligns with your goals")
        print("3. Use the suggested features to build your ML model")
        print("4. Consider the predictability and considerations mentioned")
        print("\nFor best results:")
        print("• Start with the #1 ranked variable (highest importance)")
        print("• Check that predictability is HIGH or MEDIUM")
        print("• Review considerations for any data quality issues")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR TESTING AGENT:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return None


def main():
    """
    Main test function
    """
    print("=" * 70)
    print("🧪 TARGET VARIABLE AGENT - TEST SUITE")
    print("=" * 70)
    
    # Check for Gemini API key
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠️ WARNING: GEMINI_API_KEY not found in environment")
        print("The agent will use fallback suggestions instead of AI analysis")
        print("To get full AI analysis, add GEMINI_API_KEY to your .env file")
        print("Get a free key at: https://makersuite.google.com/app/apikey")
        input("\nPress Enter to continue with fallback mode...")
    
    # List available workflows
    workflows = list_available_workflows()
    
    workflow_id = None
    
    if workflows:
        print("\nOptions:")
        print("1. Use an existing workflow")
        print("2. Create a new test workflow")
        choice = input("\nChoose option (1 or 2): ").strip()
        
        if choice == "1":
            # Use existing workflow
            if len(workflows) == 1:
                workflow_id = workflows[0]['workflow_id']
                print(f"\n✅ Using workflow: {workflow_id}")
            else:
                idx = input(f"\nSelect workflow (1-{len(workflows)}): ").strip()
                try:
                    workflow_id = workflows[int(idx) - 1]['workflow_id']
                    print(f"✅ Using workflow: {workflow_id}")
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    return
        else:
            # Create new workflow
            workflow_id = create_test_workflow()
    else:
        print("\n📦 No existing workflows found. Creating a test workflow...")
        workflow_id = create_test_workflow()
    
    if not workflow_id:
        print("\n❌ Could not get or create a workflow. Exiting.")
        return
    
    # Ask for sample size
    print("\n" + "=" * 70)
    sample_input = input("Enter sample size (default 100, press Enter to use default): ").strip()
    sample_size = int(sample_input) if sample_input else 100
    
    # Run the test
    result = test_agent(workflow_id, sample_size)
    
    if result:
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
