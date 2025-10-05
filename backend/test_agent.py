"""
Test script for Target Variable Agent
Tests the agent on current workflows in Snowflake
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.target_variable_agent import TargetVariableAgent
from services.workflow_manager import WorkflowManager


def list_workflows():
    """List all available workflows"""
    print("\n" + "=" * 70)
    print("📋 AVAILABLE WORKFLOWS")
    print("=" * 70)
    
    manager = WorkflowManager()
    try:
        workflows = manager.list_workflows()
        
        if not workflows:
            print("\n⚠️ No workflows found!")
            print("\nTo create a workflow, upload a CSV file:")
            print("  - Via API: POST /api/upload")
            print("  - Via Python: Use services.workflow_manager\n")
            return None
        
        for i, wf in enumerate(workflows, 1):
            print(f"\n{i}. Workflow ID: {wf['workflow_id']}")
            print(f"   Schema: {wf['schema_name']}")
            if wf.get('created_on'):
                print(f"   Created: {wf['created_on']}")
        
        return workflows
        
    finally:
        manager.close()


def test_agent(workflow_id):
    """
    Test the Target Variable Agent on a workflow
    
    Args:
        workflow_id: UUID of the workflow to analyze
    """
    print("\n" + "=" * 70)
    print("🤖 TARGET VARIABLE AGENT - TEST")
    print("=" * 70)
    print(f"Workflow ID: {workflow_id}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Initialize agent
        print("\n📍 Step 1: Initializing agent...")
        agent = TargetVariableAgent()
        print("✅ Agent initialized")
        
        # Analyze workflow
        print("\n📍 Step 2: Analyzing workflow...")
        print("   (This may take 10-30 seconds depending on API response)")
        
        result = agent.analyze_workflow(
            workflow_id=workflow_id,
            sample_size=100
        )
        
        print("✅ Analysis complete!")
        
        # Display results
        print("\n" + "=" * 70)
        print("📊 DATASET SUMMARY")
        print("=" * 70)
        print(f"Table:       {result['table_name']}")
        print(f"Schema:      {result['schema_name']}")
        print(f"Total Rows:  {result['row_count']:,}")
        print(f"Columns:     {len(result['columns'])}")
        
        print("\n📋 Column Names:")
        for i, col in enumerate(result['columns'], 1):
            print(f"  {i:2d}. {col}")
        
        # Display recommendations
        print("\n" + "=" * 70)
        print("🎯 TARGET VARIABLE RECOMMENDATIONS")
        print("=" * 70)
        print("(Ranked by importance from most to least important)\n")
        
        # Handle both regular recommendations and fallback suggestions
        recommendations = result['suggestions'].get('recommendations')
        
        if not recommendations:
            # Check if we have fallback suggestions
            if 'fallback_suggestions' in result['suggestions']:
                recommendations = result['suggestions']['fallback_suggestions']
                print("⚠️ Using fallback suggestions (Gemini API unavailable)\n")
            elif 'error' in result['suggestions']:
                print(f"⚠️ Error: {result['suggestions']['error']}")
                return result
            else:
                print("⚠️ No recommendations generated")
                return result
        
        # Show each recommendation
        for rec in recommendations:
            print("─" * 70)
            print(f"RANK #{rec['rank']}: {rec['variable'].upper()}")
            print("─" * 70)
            
            # Importance score with visual bar
            if rec.get('importance_score'):
                score = rec['importance_score']
                filled = int(score / 5)
                bar = '█' * filled + '░' * (20 - filled)
                print(f"⭐ Importance:     {score}/100 [{bar}]")
            
            # Problem type
            print(f"📊 Problem Type:   {rec['problem_type']}")
            
            # Predictability with emoji
            if rec.get('predictability'):
                emoji = {
                    'HIGH': '🟢',
                    'MEDIUM': '🟡',
                    'LOW': '🔴'
                }.get(rec['predictability'], '⚪')
                print(f"🎯 Predictability: {emoji} {rec['predictability']}")
            
            # Why important
            if rec.get('why_important'):
                print(f"\n💡 Why Important:")
                print(f"   {rec['why_important']}")
            
            # Suggested features
            if rec.get('suggested_features'):
                features = rec['suggested_features'][:5]
                print(f"\n🔧 Suggested Features ({len(rec['suggested_features'])} total):")
                for feat in features:
                    print(f"   • {feat}")
                if len(rec['suggested_features']) > 5:
                    print(f"   ... and {len(rec['suggested_features']) - 5} more")
            
            # Considerations
            if rec.get('considerations'):
                print(f"\n⚠️  Considerations:")
                print(f"   {rec['considerations']}")
            
            print()
        
        # Ranking rationale
        print("=" * 70)
        print("📋 RANKING RATIONALE")
        print("=" * 70)
        rationale = result['suggestions'].get('ranking_rationale', 
                                              'Variables ranked by importance score.')
        print(rationale)
        
        # Top recommendation
        if recommendations:
            top = recommendations[0]
            print("\n" + "=" * 70)
            print("🏆 RECOMMENDED ACTION")
            print("=" * 70)
            print(f"Start with: {top['variable']}")
            if top.get('importance_score'):
                print(f"Score: {top['importance_score']}/100")
            print(f"Type: {top['problem_type']}")
            if top.get('predictability'):
                print(f"Predictability: {top['predictability']}")
        
        # Create test_results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"analysis_{workflow_id[:8]}_{timestamp}.json"
        output_path = os.path.join(results_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full results saved to: {output_path}")
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        
        import traceback
        print("\nFull Traceback:")
        traceback.print_exc()
        
        return None


def main():
    """Main test function"""
    print("=" * 70)
    print("🧪 TARGET VARIABLE AGENT - TEST SUITE")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠️  WARNING: GEMINI_API_KEY not found in environment")
        print("   The agent will use fallback suggestions instead of AI")
        print("   To use AI: Add GEMINI_API_KEY to your .env file")
        print("   Get free key: https://makersuite.google.com/app/apikey\n")
        
        choice = input("Continue with fallback mode? (y/n): ").lower()
        if choice != 'y':
            print("Test cancelled")
            return
    
    # List workflows
    workflows = list_workflows()
    
    if not workflows:
        return
    
    # Select workflow
    print("\n" + "=" * 70)
    if len(workflows) == 1:
        workflow_id = workflows[0]['workflow_id']
        print(f"Using workflow: {workflow_id}")
    else:
        try:
            choice = input(f"Select workflow (1-{len(workflows)}): ").strip()
            idx = int(choice) - 1
            workflow_id = workflows[idx]['workflow_id']
            print(f"Selected: {workflow_id}")
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return
    
    # Run test
    print("\n" + "=" * 70)
    print("Starting analysis with sample size of 100 rows...")
    result = test_agent(workflow_id)
    
    if result:
        print("\n✨ Test completed! Review the recommendations above.")
    else:
        print("\n❌ Test failed. Check the error messages above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
