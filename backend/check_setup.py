"""
Setup verification script for Foresee App
Checks if all dependencies and configurations are correct
"""
import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        return False

def check_packages():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'flask_cors',
        'pandas',
        'snowflake.connector',
        'dotenv'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            all_installed = False
    
    return all_installed

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        print("❌ .env file not found")
        print("   Create it from .env.example and add your Snowflake credentials")
        return False
    
    print("✅ .env file exists")
    
    # Check if it has content
    with open(env_path, 'r') as f:
        content = f.read()
        required_vars = [
            'SNOWFLAKE_ACCOUNT',
            'SNOWFLAKE_USER',
            'SNOWFLAKE_PASSWORD',
            'INGESTION_WAREHOUSE'
        ]
        
        missing = []
        for var in required_vars:
            if var not in content or f"{var}=your_" in content:
                missing.append(var)
        
        if missing:
            print(f"⚠️  Missing or unconfigured variables: {', '.join(missing)}")
            return False
        else:
            print("✅ All required environment variables are configured")
    
    return True

def check_snowflake_connection():
    """Try to connect to Snowflake"""
    try:
        from services.config import Config
        import snowflake.connector
        
        print("\n🔌 Testing Snowflake connection...")
        conn = snowflake.connector.connect(
            user=Config.SNOWFLAKE_USER,
            password=Config.SNOWFLAKE_PASSWORD,
            account=Config.SNOWFLAKE_ACCOUNT,
            warehouse=Config.INGESTION_WAREHOUSE
        )
        conn.close()
        print("✅ Snowflake connection successful")
        return True
    except Exception as e:
        print(f"❌ Snowflake connection failed: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("🔍 Foresee App - Setup Verification")
    print("=" * 70)
    print()
    
    print("📦 Checking Python & Packages:")
    print("-" * 70)
    python_ok = check_python_version()
    packages_ok = check_packages()
    
    print()
    print("🔧 Checking Configuration:")
    print("-" * 70)
    env_ok = check_env_file()
    
    print()
    all_ok = python_ok and packages_ok and env_ok
    
    if all_ok:
        snowflake_ok = check_snowflake_connection()
        all_ok = all_ok and snowflake_ok
    
    print()
    print("=" * 70)
    if all_ok:
        print("✅ All checks passed! You're ready to run the demo.")
        print()
        print("Run the demo with:")
        print("  Terminal 1: python backend/app.py")
        print("  Terminal 2: cd frontend && npm run dev")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print()
        print("Next steps:")
        if not packages_ok:
            print("  1. Install packages: pip install -r backend/requirements.txt")
        if not env_ok:
            print("  2. Create and configure .env file")
        if not all_ok:
            print("  3. Run this check again: python backend/check_setup.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
