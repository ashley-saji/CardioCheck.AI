"""
Deployment Validation Script
=============================
This script validates that all components needed for Streamlit deployment are working.
"""

import sys
import os

def validate_python_version():
    """Validate Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"⚠️ Python {version.major}.{version.minor}.{version.micro} - May have compatibility issues")
        return True  # Continue anyway

def validate_dependencies():
    """Validate required dependencies."""
    print("\n📦 Checking dependencies...")
    required = {
        'numpy': None,
        'pandas': None,
        'sklearn': 'scikit-learn',
        'streamlit': None,
        'joblib': None,
        'xgboost': None,
        'shap': None,
        'matplotlib': None,
        'seaborn': None,
        'plotly': None,
        'reportlab': None,
    }
    
    all_ok = True
    for module, pip_name in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {pip_name or module}: {version}")
        except ImportError:
            print(f"❌ {pip_name or module}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def validate_models():
    """Validate model loading."""
    print("\n🤖 Checking model loading...")
    try:
        from models.heart_disease_prediction import HeartDiseasePredictor
        print("✅ HeartDiseasePredictor imported")
        
        predictor = HeartDiseasePredictor()
        print("✅ Predictor instantiated")
        
        result = predictor.load_models()
        if result:
            print(f"✅ Models loaded successfully")
            print(f"   Best model: {predictor.best_model_name}")
            print(f"   Total models: {len(predictor.models)}")
            return True
        else:
            print("❌ Model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_data_files():
    """Validate required data files exist."""
    print("\n📁 Checking data files...")
    required_files = [
        'data/heart_disease_uci.csv',
        'saved_models/scaler.joblib',
        'saved_models/random_forest_model.joblib',
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"⚠️ {file_path} - NOT FOUND")
            if 'scaler' in file_path or 'model' in file_path:
                all_ok = False
    
    return all_ok

def validate_config_files():
    """Validate configuration files."""
    print("\n⚙️ Checking configuration files...")
    config_files = {
        'requirements.txt': True,
        'runtime.txt': True,
        'packages.txt': False,
        'streamlit_app.py': True,
    }
    
    all_ok = True
    for file_path, required in config_files.items():
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            if required:
                print(f"❌ {file_path} - REQUIRED")
                all_ok = False
            else:
                print(f"⚠️ {file_path} - Optional, not found")
    
    return all_ok

def main():
    """Run all validations."""
    print("="*60)
    print("🚀 Streamlit Deployment Validation")
    print("="*60)
    
    results = {
        'Python Version': validate_python_version(),
        'Dependencies': validate_dependencies(),
        'Models': validate_models(),
        'Data Files': validate_data_files(),
        'Config Files': validate_config_files(),
    }
    
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Ready for deployment!")
    else:
        print("⚠️ SOME CHECKS FAILED - Review issues above")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
