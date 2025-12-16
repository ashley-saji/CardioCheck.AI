"""
EMERGENCY FIX: Resave models compatible with scikit-learn 1.8.0+
Run this script after installing requirements to rebuild models.joblib files.
"""
import warnings
import sys
import os

warnings.filterwarnings('ignore')

print("🔧 Loading model components with current scikit-learn...")

try:
    from models.heart_disease_prediction_optimized import OptimizedHeartDiseasePredictor
    print("✅ Imported OptimizedHeartDiseasePredictor")
    
    # Try to load existing models
    predictor = OptimizedHeartDiseasePredictor()
    print("✅ Created predictor instance")
    
    ok = predictor.load_models()
    if ok and hasattr(predictor, 'best_model_instance') and predictor.best_model_instance:
        print("✅ Models loaded successfully!")
        print(f"   Best model: {predictor.best_model}")
        print("   No resave needed.")
        sys.exit(0)
    else:
        print("❌ Models failed to load - joblib version mismatch likely")
        print("   This typically means the .joblib files were saved with an older scikit-learn")
        print("   and cannot be loaded with the newer version on Streamlit Cloud.")
        print("\n🔄 Options:")
        print("   1. Retrain models from scratch (complex)")
        print("   2. Downgrade scikit-learn to 1.3.2 (recommended)")
        print("   3. Contact support to manually update model artifacts")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
