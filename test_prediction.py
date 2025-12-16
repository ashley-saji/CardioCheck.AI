#!/usr/bin/env python3
"""Test script to verify model loading and predictions work correctly."""

import sys
import pandas as pd
from models.heart_disease_prediction import HeartDiseasePredictor

def test_model_loading():
    """Test that models load correctly."""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    predictor = HeartDiseasePredictor()
    loaded = predictor.load_models('saved_models')
    
    print(f"✓ Models loaded: {loaded}")
    print(f"✓ Best model: {predictor.best_model_name if hasattr(predictor, 'best_model_name') else 'N/A'}")
    print(f"✓ Best model object exists: {predictor.best_model is not None if hasattr(predictor, 'best_model') else False}")
    print(f"✓ Scaler exists: {predictor.scaler is not None if hasattr(predictor, 'scaler') else False}")
    print(f"✓ Feature names: {len(predictor.feature_names) if hasattr(predictor, 'feature_names') and predictor.feature_names else 0} features")
    
    return predictor if loaded else None

def test_predictions(predictor):
    """Test that predictions work and vary with different inputs."""
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)
    
    # Test case 1: Low risk profile
    test_case_1 = pd.DataFrame([{
        'age': 35,
        'sex': 0,  # Female
        'cp': 0,  # Typical angina
        'trestbps': 110,
        'chol': 180,
        'fbs': 0,
        'restecg': 0,
        'thalch': 180,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }])
    
    # Test case 2: High risk profile
    test_case_2 = pd.DataFrame([{
        'age': 65,
        'sex': 1,  # Male
        'cp': 3,  # Asymptomatic
        'trestbps': 160,
        'chol': 280,
        'fbs': 1,
        'restecg': 2,
        'thalch': 120,
        'exang': 1,
        'oldpeak': 3.0,
        'slope': 2,
        'ca': 3,
        'thal': 3
    }])
    
    try:
        # Predict for test case 1
        pred1 = predictor.predict(test_case_1)
        prob1 = predictor.predict_proba(test_case_1)
        print(f"\nTest Case 1 (Low Risk Profile):")
        print(f"  Prediction: {pred1[0]}")
        print(f"  Probability: {prob1[0][1]:.2%}" if len(prob1[0]) > 1 else f"  Probability: {prob1[0][0]:.2%}")
        
        # Predict for test case 2
        pred2 = predictor.predict(test_case_2)
        prob2 = predictor.predict_proba(test_case_2)
        print(f"\nTest Case 2 (High Risk Profile):")
        print(f"  Prediction: {pred2[0]}")
        print(f"  Probability: {prob2[0][1]:.2%}" if len(prob2[0]) > 1 else f"  Probability: {prob2[0][0]:.2%}")
        
        # Check that predictions are different
        if prob1[0][1] != prob2[0][1]:
            print("\n✅ SUCCESS: Predictions vary based on input!")
        else:
            print("\n❌ FAIL: Predictions are the same (stuck at same value)")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    predictor = test_model_loading()
    
    if predictor:
        success = test_predictions(predictor)
        sys.exit(0 if success else 1)
    else:
        print("\n❌ Failed to load models")
        sys.exit(1)
