"""
Test script to verify predictions and check if the model is working correctly.
Run this locally to see if predictions vary based on risk factors.
"""

import sys
sys.path.append('/workspaces/CardioCheck.AI')

from models.heart_disease_prediction import HeartDiseasePredictor
import pandas as pd

# Initialize predictor
predictor = HeartDiseasePredictor()

# Load the trained models
print("Loading models...")
success = predictor.load_models('saved_models')

if not success:
    print("❌ Failed to load models")
    sys.exit(1)

print(f"✅ Models loaded. Best model: {predictor.best_model_name}")
print(f"✅ Scaler loaded: {predictor.scaler is not None}")
print(f"✅ Feature names: {predictor.feature_names}")

# Test Case 1: Low risk profile (young, healthy)
low_risk = {
    'age': 35,
    'sex': 1,  # Male
    'cp': 0,  # Typical angina
    'trestbps': 110,  # Good BP
    'chol': 180,  # Good cholesterol
    'fbs': 0,  # Normal blood sugar
    'restecg': 0,  # Normal ECG
    'thalch': 180,  # High max heart rate (good)
    'exang': 0,  # No exercise angina
    'oldpeak': 0.2,  # Low ST depression
    'slope': 0,  # Upsloping
    'ca': 0,  # No vessels
    'thal': 1  # Normal thalassemia
}

# Test Case 2: High risk profile (elderly, multiple risk factors)
high_risk = {
    'age': 70,
    'sex': 1,  # Male
    'cp': 3,  # Asymptomatic
    'trestbps': 170,  # High BP
    'chol': 300,  # High cholesterol
    'fbs': 1,  # High blood sugar
    'restecg': 2,  # Abnormal ECG
    'thalch': 100,  # Low max heart rate (bad)
    'exang': 1,  # Exercise angina present
    'oldpeak': 4.0,  # High ST depression
    'slope': 2,  # Downsloping
    'ca': 3,  # Multiple vessels
    'thal': 3  # Reversible defect
}

print("\n" + "="*60)
print("Testing Predictions...")
print("="*60)

# Test low risk
low_risk_df = pd.DataFrame([low_risk])
low_pred = predictor.predict(low_risk_df)
low_proba = predictor.predict_proba(low_risk_df)

print("\n🟢 Test Case 1: LOW RISK PROFILE")
print(f"Features: {low_risk}")
print(f"Prediction: {low_pred[0]} (0=No Disease, 1=Disease)")
print(f"Probability of Disease: {low_proba[0][1]:.2%}")

# Test high risk
high_risk_df = pd.DataFrame([high_risk])
high_pred = predictor.predict(high_risk_df)
high_proba = predictor.predict_proba(high_risk_df)

print("\n🔴 Test Case 2: HIGH RISK PROFILE")
print(f"Features: {high_risk}")
print(f"Prediction: {high_pred[0]} (0=No Disease, 1=Disease)")
print(f"Probability of Disease: {high_proba[0][1]:.2%}")

# Verify predictions are different
print("\n" + "="*60)
print("Verification Results:")
print("="*60)

if low_proba[0][1] < high_proba[0][1]:
    print("✅ SUCCESS: Low risk < High risk (model is working correctly!)")
    print(f"   Low risk: {low_proba[0][1]:.2%}")
    print(f"   High risk: {high_proba[0][1]:.2%}")
    print(f"   Difference: {(high_proba[0][1] - low_proba[0][1]):.2%}")
else:
    print("❌ PROBLEM: Model may be inverted or not working correctly")
    print(f"   Low risk: {low_proba[0][1]:.2%}")
    print(f"   High risk: {high_proba[0][1]:.2%}")
    print("   Expected: Low risk probability should be LOWER than high risk")

# Check if predictions are reasonable
if low_proba[0][1] < 0.3:
    print("✅ Low risk probability is reasonable (<30%)")
else:
    print(f"⚠️  Low risk probability seems high: {low_proba[0][1]:.2%}")

if high_proba[0][1] > 0.7:
    print("✅ High risk probability is reasonable (>70%)")
else:
    print(f"⚠️  High risk probability seems low: {high_proba[0][1]:.2%}")
