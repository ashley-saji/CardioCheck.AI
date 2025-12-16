"""
Retrain all models with current package versions for deployment compatibility.
"""

import os
import sys
from models.heart_disease_prediction import HeartDiseasePredictor

def main():
    print("🚀 Retraining Heart Disease Prediction Models")
    print("=" * 60)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Load data
    data_path = "data/heart_disease_uci.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        return False
    
    print(f"📂 Loading dataset from: {data_path}")
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data(data_path)
    
    if X is None or y is None:
        print("❌ Failed to load data")
        return False
    
    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Train models
    print("\n🔧 Training models...")
    predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Create ensemble model
    print("\n🔧 Creating ensemble model...")
    predictor.create_ensemble_model(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n📊 Results:")
    predictor.display_results()
    
    # Save models
    print("\n💾 Saving models...")
    predictor.save_models()
    
    print("\n✅ Retraining complete!")
    print(f"🏆 Best model: {predictor.best_model}")
    print(f"🎯 Best accuracy: {predictor.best_score:.4f}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
