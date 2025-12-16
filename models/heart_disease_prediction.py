"""
Heart Disease Prediction with Multiple Machine Learning Models
=============================================================

This script implements multiple machine learning models for heart disease prediction
with comprehensive evaluation and SHAP interpretability analysis.

Author: Heart Disease Prediction Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import joblib
import shap
from tqdm import tqdm
import os

# Try importing custom modules
try:
    from model_evaluation import ModelEvaluator
    print("✅ ModelEvaluator imported successfully")
except ImportError:
    print("⚠️  ModelEvaluator not available")
    ModelEvaluator = None

try:
    from shap_analysis import run_shap_analysis_for_model
    print("✅ SHAP analysis module imported successfully")
except ImportError:
    print("⚠️  SHAP analysis module not available")
    run_shap_analysis_for_model = None

# Import our custom modules
try:
    from .shap_analysis import run_shap_analysis_for_model
    from .model_evaluation import ModelEvaluator
except ImportError:
    # Handle case when running as standalone script
    import sys
    sys.path.append('.')
    try:
        from shap_analysis import run_shap_analysis_for_model
        from model_evaluation import ModelEvaluator
    except ImportError:
        print("⚠️  SHAP analysis and model evaluation modules not found.")
        print("    Some advanced features may not be available.")
        run_shap_analysis_for_model = None
        ModelEvaluator = None

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HeartDiseasePredictor:
    """
    A comprehensive heart disease prediction system using multiple ML models.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all machine learning models."""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': None,  # Will be initialized if xgboost is available
            'SVM': SVC(
                probability=True, 
                random_state=self.random_state,
                kernel='rbf'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Try to import XGBoost
        try:
            import xgboost as xgb
            self.models['XGBoost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )
            print("✅ XGBoost successfully loaded")
        except ImportError:
            print("⚠️  XGBoost not available. Install with: pip install xgboost")
            del self.models['XGBoost']
    
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the heart disease dataset.
        
        Args:
            file_path (str): Path to the dataset CSV file
            
        Returns:
            tuple: X (features), y (target)
        """
        print("📊 Loading and preprocessing data...")
        
        # Load data
        try:
            data = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {data.shape}")
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            print("Please ensure your dataset is in the correct location.")
            return None, None
        
        # Display basic info
        print(f"\n📈 Dataset Info:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\n🔍 Missing values:")
        print(data.isnull().sum())
        
        # Handle missing values
        if data.isnull().sum().sum() > 0:
            print("🔧 Handling missing values...")
            # Fill numeric columns with median
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                data[col] = data[col].fillna(data[col].mode()[0])
        
        # Encode categorical variables
        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            if column != 'target' and column in data.columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                label_encoders[column] = le
        
        # Separate features and target
        # Common heart disease dataset target column names
        target_columns = ['target', 'num', 'heart_disease', 'diagnosis', 'class']
        target_col = None
        
        for col in target_columns:
            if col in data.columns:
                target_col = col
                break
        
        if target_col is None:
            # Assume last column is target
            target_col = data.columns[-1]
            print(f"⚠️  Target column not found. Using last column: {target_col}")
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Convert multi-class to binary classification (0 = no disease, 1+ = disease)
        if y.nunique() > 2:
            print(f"🔄 Converting multi-class target to binary: 0 = no disease, 1+ = disease")
            print(f"📊 Original distribution: {y.value_counts().to_dict()}")
            y = (y > 0).astype(int)
            print(f"📊 Binary distribution: {y.value_counts().to_dict()}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Features: {len(X.columns)}")
        print(f"✅ Target distribution:")
        print(y.value_counts())
        
        return X, y
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate their performance.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
        """
        print("\n🚀 Training multiple ML models...")
        print("=" * 60)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in tqdm(self.models.items(), desc="Training models"):
            if model is None:
                continue
                
            print(f"\n🔧 Training {name}...")
            
            try:
                # Train model
                if name in ['SVM', 'Neural Network', 'Logistic Regression', 'K-Nearest Neighbors']:
                    # These models benefit from scaled features
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                else:
                    # Tree-based models don't need scaling
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # ROC AUC (if probability predictions available)
                roc_auc = None
                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Update best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = name
                
                print(f"✅ {name}: Accuracy = {accuracy:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
    
    def create_ensemble_model(self, X_train, y_train, X_test, y_test):
        """
        Create an ensemble model using the best performing models.
        """
        print("\n🎯 Creating Ensemble Model...")
        
        # Select top models for ensemble (exclude models with low performance)
        top_models = []
        for name, result in self.results.items():
            if result['accuracy'] > 0.75:  # Only include models with >75% accuracy
                top_models.append((name, result['model']))
        
        if len(top_models) >= 2:
            ensemble = VotingClassifier(
                estimators=top_models,
                voting='soft'  # Use probability predictions
            )
            
            # Scale features for ensemble
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            try:
                ensemble.fit(X_train_scaled, y_train)
                y_pred = ensemble.predict(X_test_scaled)
                y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                self.results['Ensemble'] = {
                    'model': ensemble,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"✅ Ensemble Model: Accuracy = {accuracy:.4f}")
                
                # Update best model if ensemble is better
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = 'Ensemble'
                    
            except Exception as e:
                print(f"❌ Error creating ensemble: {str(e)}")
    
    def display_results(self):
        """Display comprehensive results for all models."""
        print("\n📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [result['accuracy'] for result in self.results.values()],
            'Precision': [result['precision'] for result in self.results.values()],
            'Recall': [result['recall'] for result in self.results.values()],
            'F1-Score': [result['f1_score'] for result in self.results.values()],
            'ROC-AUC': [result['roc_auc'] if result['roc_auc'] else 0 for result in self.results.values()]
        })
        
        # Sort by accuracy
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print(results_df.to_string(index=False, float_format='%.4f'))
        print(f"\n🏆 Best Model: {self.best_model} (Accuracy: {self.best_score:.4f})")
    
    def plot_model_comparison(self):
        """Create visualization comparing all models."""
        if not self.results:
            print("❌ No results to plot. Train models first.")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[model][metric] for model in models]
            
            bars = ax.bar(models, values, alpha=0.8)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, y_test):
        """Plot confusion matrices for all models."""
        if not self.results:
            print("❌ No results to plot. Train models first.")
            return
        
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, result) in enumerate(self.results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = confusion_matrix(y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            if rows > 1:
                row, col = idx // cols, idx % cols
                axes[row, col].axis('off')
            elif cols > n_models:
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, save_dir='saved_models'):
        """Save all trained models."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"\n💾 Saving models to {save_dir}/...")
        
        for name, result in self.results.items():
            filename = f"{save_dir}/{name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(result['model'], filename)
            print(f"✅ Saved {name} model")
        
        # Save scaler
        joblib.dump(self.scaler, f"{save_dir}/scaler.joblib")
        print("✅ Saved scaler")
        
        # Save feature names
        joblib.dump(self.feature_names, f"{save_dir}/feature_names.joblib")
        print("✅ Saved feature names")


def main():
    """Main function to run the heart disease prediction analysis."""
    print("🚀 Heart Disease Prediction with Multiple ML Models")
    print("=" * 60)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Load data (you'll need to specify the correct path to your dataset)
    data_path = "data/heart_disease_uci.csv"  # Update this path
    
    print(f"📂 Looking for dataset at: {data_path}")
    print("📝 Note: Please ensure your heart disease dataset is placed in the data/ directory")
    print("🔍 Common dataset formats supported:")
    print("   - Cleveland Heart Disease Dataset")
    print("   - Heart Disease UCI Dataset")
    print("   - Any CSV with heart disease features and target column")
    
    # For demonstration, create a sample dataset if file doesn't exist
    if not os.path.exists(data_path):
        print("\n⚠️  Dataset not found. Creating sample dataset for demonstration...")
        create_sample_dataset(data_path)
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data(data_path)
    
    if X is None or y is None:
        print("❌ Failed to load data. Please check the file path and format.")
        return
    
    # Split data
    print("\n📊 Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Testing set: {X_test.shape[0]} samples")
    
    # Train models
    predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Create ensemble model
    predictor.create_ensemble_model(X_train, y_train, X_test, y_test)
    
    # Display results
    predictor.display_results()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    predictor.plot_model_comparison()
    predictor.plot_confusion_matrices(y_test)
    
    # Advanced Model Evaluation
    if ModelEvaluator is not None:
        print("\n🔬 Running Advanced Model Evaluation...")
        evaluator = ModelEvaluator(
            models_dict=predictor.models,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=predictor.feature_names
        )
        
        evaluator.evaluate_all_models()
        evaluator.cross_validate_models(cv=5)
        summary_df = evaluator.generate_comprehensive_report()
    
    # SHAP Analysis for Best Models
    if run_shap_analysis_for_model is not None:
        print("\n🔍 Running SHAP Analysis for Model Interpretability...")
        
        # Get top 3 performing models for SHAP analysis
        top_models = []
        for name, result in predictor.results.items():
            top_models.append((name, result['accuracy'], result['model']))
        
        top_models.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, accuracy, model) in enumerate(top_models[:3]):
            print(f"\n🎯 SHAP Analysis {i+1}/3: {model_name} (Accuracy: {accuracy:.4f})")
            
            try:
                analyzer = run_shap_analysis_for_model(
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    feature_names=predictor.feature_names,
                    model_name=model_name
                )
                
                if analyzer:
                    print(f"✅ SHAP analysis completed for {model_name}")
                else:
                    print(f"⚠️  SHAP analysis partially completed for {model_name}")
                    
            except Exception as e:
                print(f"❌ SHAP analysis failed for {model_name}: {str(e)}")
    
    # Save models
    predictor.save_models()
    
    print("\n✅ Analysis complete!")
    print(f"🏆 Best performing model: {predictor.best_model}")
    print(f"🎯 Best accuracy achieved: {predictor.best_score:.4f}")
    
    # Final recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("=" * 40)
    print(f"1. 🥇 Deploy the {predictor.best_model} model for production")
    print("2. 📊 Review SHAP plots for feature importance insights")
    print("3. 🔧 Consider hyperparameter tuning for further optimization")
    print("4. 📈 Monitor model performance on new data")
    print("5. 🔄 Retrain periodically with updated datasets")


def create_sample_dataset(file_path):
    """Create a sample heart disease dataset for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic heart disease data
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),  # chest pain type
        'trestbps': np.random.randint(90, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(120, 400, n_samples),  # cholesterol
        'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar
        'restecg': np.random.randint(0, 3, n_samples),  # resting ECG
        'thalach': np.random.randint(80, 200, n_samples),  # max heart rate
        'exang': np.random.randint(0, 2, n_samples),  # exercise induced angina
        'oldpeak': np.random.uniform(0, 6, n_samples),  # ST depression
        'slope': np.random.randint(0, 3, n_samples),  # slope of ST segment
        'ca': np.random.randint(0, 4, n_samples),  # number of major vessels
        'thal': np.random.randint(0, 4, n_samples),  # thalassemia
    }
    
    # Create target variable with some correlation to features
    target_prob = (
        0.1 * (data['age'] > 55) +
        0.2 * (data['cp'] > 0) +
        0.15 * (data['chol'] > 240) +
        0.1 * (data['thalach'] < 120) +
        0.15 * (data['exang'] == 1) +
        0.1 * (data['oldpeak'] > 2) +
        0.2 * np.random.random(n_samples)
    )
    
    data['target'] = (target_prob > 0.5).astype(int)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    
    print(f"✅ Sample dataset created: {file_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Target distribution: {df['target'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()