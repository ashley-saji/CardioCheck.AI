"""
Optimized Heart Disease Prediction with Maximum Accuracy
========================================================

This enhanced version implements advanced techniques to achieve maximum accuracy:
- Feature engineering and selection
- Hyperparameter tuning
- Advanced ensemble methods
- Improved data preprocessing

Author: Heart Disease Prediction Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, validation_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
import shap
from tqdm import tqdm
import os
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OptimizedHeartDiseasePredictor:
    """
    Optimized heart disease prediction system with advanced techniques for maximum accuracy.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.optimized_models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_instance = None
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        self.feature_names = None
        self.selected_features = None
        self.feature_selector = None
        
        # Initialize models with better defaults
        self._initialize_optimized_models()
    
    def load_models(self, model_dir='optimized_models'):
        """
        Load pre-trained models and preprocessing components.
        
        Args:
            model_dir: Directory containing saved models
        """
        try:
            # Load preprocessing components
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded successfully")
            
            feature_selector_path = os.path.join(model_dir, 'feature_selector.joblib')
            if os.path.exists(feature_selector_path):
                self.feature_selector = joblib.load(feature_selector_path)
                print("✅ Feature selector loaded successfully")
            
            selected_features_path = os.path.join(model_dir, 'selected_features.joblib')
            if os.path.exists(selected_features_path):
                self.selected_features = joblib.load(selected_features_path)
                print("✅ Selected features loaded successfully")
            
            # Load best model
            best_model_path = os.path.join(model_dir, 'best_model.joblib')
            if os.path.exists(best_model_path):
                self.best_model_instance = joblib.load(best_model_path)
                self.best_model = type(self.best_model_instance).__name__
                print(f"✅ Best model ({self.best_model}) loaded successfully")
            
            # Load individual models
            model_files = {
                'Random Forest': 'random_forest_model.joblib',
                'XGBoost': 'xgboost_model.joblib',
                'Gradient Boosting': 'gradient_boosting_model.joblib',
                'Neural Network': 'neural_network_model.joblib',
                'SVM': 'svm_model.joblib',
                'Logistic Regression': 'logistic_regression_model.joblib',
                'AdaBoost': 'adaboost_model.joblib',
                'Extra Trees': 'extra_trees_model.joblib'
            }
            
            self.optimized_models = {}
            for model_name, filename in model_files.items():
                model_path = os.path.join(model_dir, filename)
                if os.path.exists(model_path):
                    try:
                        self.optimized_models[model_name] = joblib.load(model_path)
                        print(f"✅ {model_name} loaded successfully")
                    except Exception as e:
                        print(f"⚠️ Failed to load {model_name}: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def apply_feature_engineering(self, X):
        """
        Apply feature engineering for new data (wrapper for Streamlit app).
        
        Args:
            X: Input DataFrame
            
        Returns:
            X_processed: Processed DataFrame ready for prediction
        """
        return self.advanced_feature_engineering(X)
    
    def _initialize_optimized_models(self):
        """Initialize optimized machine learning models with better hyperparameters."""
        
        # Optimized Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Try to initialize XGBoost with optimized parameters
        try:
            import xgboost as xgb
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
            )
            print("✅ XGBoost successfully loaded with optimized parameters")
        except ImportError:
            print("⚠️  XGBoost not available")
        
        # Optimized Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Optimized SVM
        self.models['SVM'] = SVC(
            C=10,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        # Optimized Neural Network
        self.models['Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(150, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state
        )
        
        # Optimized Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            C=10,
            penalty='l2',
            solver='liblinear',
            max_iter=2000,
            random_state=self.random_state
        )
        
        # Extra Trees (often performs better than Random Forest)
        self.models['Extra Trees'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # AdaBoost
        self.models['AdaBoost'] = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            learning_rate=1.0,
            random_state=self.random_state
        )
    
    def advanced_feature_engineering(self, X, y=None):
        """
        Perform advanced feature engineering.
        
        Args:
            X: Feature matrix
            y: Target vector (optional, for supervised feature selection)
            
        Returns:
            X_engineered: Engineered feature matrix
        """
        print("🔧 Performing advanced feature engineering...")
        
        X_engineered = X.copy()
        
        # Remove ID column if present
        if 'id' in X_engineered.columns:
            print("🗑️  Removing ID column")
            X_engineered = X_engineered.drop(columns=['id'])
        
        # Remove dataset column if present (metadata, not predictive)
        if 'dataset' in X_engineered.columns:
            print("🗑️  Removing dataset column")
            X_engineered = X_engineered.drop(columns=['dataset'])
        
        # Encode categorical variables
        print("🔄 Encoding categorical variables...")
        
        # Sex encoding
        if 'sex' in X_engineered.columns:
            if X_engineered['sex'].dtype == 'object':
                X_engineered['sex'] = X_engineered['sex'].map({'Male': 1, 'Female': 0}).fillna(0)
        
        # Chest pain type encoding (if it's categorical)
        if 'cp' in X_engineered.columns:
            if X_engineered['cp'].dtype == 'object':
                cp_mapping = {
                    'Typical Angina': 0,
                    'Atypical Angina': 1, 
                    'Non-Anginal Pain': 2,
                    'Asymptomatic': 3
                }
                X_engineered['cp'] = X_engineered['cp'].map(cp_mapping).fillna(0)
        
        # Fasting blood sugar encoding
        if 'fbs' in X_engineered.columns:
            if X_engineered['fbs'].dtype == 'object':
                X_engineered['fbs'] = X_engineered['fbs'].map({'Yes': 1, 'No': 0}).fillna(0)
        
        # Resting ECG encoding
        if 'restecg' in X_engineered.columns:
            if X_engineered['restecg'].dtype == 'object':
                restecg_mapping = {
                    'Normal': 0,
                    'ST-T Wave Abnormality': 1,
                    'Left Ventricular Hypertrophy': 2
                }
                X_engineered['restecg'] = X_engineered['restecg'].map(restecg_mapping).fillna(0)
        
        # Exercise induced angina encoding
        if 'exang' in X_engineered.columns:
            if X_engineered['exang'].dtype == 'object':
                X_engineered['exang'] = X_engineered['exang'].map({'Yes': 1, 'No': 0}).fillna(0)
        
        # ST slope encoding
        if 'slope' in X_engineered.columns:
            if X_engineered['slope'].dtype == 'object':
                slope_mapping = {
                    'Upsloping': 0,
                    'Flat': 1,
                    'Downsloping': 2
                }
                X_engineered['slope'] = X_engineered['slope'].map(slope_mapping).fillna(0)
        
        # Thalassemia encoding
        if 'thal' in X_engineered.columns:
            if X_engineered['thal'].dtype == 'object':
                thal_mapping = {
                    'Normal': 0,
                    'Fixed Defect': 1,
                    'Reversible Defect': 2
                }
                X_engineered['thal'] = X_engineered['thal'].map(thal_mapping).fillna(0)
        
        # Ensure all columns are numeric
        for col in X_engineered.columns:
            if X_engineered[col].dtype == 'object':
                try:
                    X_engineered[col] = pd.to_numeric(X_engineered[col], errors='coerce').fillna(0)
                except:
                    print(f"⚠️ Could not convert {col} to numeric")
        
        # Create interaction features for important clinical combinations
        if all(col in X_engineered.columns for col in ['age', 'chol']):
            X_engineered['age_chol_interaction'] = X_engineered['age'] * X_engineered['chol']
        
        if all(col in X_engineered.columns for col in ['age', 'trestbps']):
            X_engineered['age_bp_interaction'] = X_engineered['age'] * X_engineered['trestbps']
        
        if all(col in X_engineered.columns for col in ['cp', 'exang']):
            X_engineered['cp_exang_interaction'] = X_engineered['cp'] * X_engineered['exang']
        
        # Create age groups (risk categories)
        if 'age' in X_engineered.columns:
            age_groups = pd.cut(
                X_engineered['age'], 
                bins=[0, 40, 50, 60, 100], 
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            # Handle NaN values by filling with median category
            X_engineered['age_group'] = age_groups.fillna(1).astype(int)
        
        # Create cholesterol categories
        if 'chol' in X_engineered.columns:
            chol_categories = pd.cut(
                X_engineered['chol'],
                bins=[0, 200, 240, 1000],
                labels=[0, 1, 2],
                include_lowest=True
            )
            # Handle NaN values by filling with median category
            X_engineered['chol_category'] = chol_categories.fillna(1).astype(int)
        
        # Create blood pressure categories
        if 'trestbps' in X_engineered.columns:
            bp_categories = pd.cut(
                X_engineered['trestbps'],
                bins=[0, 120, 140, 1000],
                labels=[0, 1, 2],
                include_lowest=True
            )
            # Handle NaN values by filling with median category
            X_engineered['bp_category'] = bp_categories.fillna(1).astype(int)
        
        print(f"✅ Feature engineering complete. New shape: {X_engineered.shape}")
        return X_engineered
    
    def feature_selection(self, X, y, method='rfecv', k=15):
        """
        Perform feature selection to identify most important features.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method ('selectk', 'rfe', 'rfecv')
            k: Number of features to select
            
        Returns:
            X_selected: Selected features
        """
        print(f"🎯 Performing feature selection using {method}...")
        
        if method == 'selectk':
            # Select K best features
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'rfecv':
            # Recursive Feature Elimination with Cross-Validation
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            selector = RFECV(estimator, step=1, cv=StratifiedKFold(5), scoring='accuracy')
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            selected_indices = selector.get_support(indices=True)
            k = selector.n_features_
        
        # Store selected feature names
        if isinstance(X, pd.DataFrame):
            self.selected_features = X.columns[selected_indices].tolist()
            print(f"✅ Selected {k} features: {self.selected_features}")
        else:
            self.selected_features = [f"feature_{i}" for i in selected_indices]
            print(f"✅ Selected {k} features")
        
        return X_selected
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model: Model to tune
            param_grid: Parameter grid
            X_train: Training features
            y_train: Training targets
            cv: Cross-validation folds
            
        Returns:
            best_model: Tuned model
        """
        print(f"⚙️  Tuning hyperparameters for {type(model).__name__}...")
        
        # Use RandomizedSearchCV for faster tuning with large parameter spaces
        search = RandomizedSearchCV(
            model, 
            param_grid, 
            n_iter=20,  # Limit iterations for faster execution
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        
        print(f"✅ Best parameters: {search.best_params_}")
        print(f"✅ Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def create_advanced_ensemble(self, X_train, y_train):
        """
        Create an advanced ensemble model using multiple techniques.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            ensemble_model: Advanced ensemble model
        """
        print("🎯 Creating advanced ensemble model...")
        
        # Get the best performing individual models
        top_models = []
        for name, result in self.results.items():
            if result['accuracy'] > 0.85:  # Only include high-performing models
                top_models.append((name.replace(' ', '_'), result['model']))
        
        if len(top_models) < 2:
            print("⚠️  Not enough high-performing models for ensemble")
            return None
        
        # Create voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        
        # Create bagging ensemble of the voting classifier
        bagged_ensemble = BaggingClassifier(
            estimator=voting_ensemble,
            n_estimators=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print(f"✅ Advanced ensemble created with {len(top_models)} models")
        return bagged_ensemble
    
    def load_and_preprocess_data(self, file_path):
        """Enhanced data loading and preprocessing."""
        print("📊 Loading and preprocessing data with advanced techniques...")
        
        try:
            data = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {data.shape}")
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return None, None
        
        # Display basic info
        print(f"\n📈 Dataset Info:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Handle missing values with advanced techniques
        print("🔧 Advanced missing value handling...")
        
        # For numerical columns, use median imputation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                if col in ['ca', 'thal']:  # Categorical but numeric
                    data[col].fillna(data[col].mode()[0], inplace=True)
                else:
                    data[col].fillna(data[col].median(), inplace=True)
        
        # For categorical columns, use mode imputation
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Handle outliers using IQR method for continuous variables
        continuous_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        for col in continuous_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Separate features and target
        target_columns = ['target', 'num', 'heart_disease', 'diagnosis', 'class']
        target_col = None
        
        for col in target_columns:
            if col in data.columns:
                target_col = col
                break
        
        if target_col is None:
            target_col = data.columns[-1]
            print(f"⚠️  Target column not found. Using last column: {target_col}")
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Convert multi-class to binary classification
        if y.nunique() > 2:
            print(f"🔄 Converting multi-class target to binary: 0 = no disease, 1+ = disease")
            print(f"📊 Original distribution: {y.value_counts().to_dict()}")
            y = (y > 0).astype(int)
            print(f"📊 Binary distribution: {y.value_counts().to_dict()}")
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Apply feature engineering
        X = self.advanced_feature_engineering(X)
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                # Convert categorical to numeric if needed
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    # Handle categorical encoding
                    X[col] = pd.Categorical(X[col]).codes
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        print(f"✅ Advanced preprocessing complete!")
        print(f"✅ Final features: {len(X.columns)}")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_optimized_models(self, X_train, y_train, X_test, y_test, use_tuning=True):
        """Train models with optimization techniques."""
        print("\n🚀 Training optimized ML models...")
        print("=" * 70)
        
        # Apply feature selection
        print("🎯 Applying feature selection...")
        X_train_selected = self.feature_selection(X_train, y_train, method='rfecv')
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Handle class imbalance with SMOTE
        print("⚖️  Applying SMOTE for class balance...")
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
        print(f"✅ Balanced dataset: {np.bincount(y_train_balanced)}")
        
        # Scale features
        scaler = self.scalers['robust']  # Robust scaler handles outliers better
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Define hyperparameter grids for tuning
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        }
        
        for name, model in tqdm(self.models.items(), desc="Training optimized models"):
            print(f"\n🔧 Optimizing {name}...")
            
            try:
                # Determine which features to use (scaled vs original)
                if name in ['SVM', 'Neural Network', 'Logistic Regression']:
                    X_train_final = X_train_scaled
                    X_test_final = X_test_scaled
                else:
                    X_train_final = X_train_balanced
                    X_test_final = X_test_selected
                
                # Apply hyperparameter tuning if requested and grid is available
                if use_tuning and name in param_grids:
                    optimized_model = self.hyperparameter_tuning(
                        model, param_grids[name], X_train_final, y_train_balanced, cv=3
                    )
                else:
                    optimized_model = model
                    optimized_model.fit(X_train_final, y_train_balanced)
                
                # Make predictions
                y_pred = optimized_model.predict(X_test_final)
                y_pred_proba = None
                if hasattr(optimized_model, 'predict_proba'):
                    y_pred_proba = optimized_model.predict_proba(X_test_final)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                roc_auc = None
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        roc_auc = None
                
                # Store results
                self.results[name] = {
                    'model': optimized_model,
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
                    self.best_model_instance = optimized_model
                
                print(f"✅ {name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
        
        # Create advanced ensemble
        try:
            advanced_ensemble = self.create_advanced_ensemble(X_train_final, y_train_balanced)
            if advanced_ensemble is not None:
                advanced_ensemble.fit(X_train_scaled, y_train_balanced)  # Use scaled features for ensemble
                
                y_pred = advanced_ensemble.predict(X_test_scaled)
                y_pred_proba = advanced_ensemble.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                self.results['Advanced Ensemble'] = {
                    'model': advanced_ensemble,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = 'Advanced Ensemble'
                    self.best_model_instance = advanced_ensemble
                
                print(f"✅ Advanced Ensemble: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
        except Exception as e:
            print(f"❌ Error creating advanced ensemble: {str(e)}")
        
        # Store the scaler and feature selector for later use
        self.scaler = scaler
    
    def display_results(self):
        """Display comprehensive results."""
        print("\n📊 OPTIMIZED MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [result['accuracy'] for result in self.results.values()],
            'Precision': [result['precision'] for result in self.results.values()],
            'Recall': [result['recall'] for result in self.results.values()],
            'F1-Score': [result['f1_score'] for result in self.results.values()],
            'ROC-AUC': [result['roc_auc'] if result['roc_auc'] else 0 for result in self.results.values()]
        })
        
        results_df = results_df.sort_values('Accuracy', ascending=False)
        print(results_df.to_string(index=False, float_format='%.4f'))
        print(f"\n🏆 Best Model: {self.best_model} (Accuracy: {self.best_score:.4f})")
        
        # Check if we achieved >90% accuracy
        if self.best_score > 0.90:
            print("🎉 EXCELLENT! Achieved >90% accuracy target!")
        elif self.best_score > 0.85:
            print("✅ Great performance! >85% accuracy achieved")
        else:
            print("📈 Good baseline. Consider additional optimization techniques")
    
    def save_optimized_models(self, save_dir='optimized_models'):
        """Save all optimized models and preprocessing components."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"\n💾 Saving optimized models to {save_dir}/...")
        
        # Save models
        for name, result in self.results.items():
            filename = f"{save_dir}/{name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(result['model'], filename)
            print(f"✅ Saved {name} model")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f"{save_dir}/scaler.joblib")
        joblib.dump(self.feature_selector, f"{save_dir}/feature_selector.joblib")
        joblib.dump(self.selected_features, f"{save_dir}/selected_features.joblib")
        print("✅ Saved preprocessing components")
        
        # Save best model separately for easy access
        if self.best_model_instance is not None:
            joblib.dump(self.best_model_instance, f"{save_dir}/best_model.joblib")
            print(f"✅ Saved best model ({self.best_model}) separately")
    
    def predict_with_confidence(self, X_new):
        """
        Make predictions with confidence intervals for new data.
        
        Args:
            X_new: New data to predict
            
        Returns:
            dict: Predictions with confidence levels
        """
        if self.best_model_instance is None:
            raise ValueError("No trained model available")
        
        # Apply same preprocessing pipeline
        X_engineered = self.advanced_feature_engineering(X_new)
        X_selected = self.feature_selector.transform(X_engineered)
        X_scaled = self.scaler.transform(X_selected)
        
        # Get prediction and probability
        if hasattr(self.best_model_instance, 'predict_proba'):
            probabilities = self.best_model_instance.predict_proba(X_scaled)
            disease_prob = probabilities[:, 1]
            prediction = (disease_prob > 0.5).astype(int)
        else:
            prediction = self.best_model_instance.predict(X_scaled)
            disease_prob = None
        
        # Create confidence categories
        results = []
        for i, pred in enumerate(prediction):
            prob = disease_prob[i] if disease_prob is not None else None
            
            if prob is not None:
                if prob >= 0.8:
                    risk_level = "Highly Likely"
                    confidence = "High"
                elif prob >= 0.6:
                    risk_level = "Moderately Likely"
                    confidence = "Medium-High"
                elif prob >= 0.4:
                    risk_level = "Possible"
                    confidence = "Medium"
                elif prob >= 0.2:
                    risk_level = "Less Likely"
                    confidence = "Medium-Low"
                else:
                    risk_level = "Very Unlikely"
                    confidence = "High"
                
                results.append({
                    'prediction': int(pred),
                    'probability': float(prob),
                    'risk_level': risk_level,
                    'confidence': confidence,
                    'percentage': f"{prob*100:.1f}%"
                })
            else:
                results.append({
                    'prediction': int(pred),
                    'probability': None,
                    'risk_level': "Positive" if pred == 1 else "Negative",
                    'confidence': "Medium",
                    'percentage': "N/A"
                })
        
        return results


def main():
    """Main function for optimized heart disease prediction."""
    print("🚀 Optimized Heart Disease Prediction for Maximum Accuracy")
    print("=" * 70)
    
    # Initialize optimized predictor
    predictor = OptimizedHeartDiseasePredictor()
    
    # Load and preprocess data
    data_path = "data/heart_disease_uci.csv"
    X, y = predictor.load_and_preprocess_data(data_path)
    
    if X is None or y is None:
        print("❌ Failed to load data.")
        return
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split completed:")
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Testing set: {X_test.shape[0]} samples")
    
    # Train optimized models
    predictor.train_optimized_models(X_train, y_train, X_test, y_test, use_tuning=True)
    
    # Display results
    predictor.display_results()
    
    # Save optimized models
    predictor.save_optimized_models()
    
    print(f"\n✅ Optimization complete!")
    print(f"🏆 Best model: {predictor.best_model}")
    print(f"🎯 Best accuracy: {predictor.best_score:.4f}")
    
    return predictor


if __name__ == "__main__":
    optimized_predictor = main()