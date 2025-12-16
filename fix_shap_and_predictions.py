"""
Patch to fix SHAP background data and verify predictions
"""

# This code should replace the get_background_data method in streamlit_app.py

def get_background_data_fixed(self):
    """Get background data for SHAP explainer with proper encoding."""
    try:
        # Try to load training data
        data_path = "data/heart_disease_uci.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Encode categorical columns to numeric
            # Sex: Male=1, Female=0
            if 'sex' in df.columns:
                df['sex'] = df['sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
            
            # Boolean columns
            bool_cols = ['fbs', 'exang']
            for col in bool_cols:
                if col in df.columns:
                    df[col] = df[col].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
            
            # Chest pain type
            if 'cp' in df.columns:
                cp_map = {
                    'typical angina': 0,
                    'atypical angina': 1,
                    'non-anginal': 2,
                    'asymptomatic': 3,
                    0: 0, 1: 1, 2: 2, 3: 3
                }
                df['cp'] = df['cp'].map(cp_map)
            
            # Resting ECG
            if 'restecg' in df.columns:
                restecg_map = {
                    'normal': 0,
                    'lv hypertrophy': 1,
                    'st-t abnormality': 2,
                    0: 0, 1: 1, 2: 2
                }
                df['restecg'] = df['restecg'].map(restecg_map)
            
            # Slope
            if 'slope' in df.columns:
                slope_map = {
                    'upsloping': 0,
                    'flat': 1,
                    'downsloping': 2,
                    0: 0, 1: 1, 2: 2
                }
                df['slope'] = df['slope'].map(slope_map)
            
            # Thalassemia
            if 'thal' in df.columns:
                thal_map = {
                    'normal': 1,
                    'fixed defect': 2,
                    'reversable defect': 3,
                    'reversible defect': 3,
                    1: 1, 2: 2, 3: 3
                }
                df['thal'] = df['thal'].map(thal_map)
            
            # Remove target and non-feature columns
            feature_cols = [col for col in df.columns if col not in ['target', 'num', 'id', 'dataset']]
            background_raw = df[feature_cols].sample(min(100, len(df)), random_state=42)
            
            # Convert all to numeric, drop any remaining non-numeric
            for col in background_raw.columns:
                background_raw[col] = pd.to_numeric(background_raw[col], errors='coerce')
            
            # Fill NaN with column median
            background_raw = background_raw.fillna(background_raw.median())
            
            # Apply same preprocessing pipeline as predictions
            background_processed = self.apply_feature_engineering(background_raw)
            
            # Use scaler to transform
            if hasattr(self, 'scaler') and self.scaler is not None:
                # Get expected features from predictor
                expected_features = None
                if hasattr(self.predictor, 'feature_names') and self.predictor.feature_names is not None:
                    expected_features = self.predictor.feature_names
                elif hasattr(self.scaler, 'feature_names_in_'):
                    expected_features = self.scaler.feature_names_in_
                
                if expected_features is not None:
                    # Align columns to match training
                    missing_cols = set(expected_features) - set(background_processed.columns)
                    extra_cols = set(background_processed.columns) - set(expected_features)
                    
                    if missing_cols:
                        for col in missing_cols:
                            background_processed[col] = 0
                    
                    if extra_cols:
                        background_processed = background_processed.drop(columns=list(extra_cols))
                    
                    # Reorder to match training
                    background_processed = background_processed[expected_features]
                
                background_scaled = self.scaler.transform(background_processed)
                return background_scaled[:50]
            else:
                return None
        else:
            return None
            
    except Exception as e:
        import traceback
        print(f"⚠️ Could not load background data: {str(e)}")
        print(traceback.format_exc())
        return None
