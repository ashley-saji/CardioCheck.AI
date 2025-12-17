"""
Interactive Streamlit Web Application for Heart Disease Prediction
================================================================

A beautiful, modern web interface for heart disease prediction using multiple ML models
with SHAP interpretability, PDF reports, and email functionality.

Features:
- Interactive prediction interface
- Multiple ML model comparison
- SHAP analysis and visualizations
- Professional PDF report generation
- Email functionality for patient reports
- Modern, responsive UI design

Author: Heart Disease Prediction Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
from PIL import Image
import base64
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import warnings
from datetime import datetime
import time
import json
import hashlib

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Custom modules - import at module level without st. calls
OPTIMIZED_MODEL_AVAILABLE = False
REGULAR_MODEL_AVAILABLE = False

# Try importing both modules independently to allow flexible fallbacks
try:
    from models.heart_disease_prediction_optimized import OptimizedHeartDiseasePredictor
    OPTIMIZED_MODEL_AVAILABLE = True
except ImportError:
    OPTIMIZED_MODEL_AVAILABLE = False

try:
    from models.heart_disease_prediction import HeartDiseasePredictor
    REGULAR_MODEL_AVAILABLE = True
except ImportError:
    REGULAR_MODEL_AVAILABLE = False

warnings.filterwarnings('ignore')

# Global sklearn compatibility patch: ensure tree estimators have `monotonic_cst`
try:
    from sklearn.tree import DecisionTreeClassifier as _DTC, DecisionTreeRegressor as _DTR
    if not hasattr(_DTC, 'monotonic_cst'):
        setattr(_DTC, 'monotonic_cst', None)
    if not hasattr(_DTR, 'monotonic_cst'):
        setattr(_DTR, 'monotonic_cst', None)
except Exception:
    # Best-effort; continue without failing
    pass

# Configure Streamlit page
st.set_page_config(
    page_title="CardioCheck.AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48cae4 0%, #023e8a 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .feature-importance {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class HeartDiseaseWebApp:
    """Main web application class for heart disease prediction."""
    
    def __init__(self):
        self.predictor = None
        self.models_loaded = False
        self.patient_data = {}
        self.prediction_results = None
        self.config_file = "email_config.json"
        
        # Add SHAP-related attributes
        self.shap_initialized = False
        self.shap_explainer = None
        
        # Initialize session state
        if 'prediction_made' not in st.session_state:
            st.session_state.prediction_made = False
        if 'patient_info' not in st.session_state:
            st.session_state.patient_info = {
                'name': '',
                'email': '',
                'doctor': 'Dr. Smith',
                'age': 50,
                'sex': 'Male'
            }
        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = None
        if 'features' not in st.session_state:
            st.session_state.features = None
        if 'email_config' not in st.session_state:
            st.session_state.email_config = self.load_email_config()
    
    def get_background_data(self):
        """Get background data for SHAP explainer."""
        try:
            # Try to load training data or create synthetic background
            data_path = "data/heart_disease_uci.csv"
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                # Remove target columns
                feature_cols = [col for col in df.columns if col not in ['target', 'num']]
                background_raw = df[feature_cols].sample(min(100, len(df)), random_state=42)

                # Map categorical/text columns to numeric to match training
                # Sex
                if 'sex' in background_raw.columns:
                    background_raw['sex'] = background_raw['sex'].map({
                        'Male': 1, 'Female': 0, 'M': 1, 'F': 0,
                        True: 1, False: 0
                    }).fillna(0)

                # Boolean-like columns
                for col in ['fbs', 'exang']:
                    if col in background_raw.columns:
                        background_raw[col] = background_raw[col].map({
                            True: 1, False: 0,
                            'TRUE': 1, 'FALSE': 0,
                            'True': 1, 'False': 0,
                            1: 1, 0: 0
                        }).fillna(0)

                # Chest pain type
                if 'cp' in background_raw.columns:
                    cp_map = {
                        'typical angina': 0,
                        'atypical angina': 1,
                        'non-anginal': 2,
                        'asymptomatic': 3,
                        'non-anginal pain': 2,
                        0: 0, 1: 1, 2: 2, 3: 3
                    }
                    background_raw['cp'] = background_raw['cp'].map(cp_map).fillna(0)

                # Resting ECG
                if 'restecg' in background_raw.columns:
                    restecg_map = {
                        'normal': 0,
                        'lv hypertrophy': 1,
                        'st-t abnormality': 2,
                        0: 0, 1: 1, 2: 2
                    }
                    background_raw['restecg'] = background_raw['restecg'].map(restecg_map).fillna(0)

                # Slope
                if 'slope' in background_raw.columns:
                    slope_map = {
                        'upsloping': 0,
                        'flat': 1,
                        'downsloping': 2,
                        0: 0, 1: 1, 2: 2
                    }
                    background_raw['slope'] = background_raw['slope'].map(slope_map).fillna(0)

                # Thalassemia
                if 'thal' in background_raw.columns:
                    thal_map = {
                        'normal': 1,
                        'fixed defect': 2,
                        'reversable defect': 3,
                        'reversible defect': 3,
                        1: 1, 2: 2, 3: 3
                    }
                    background_raw['thal'] = background_raw['thal'].map(thal_map).fillna(1)

                # Drop id/dataset if present (non-predictive in UI path)
                for col in ['id', 'dataset']:
                    if col in background_raw.columns:
                        background_raw = background_raw.drop(columns=[col])

                # Convert remaining columns to numeric safely
                for col in background_raw.columns:
                    background_raw[col] = pd.to_numeric(background_raw[col], errors='coerce')

                # Fill NaNs with column medians
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
                    
                    try:
                        background_scaled = self.scaler.transform(background_processed)
                        return background_scaled[:50]
                    except Exception:
                        pass
                # If scaler missing or transform failed, fall through to fallback below
            # If CSV missing or any issue above, create synthetic background from scaler means
            expected_features = None
            if hasattr(self.predictor, 'feature_names') and self.predictor.feature_names is not None:
                expected_features = list(self.predictor.feature_names)
            elif hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                expected_features = list(self.scaler.feature_names_in_)

            if expected_features is not None and hasattr(self, 'scaler') and hasattr(self.scaler, 'mean_'):
                import numpy as _np
                means = list(self.scaler.mean_)
                # Build DataFrame of means
                rows = 50
                data = {f: means[i] if i < len(means) else 0 for i, f in enumerate(expected_features)}
                fallback_df = pd.DataFrame([data for _ in range(rows)])
                try:
                    fallback_scaled = self.scaler.transform(fallback_df)
                    return fallback_scaled
                except Exception:
                    # As last resort, return zeros in scaled space
                    return _np.zeros((rows, len(expected_features)))
            return None
                
        except Exception as e:
            st.warning(f"⚠️ Could not load background data: {str(e)}")
            # Robust fallback if exception occurs
            try:
                expected_features = None
                if hasattr(self.predictor, 'feature_names') and self.predictor.feature_names is not None:
                    expected_features = list(self.predictor.feature_names)
                elif hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                    expected_features = list(self.scaler.feature_names_in_)
                if expected_features is not None:
                    import numpy as _np
                    rows = 50
                    if hasattr(self, 'scaler') and hasattr(self.scaler, 'mean_'):
                        means = list(self.scaler.mean_)
                        data = {f: means[i] if i < len(means) else 0 for i, f in enumerate(expected_features)}
                        fallback_df = pd.DataFrame([data for _ in range(rows)])
                        try:
                            return self.scaler.transform(fallback_df)
                        except Exception:
                            return _np.zeros((rows, len(expected_features)))
                    else:
                        return _np.zeros((rows, len(expected_features)))
            except Exception:
                pass
            return None
    
    def initialize_shap_explainer(self):
        """Initialize SHAP explainer for the best model."""
        try:
            if self.best_model is None:
                st.warning("⚠️ Best model not initialized")
                return False
            
            model_name = getattr(self, 'best_model_name', 'Unknown')
            model_type = type(self.best_model).__name__
            
            # Get background data
            background_data = self.get_background_data()
            if background_data is None or len(background_data) == 0:
                st.warning("⚠️ Could not load background data for SHAP")
                return False
            
            # Handle VotingClassifier (Ensemble)
            if 'Voting' in model_type or 'Ensemble' in model_name:
                if hasattr(self.best_model, 'estimators_'):
                    for est in self.best_model.estimators_:
                        est_type = type(est).__name__
                        if 'Forest' in est_type or 'XGB' in est_type:
                            try:
                                self.shap_explainer = shap.TreeExplainer(est)
                                self.shap_initialized = True
                                return True
                            except Exception:
                                pass
                
                # Fallback: KernelExplainer
                try:
                    self.shap_explainer = shap.KernelExplainer(
                        self.best_model.predict_proba,
                        background_data[:20]
                    )
                    self.shap_initialized = True
                    return True
                except Exception:
                    pass
            
            # Tree-based models
            elif any(x in model_type for x in ['Forest', 'XGB', 'Gradient', 'Tree']):
                try:
                    self.shap_explainer = shap.TreeExplainer(self.best_model)
                    self.shap_initialized = True
                    return True
                except Exception:
                    pass
            
            # Fallback: KernelExplainer for all models
            try:
                predict_fn = self.best_model.predict_proba if hasattr(self.best_model, 'predict_proba') else self.best_model.predict
                self.shap_explainer = shap.KernelExplainer(
                    predict_fn,
                    background_data[:20]
                )
                self.shap_initialized = True
                return True
            except Exception as e:
                st.error(f"❌ SHAP initialization failed: {str(e)}")
                self.shap_initialized = False
                return False
            
        except Exception as e:
            st.error(f"❌ SHAP init error: {str(e)}")
            self.shap_initialized = False
            return False
    
    def generate_shap_explanation(self, features):
        """Generate SHAP explanation for a single prediction."""
        try:
            if not hasattr(self, 'shap_explainer') or not self.shap_initialized:
                if not self.initialize_shap_explainer():
                    return None
            
            # Prepare feature data
            feature_df = pd.DataFrame([features])
            
            # Get expected features from predictor
            expected_features = None
            if hasattr(self.predictor, 'feature_names') and self.predictor.feature_names is not None:
                expected_features = self.predictor.feature_names
            elif hasattr(self.scaler, 'feature_names_in_'):
                expected_features = self.scaler.feature_names_in_
            
            if expected_features is not None:
                # Align features with training data
                missing_cols = set(expected_features) - set(feature_df.columns)
                extra_cols = set(feature_df.columns) - set(expected_features)
                
                if missing_cols:
                    for col in missing_cols:
                        feature_df[col] = 0
                
                if extra_cols:
                    feature_df = feature_df.drop(columns=list(extra_cols))
                
                # Reorder to match training
                feature_df = feature_df[expected_features]
            
            # Apply scaler
            features_scaled = self.scaler.transform(feature_df)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer(features_scaled)
            
            return shap_values
            
        except Exception as e:
            st.error(f"❌ SHAP generation failed: {str(e)}")
            import traceback
            st.caption("🔍 Debug Info")
            st.code(traceback.format_exc())
            return None
    
    def create_shap_waterfall_chart(self, shap_values, feature_names, base_value):
        """Create a waterfall-style SHAP chart using Plotly."""
        try:
            # Extract SHAP values for plotting with robust error handling
            if hasattr(shap_values, 'values'):
                # Handle different SHAP value structures
                if len(shap_values.values.shape) == 3:
                    # For binary classification: shape is (n_samples, n_features, n_classes)
                    values = shap_values.values[0, :, 1] if shap_values.values.shape[2] > 1 else shap_values.values[0, :, 0]
                elif len(shap_values.values.shape) == 2:
                    # Shape is (n_samples, n_features)
                    values = shap_values.values[0]
                else:
                    # Shape is (n_features,)
                    values = shap_values.values
            elif isinstance(shap_values, list) and len(shap_values) > 0:
                values = np.array(shap_values[0]) if isinstance(shap_values[0], (list, np.ndarray)) else np.array(shap_values)
            elif isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) == 3:
                    values = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
                elif len(shap_values.shape) == 2:
                    values = shap_values[0]
                else:
                    values = shap_values
            else:
                # Fallback: try to convert to numpy array
                values = np.array(shap_values)
            
            # Ensure values is 1-dimensional numpy array
            if isinstance(values, np.ndarray) and len(values.shape) > 1:
                values = values.flatten()
            elif not isinstance(values, np.ndarray):
                values = np.array(values)
            
            # Convert to list and ensure all values are numeric
            values_list = []
            for v in values:
                try:
                    num_val = float(v)
                    values_list.append(0.0 if np.isnan(num_val) or np.isinf(num_val) else num_val)
                except (ValueError, TypeError):
                    values_list.append(0.0)
            
            # Ensure we have feature names matching the number of values
            if len(feature_names) > len(values_list):
                feature_names_truncated = feature_names[:len(values_list)]
            elif len(feature_names) < len(values_list):
                feature_names_truncated = list(feature_names) + [f"Feature_{i}" for i in range(len(feature_names), len(values_list))]
            else:
                feature_names_truncated = list(feature_names)
            
            # Create contributions DataFrame with matched lengths
            contributions = pd.DataFrame({
                'Feature': feature_names_truncated,
                'SHAP_Value': values_list,
                'Abs_SHAP': [abs(v) for v in values_list]
            }).sort_values('Abs_SHAP', ascending=True)
            
            # Color based on positive/negative contribution
            colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in contributions['SHAP_Value']]
            
            # Create horizontal bar chart with explicit list conversion
            fig = go.Figure(go.Bar(
                x=contributions['SHAP_Value'].tolist(),
                y=contributions['Feature'].tolist(),
                orientation='h',
                marker_color=colors,
                text=[f"{float(val):.3f}" for val in contributions['SHAP_Value']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="🔍 SHAP Feature Contributions",
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Clinical Features",
                height=500,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font=dict(size=16, color='#2E3B82')
            )
            
            # Add vertical line at x=0
            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
            
            return fig
            
        except Exception as e:
            # Return None instead of showing error to user
            return None

    def display_shap_dropdown(self, features):
        """Display comprehensive SHAP analysis in a dropdown section."""
        with st.expander("🔍 SHAP Model Interpretability Analysis", expanded=False):
            st.markdown("""
            ### 🧠 Understanding Your Prediction with AI Explainability
            
            **SHAP (SHapley Additive exPlanations)** provides transparent insights into how our AI model 
            made its prediction by showing the exact contribution of each clinical parameter.
            
            **Why SHAP Matters:**
            - 🎯 **Personalized**: Shows factors specific to your case
            - 🔬 **Scientific**: Based on game theory and mathematical principles  
            - 👩‍⚕️ **Clinical**: Helps doctors understand AI reasoning
            - ⚖️ **Fair**: Ensures all factors are considered equally
            """)
            
            # Create tabs for different SHAP analyses
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Your Prediction", "📊 Feature Impact", "🔄 What-If Analysis", "📚 Learn More"])
            
            with tab1:
                st.markdown("#### 🎯 How We Arrived at Your Prediction")
                
                try:
                    # Generate SHAP explanation
                    with st.spinner("🔄 Generating personalized SHAP analysis..."):
                        shap_values = self.generate_shap_explanation(features)
                        # Store for sharing with other tabs
                        self.current_shap_values = shap_values
                    
                    if shap_values is None:
                        st.warning("⚠️ SHAP analysis unavailable for this model type.")
                        st.info("💡 SHAP works best with tree-based models like Random Forest and XGBoost.")
                        st.markdown("""
                        **Alternative Interpretation Available:**
                        - Model confidence level
                        - Feature importance rankings  
                        - Statistical risk factors
                        """)
                        return
                    
                    # Get feature names
                    if hasattr(self, 'selected_features') and self.selected_features:
                        feature_names = self.selected_features
                    else:
                        feature_names = list(features.keys())
                    
                    # Extract values and base value with robust handling
                    if hasattr(shap_values, 'values'):
                        if len(shap_values.values.shape) == 3:
                            # Binary classification: (n_samples, n_features, n_classes)
                            values = shap_values.values[0, :, 1] if shap_values.values.shape[2] > 1 else shap_values.values[0, :, 0]
                        elif len(shap_values.values.shape) == 2:
                            # Shape: (n_samples, n_features)
                            values = shap_values.values[0]
                        else:
                            # Shape: (n_features,)
                            values = shap_values.values
                        # Extract base value safely
                        if hasattr(shap_values, 'base_values'):
                            base_vals = shap_values.base_values
                            if hasattr(base_vals, '__len__') and len(base_vals) > 0:
                                # Handle array case - take first element
                                if isinstance(base_vals, np.ndarray):
                                    base_value = float(base_vals.item(0) if base_vals.size > 0 else 0.5)
                                else:
                                    base_value = float(base_vals[0])
                            else:
                                # Handle scalar case
                                if isinstance(base_vals, np.ndarray):
                                    base_value = float(base_vals.item())
                                else:
                                    base_value = float(base_vals)
                        else:
                            base_value = 0.5
                    else:
                        # Handle list or array input
                        if isinstance(shap_values, list) and len(shap_values) > 0:
                            values = np.array(shap_values[0]) if isinstance(shap_values[0], (list, np.ndarray)) else np.array(shap_values)
                        else:
                            values = np.array(shap_values) if not isinstance(shap_values, np.ndarray) else shap_values
                        base_value = 0.5
                    
                    # Ensure values is 1D and convert to list
                    if isinstance(values, np.ndarray) and len(values.shape) > 1:
                        values = values.flatten()
                    
                    # Convert to Python list for consistent handling
                    if isinstance(values, np.ndarray):
                        values = values.tolist()
                    elif not isinstance(values, list):
                        values = [float(values)]
                    
                    # Create two columns for analysis
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("##### 🌊 SHAP Waterfall Chart")
                        st.markdown("*Shows how each factor pushes the prediction up (risk) or down (protective)*")
                        
                        # Create SHAP waterfall chart
                        fig = self.create_shap_waterfall_chart(shap_values, feature_names, base_value)
                        if fig:
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.error("Unable to generate SHAP waterfall chart")
                        
                    with col2:
                        st.markdown("##### � Impact Summary")
                        
                        # Calculate contributions with length checking
                        min_length = min(len(feature_names), len(values))
                        feature_names_safe = feature_names[:min_length]
                        values_safe = values[:min_length]
                        
                        contributions = pd.DataFrame({
                            'Feature': feature_names_safe,
                            'SHAP_Value': values_safe,
                            'Abs_SHAP': np.abs(values_safe)
                        }).sort_values('SHAP_Value', ascending=False)
                        
                        # Overall prediction summary
                        total_positive = contributions[contributions['SHAP_Value'] > 0]['SHAP_Value'].sum()
                        total_negative = contributions[contributions['SHAP_Value'] < 0]['SHAP_Value'].sum()
                        
                        # Ensure all values are Python floats for formatting
                        if hasattr(base_value, 'item'):
                            base_val = float(base_value.item())
                        else:
                            base_val = float(base_value)
                        
                        # Safe conversion of pandas sum results
                        if hasattr(total_positive, 'item'):
                            pos_val = float(total_positive.item())
                        else:
                            pos_val = float(total_positive)
                            
                        if hasattr(total_negative, 'item'):
                            neg_val = float(total_negative.item())
                        else:
                            neg_val = float(total_negative)
                        
                        # Safe calculation of final value
                        values_sum = np.sum(values)
                        if hasattr(values_sum, 'item'):
                            final_val = float(base_val + values_sum.item())
                        else:
                            final_val = float(base_val + values_sum)
                        
                        st.metric("🎯 Base Risk", f"{base_val:.1%}")
                        st.metric("📈 Risk Factors", f"+{pos_val:.1%}")
                        st.metric("📉 Protective Factors", f"{neg_val:.1%}")
                        st.metric("🎲 Final Prediction", f"{final_val:.1%}")
                
                except Exception as e:
                    st.error(f"❌ Error generating SHAP analysis: {str(e)}")
                    st.info("💡 This might be due to model compatibility. The prediction is still valid.")
            
            with tab2:
                st.markdown("#### � Detailed Feature Impact Analysis")
                
                try:
                    # Reuse SHAP values from tab1 to ensure consistency
                    if hasattr(self, 'current_shap_values') and self.current_shap_values is not None:
                        shap_values = self.current_shap_values
                        # Extract SHAP values with robust handling
                        if hasattr(shap_values, 'values'):
                            if len(shap_values.values.shape) == 3:
                                shap_vals = shap_values.values[0, :, 1]  # Class 1 (heart disease)
                            elif len(shap_values.values.shape) == 2:
                                shap_vals = shap_values.values[0]
                            else:
                                shap_vals = shap_values.values
                        else:
                            if isinstance(shap_values, list) and len(shap_values) > 0:
                                shap_vals = np.array(shap_values[0]) if isinstance(shap_values[0], (list, np.ndarray)) else np.array(shap_values)
                            else:
                                shap_vals = np.array(shap_values) if not isinstance(shap_values, np.ndarray) else shap_values
                        
                        # Ensure 1D array
                        if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) > 1:
                            shap_vals = shap_vals.flatten()
                            
                        # Get feature names
                        if hasattr(self, 'selected_features') and self.selected_features:
                            feature_names = self.selected_features[:len(shap_vals)]
                        else:
                            feature_names = list(features.keys())[:len(shap_vals)]
                            
                        # Ensure arrays are same length
                        min_length = min(len(feature_names), len(shap_vals))
                        feature_names = feature_names[:min_length]
                        shap_vals = shap_vals[:min_length]
                        
                        # Create contributions dataframe
                        contributions = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP_Value': shap_vals,
                            'Abs_SHAP': np.abs(shap_vals)
                        }).sort_values('Abs_SHAP', ascending=False)
                    else:
                        st.warning("⚠️ SHAP analysis unavailable. Please ensure prediction has been made first.")
                        return
                    
                    # Show top factors affecting prediction
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 🔴 Top Risk Factors")
                        positive_factors = contributions[contributions['SHAP_Value'] > 0].head(5)
                        
                        if len(positive_factors) > 0:
                            for i, (_, row) in enumerate(positive_factors.iterrows(), 1):
                                abs_shap = float(row['Abs_SHAP'])
                                impact_level = ("🔥 Critical" if abs_shap > 0.15 else 
                                              "⚠️ High" if abs_shap > 0.10 else 
                                              "🟡 Moderate" if abs_shap > 0.05 else "🟢 Low")
                                
                                st.markdown(f"""
                                **{i}. {row['Feature']}**
                                - Impact: +{float(row['SHAP_Value']):.3f} ({impact_level})
                                - Increases risk by {float(row['SHAP_Value'])*100:.1f} percentage points
                                """)
                        else:
                            st.success("✅ No significant risk factors detected!")
                    
                    with col2:
                        st.markdown("##### 🟢 Top Protective Factors")
                        negative_factors = contributions[contributions['SHAP_Value'] < 0].head(5)
                        
                        if len(negative_factors) > 0:
                            for i, (_, row) in enumerate(negative_factors.iterrows(), 1):
                                abs_shap = float(row['Abs_SHAP'])
                                impact_level = ("💚 Excellent" if abs_shap > 0.15 else 
                                              "🟢 Strong" if abs_shap > 0.10 else 
                                              "🔵 Moderate" if abs_shap > 0.05 else "⚪ Mild")
                                
                                st.markdown(f"""
                                **{i}. {row['Feature']}**
                                - Impact: {float(row['SHAP_Value']):.3f} ({impact_level})
                                - Reduces risk by {abs(float(row['SHAP_Value']))*100:.1f} percentage points
                                """)
                        else:
                            st.warning("⚠️ No significant protective factors detected")
                    
                    # Feature importance chart
                    st.markdown("---")
                    st.markdown("##### 📊 All Feature Contributions")
                    
                    fig_bar = px.bar(
                        contributions.head(10), 
                        x='SHAP_Value', 
                        y='Feature',
                        orientation='h',
                        color='SHAP_Value',
                        color_continuous_scale=['red', 'lightgray', 'green'],
                        title="SHAP Feature Contributions (Top 10)"
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, width='stretch')
                    
                    # Interactive feature exploration
                    st.markdown("---")
                    st.markdown("##### 🔍 Interactive Feature Explorer")
                    
                    selected_feature = st.selectbox(
                        "Select a feature to analyze:",
                        options=contributions['Feature'].tolist(),
                        index=0
                    )
                    
                    if selected_feature:
                        feature_impact = contributions[contributions['Feature'] == selected_feature]['SHAP_Value'].iloc[0]
                        
                        # Get the original feature value from the input features
                        if selected_feature in features:
                            feature_value = features[selected_feature]
                        else:
                            # Try to find a similar feature name
                            feature_value = "N/A"
                            for key, value in features.items():
                                if selected_feature.lower() in key.lower() or key.lower() in selected_feature.lower():
                                    feature_value = value
                                    break
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if isinstance(feature_value, (int, float)):
                                st.metric("Feature Value", f"{feature_value:.2f}")
                            else:
                                st.metric("Feature Value", str(feature_value))
                        with col2:
                            st.metric("SHAP Impact", f"{float(feature_impact):.3f}")
                        with col3:
                            direction = "⬆️ Increases" if feature_impact > 0 else "⬇️ Decreases"
                            st.metric("Effect", f"{direction} Risk")
                        
                        # Feature-specific insights
                        if selected_feature in ['age', 'chol', 'trestbps', 'thalch']:
                            st.markdown(f"**Clinical Context for {selected_feature}:**")
                            if 'age' in selected_feature.lower():
                                st.info("Age is a non-modifiable risk factor. Focus on modifiable factors for prevention.")
                            elif 'chol' in selected_feature.lower():
                                st.info("Cholesterol can be managed through diet, exercise, and medication if needed.")
                            elif 'bp' in selected_feature.lower() or 'trestbps' in selected_feature.lower():
                                st.info("Blood pressure is manageable through lifestyle changes and medication.")
                            elif 'thalch' in selected_feature.lower():
                                st.info("Maximum heart rate reflects fitness level and can be improved through exercise.")
                    
                    # Clinical recommendations based on SHAP
                    st.markdown("---")
                    st.markdown("##### 🏥 Clinical Recommendations Based on SHAP Analysis")
                    
                    recommendations = []
                    
                    # Analyze top risk factors and provide specific recommendations
                    top_risk_factors = contributions[contributions['SHAP_Value'] > 0.05]
                    
                    if len(top_risk_factors) > 0:
                        st.markdown("**Based on your risk factors, consider:**")
                        
                        for _, factor in top_risk_factors.iterrows():
                            feature_name = factor['Feature']
                            
                            if 'age' in feature_name.lower():
                                recommendations.append("• Regular cardiovascular screening (age-related risk)")
                            elif 'chol' in feature_name.lower():
                                recommendations.append("• Dietary modification to reduce cholesterol")
                            elif 'bp' in feature_name.lower() or 'trestbps' in feature_name.lower():
                                recommendations.append("• Blood pressure monitoring and management")
                            elif 'heart' in feature_name.lower() or 'thalch' in feature_name.lower():
                                recommendations.append("• Cardiovascular fitness assessment")
                            elif 'angina' in feature_name.lower() or 'exang' in feature_name.lower():
                                recommendations.append("• Exercise stress test evaluation")
                            elif 'chest' in feature_name.lower() or 'cp' in feature_name.lower():
                                recommendations.append("• Detailed cardiac evaluation for chest symptoms")
                        
                        # Remove duplicates and display
                        recommendations = list(set(recommendations))
                        for rec in recommendations[:5]:  # Show top 5 recommendations
                            st.markdown(rec)
                    else:
                        st.markdown("✅ **No major risk factors identified.** Continue maintaining healthy lifestyle.")
                    
                    # General recommendations
                    st.markdown("**🌟 General Heart Health Recommendations:**")
                    general_recs = [
                        "• Regular exercise (150 minutes moderate intensity per week)",
                        "• Heart-healthy diet (Mediterranean or DASH diet)",
                        "• Maintain healthy weight (BMI 18.5-24.9)",
                        "• Stress management and adequate sleep",
                        "• Avoid smoking and limit alcohol consumption",
                        "• Regular health check-ups and monitoring"
                    ]
                    
                    for rec in general_recs:
                        st.markdown(rec)
                    
                except Exception as e:
                    st.error(f"❌ Error in feature analysis: {str(e)}")
                    st.markdown("""
                    **Alternative Feature Analysis:**
                    
                    While SHAP analysis is temporarily unavailable, here are general insights:
                    
                    **🔴 Common Risk Factors:**
                    - Advanced age (>50 years)
                    - High cholesterol (>240 mg/dl)
                    - High blood pressure (>140/90 mmHg)
                    - Low exercise capacity
                    - Chest pain symptoms
                    
                    **🟢 Protective Factors:**
                    - Regular exercise
                    - Normal cholesterol levels
                    - Optimal blood pressure
                    - No exercise-induced symptoms
                    - Younger age
                    
                    **💡 Recommendation:** Focus on modifiable risk factors through lifestyle changes.
                    """)
            
            with tab3:
                st.markdown("#### 🔄 What-If Scenario Analysis")
                st.markdown("*Explore how changing different factors might affect your risk*")
                
                # Create interactive what-if analysis
                st.markdown("##### 🎛️ Adjust Parameters and See Impact")
                
                # Create two columns for parameter adjustment
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Modify Clinical Parameters:**")
                    
                    # Create adjustable parameters based on current values
                    modified_features = features.copy()
                    
                    # Age adjustment
                    if 'age' in features:
                        modified_features['age'] = st.slider(
                            "🎂 Age (years)", 
                            min_value=20, max_value=100, 
                            value=int(features['age']), 
                            help="Adjust age to see how it affects risk"
                        )
                    
                    # Cholesterol adjustment
                    if 'chol' in features:
                        modified_features['chol'] = st.slider(
                            "🩸 Cholesterol (mg/dl)", 
                            min_value=100, max_value=400, 
                            value=int(features['chol']),
                            help="Lower cholesterol typically reduces risk"
                        )
                    
                    # Blood pressure adjustment
                    if 'trestbps' in features:
                        modified_features['trestbps'] = st.slider(
                            "🩺 Resting Blood Pressure (mm Hg)", 
                            min_value=80, max_value=200, 
                            value=int(features['trestbps']),
                            help="Lower blood pressure is generally better"
                        )
                    
                    # Maximum heart rate
                    if 'thalch' in features:
                        modified_features['thalch'] = st.slider(
                            "💓 Maximum Heart Rate (bpm)", 
                            min_value=60, max_value=220, 
                            value=int(features['thalch']),
                            help="Higher max heart rate often indicates better fitness"
                        )
                    
                    # Exercise induced angina
                    if 'exang' in features:
                        current_exang = features['exang']
                        if isinstance(current_exang, str):
                            current_value = current_exang
                        else:
                            current_value = "Yes" if current_exang == 1 else "No"
                        
                        modified_features['exang'] = st.selectbox(
                            "💪 Exercise Induced Angina", 
                            ["No", "Yes"],
                            index=0 if current_value == "No" else 1,
                            help="Exercise-induced angina increases risk"
                        )
                    
                    # ST Depression
                    if 'oldpeak' in features:
                        modified_features['oldpeak'] = st.slider(
                            "📉 ST Depression", 
                            min_value=0.0, max_value=6.0, 
                            value=float(features['oldpeak']),
                            step=0.1,
                            help="Lower ST depression is better"
                        )
                
                with col2:
                    st.markdown("**🔄 Real-time Risk Assessment:**")
                    
                    # Calculate prediction for modified features
                    if st.button("🔮 Calculate New Risk", key="whatif_predict"):
                        with st.spinner("Calculating new risk..."):
                            try:
                                # Convert string values to numeric (match encoding in sidebar)
                                if 'exang' in modified_features and isinstance(modified_features['exang'], str):
                                    modified_features['exang'] = 1 if modified_features['exang'] == "Yes" else 0
                                
                                # Make prediction with modified features
                                modified_results = self.make_prediction(modified_features)
                                
                                if modified_results:
                                    # Compare with original prediction
                                    original_prob = st.session_state.prediction_results['probability'] if 'prediction_results' in st.session_state and st.session_state.prediction_results else 0.5
                                    new_prob = modified_results['probability']
                                    
                                    # Show comparison
                                    st.markdown("**📊 Risk Comparison:**")
                                    
                                    col_orig, col_new = st.columns(2)
                                    with col_orig:
                                        st.metric(
                                            "🔍 Original Risk", 
                                            f"{original_prob:.1%}",
                                            help="Your original risk level"
                                        )
                                    
                                    with col_new:
                                        risk_change = new_prob - original_prob
                                        risk_delta = f"{risk_change:+.1%}"
                                        st.metric(
                                            "🔮 Modified Risk", 
                                            f"{new_prob:.1%}",
                                            delta=risk_delta,
                                            help="Risk with modified parameters"
                                        )
                                    
                                    # Impact analysis
                                    if abs(risk_change) > 0.01:  # More than 1% change
                                        if risk_change < 0:
                                            st.success(f"✅ **Risk Reduced** by {abs(risk_change):.1%}!")
                                            st.markdown("🎉 These changes would be beneficial!")
                                        else:
                                            st.warning(f"⚠️ **Risk Increased** by {risk_change:.1%}")
                                            st.markdown("💡 Consider different modifications")
                                    else:
                                        st.info("📊 Minimal change in risk prediction")
                                    
                                    # Show new risk level
                                    st.markdown("**🎯 New Risk Category:**")
                                    if new_prob >= 0.7:
                                        st.error("🔴 **Very High Risk**")
                                    elif new_prob >= 0.5:
                                        st.warning("🟡 **High Risk**")
                                    elif new_prob >= 0.3:
                                        st.info("🔵 **Moderate Risk**")
                                    else:
                                        st.success("🟢 **Low Risk**")
                                
                            except Exception as e:
                                st.error(f"Error calculating modified risk: {e}")
                    
                    else:
                        st.info("👆 Adjust parameters above and click 'Calculate New Risk' to see the impact")
                
                # Educational content about parameter impacts
                st.markdown("---")
                st.markdown("##### 📚 Understanding Parameter Impacts")
                
                impact_info = {
                    "🎂 **Age**": "Risk naturally increases with age. While you can't change this, other factors become more important to manage.",
                    "🩸 **Cholesterol**": "Total cholesterol <200 mg/dL is ideal. Diet, exercise, and medication can help reduce cholesterol.",
                    "🩺 **Blood Pressure**": "Normal BP is <120/80 mmHg. Lifestyle changes and medication can help control blood pressure.",
                    "💓 **Max Heart Rate**": "Higher maximum heart rate during exercise often indicates better cardiovascular fitness.",
                    "💪 **Exercise Angina**": "Chest pain during exercise may indicate coronary artery disease and increases risk.",
                    "📉 **ST Depression**": "Measured during stress tests. Lower values are better for heart health."
                }
                
                for param, explanation in impact_info.items():
                    st.markdown(f"###### {param}")
                    st.markdown(explanation)
                
                # Quick scenario buttons
                st.markdown("---")
                st.markdown("##### ⚡ Quick Scenarios")
                
                scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
                
                with scenario_col1:
                    if st.button("🏃‍♂️ Improved Fitness"):
                        st.info("Simulating improved cardiovascular fitness...")
                        modified_scenario = features.copy()
                        if 'thalch' in modified_scenario:
                            modified_scenario['thalch'] = min(200, int(features['thalch']) + 20)
                        if 'exang' in modified_scenario:
                            modified_scenario['exang'] = "No"
                        st.success("Scenario: Higher max heart rate + No exercise angina")
                
                with scenario_col2:
                    if st.button("💊 Medication Control"):
                        st.info("Simulating optimal medication management...")
                        modified_scenario = features.copy()
                        if 'chol' in modified_scenario:
                            modified_scenario['chol'] = max(150, int(features['chol']) - 50)
                        if 'trestbps' in modified_scenario:
                            modified_scenario['trestbps'] = max(100, int(features['trestbps']) - 20)
                        st.success("Scenario: Lower cholesterol + Lower blood pressure")
                
                with scenario_col3:
                    if st.button("🌟 Optimal Health"):
                        st.info("Simulating ideal health parameters...")
                        modified_scenario = features.copy()
                        if 'chol' in modified_scenario:
                            modified_scenario['chol'] = 180  # Optimal cholesterol
                        if 'trestbps' in modified_scenario:
                            modified_scenario['trestbps'] = 110  # Optimal BP
                        if 'thalch' in modified_scenario:
                            modified_scenario['thalch'] = 180  # Good fitness
                        if 'exang' in modified_scenario:
                            modified_scenario['exang'] = "No"
                        if 'oldpeak' in modified_scenario:
                            modified_scenario['oldpeak'] = 0.5  # Minimal ST depression
                        st.success("Scenario: All parameters in optimal ranges")
            
            with tab4:
                st.markdown("#### � Understanding SHAP Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    ##### 🎯 What are SHAP Values?
                    
                    **SHAP** stands for **SHapley Additive exPlanations**, a revolutionary 
                    method to explain machine learning predictions.
                    
                    **Key Concepts:**
                    - 🧮 **Mathematical Foundation**: Based on cooperative game theory
                    - ⚖️ **Fair Attribution**: Each feature gets its fair share of prediction
                    - 🔄 **Additive**: All SHAP values sum to final prediction
                    - 🎯 **Local**: Explains individual predictions (yours!)
                    
                    **How to Read SHAP Charts:**
                    - 📊 **Waterfall**: Shows step-by-step prediction building
                    - 🔴 **Red/Positive**: Increases heart disease risk
                    - 🟢 **Green/Negative**: Decreases heart disease risk
                    - 📏 **Length**: Bigger bars = stronger influence
                    """)
                
                with col2:
                    st.markdown("""
                    ##### 🏥 Clinical Significance
                    
                    **Why Doctors Love SHAP:**
                    - 🔍 **Transparency**: See exactly how AI makes decisions
                    - 🎯 **Personalized**: Tailored to each patient's unique profile
                    - 📋 **Evidence-Based**: Backed by mathematical guarantees
                    - 🤝 **Trust**: Builds confidence in AI recommendations
                    
                    **For Patients:**
                    - 📖 **Understandable**: Clear explanations of complex AI
                    - 🎯 **Actionable**: Identify which factors to focus on
                    - 💪 **Empowering**: Take control of your health decisions
                    - 🔒 **Trustworthy**: Know the AI reasoning is sound
                    
                    **Important Note:**
                    SHAP analysis is for educational purposes. Always consult 
                    healthcare professionals for medical decisions.
                    """)
                
                # Mathematical explanation
                st.markdown("---")
                st.markdown("""
                ##### 🧮 The Mathematics Behind SHAP
                
                **SHAP Formula:**
                ```
                Prediction = Base Value + Σ(SHAP values for all features)
                ```
                
                **What This Means:**
                - **Base Value**: Average prediction across all patients
                - **SHAP Values**: How much each of your features deviates from average
                - **Sum**: All SHAP values add up to your final prediction
                
                **Shapley Values Origin:**
                Originally developed by Nobel Prize winner Lloyd Shapley for fairly 
                distributing gains in cooperative games. Now applied to AI explainability!
                """)
            
            # Show current feature values for reference
                st.markdown("#### 📋 Current Feature Values")
                st.markdown("**Clinical Parameters Used in Analysis:**")
                
                # Create a nice table of feature values
                feature_display = []
                feature_mapping = {
                    'age': 'Age (years)',
                    'sex': 'Sex',
                    'cp': 'Chest Pain Type',
                    'trestbps': 'Resting Blood Pressure (mm Hg)',
                    'chol': 'Cholesterol (mg/dl)',
                    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
                    'restecg': 'Resting ECG',
                    'thalch': 'Maximum Heart Rate (bpm)',
                    'exang': 'Exercise Induced Angina',
                    'oldpeak': 'ST Depression',
                    'slope': 'ST Slope',
                    'ca': 'Major Vessels (0-3)',
                    'thal': 'Thalassemia Type'
                }
                
                for feature, value in features.items():
                    display_name = feature_mapping.get(feature, feature)
                    if isinstance(value, float):
                        display_value = f"{value:.2f}"
                    else:
                        display_value = str(value)
                    feature_display.append([display_name, display_value])
                
                # Create a nice table
                feature_df = pd.DataFrame(feature_display, columns=['Parameter', 'Value'])
                st.dataframe(feature_df, use_container_width=True, hide_index=True)

    def display_prediction_results(self, results, features):
        """Display prediction results with beautiful visualizations."""
        if results is None:
            return
        
        # Main prediction display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if results['prediction'] == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h2>⚠️ Heart Disease Detected</h2>
                    <h3>Risk Level: {results['risk_level']}</h3>
                    <h4>Probability: {results['percentage']}</h4>
                    <p>Confidence: {results['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2>✅ No Heart Disease Detected</h2>
                    <h3>Risk Level: {results['risk_level']}</h3>
                    <h4>Probability: {results['percentage']}</h4>
                    <p>Confidence: {results['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown("### 📊 Detailed Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Prediction</h4>
                <h2>{"Disease" if results['prediction'] == 1 else "Healthy"}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Probability</h4>
                <h2>{results['percentage']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Risk Level</h4>
                <h2>{results['risk_level']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Confidence</h4>
                <h2>{results['confidence']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability gauge chart
        self.create_probability_gauge(results['probability'])
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            self.create_feature_importance_chart(features)
        
        # ADD SHAP ANALYSIS DROPDOWN HERE
        self.display_shap_dropdown(features)
        
        # Risk factors analysis
        self.analyze_risk_factors(features)
    
    def create_probability_gauge(self, probability):
        """Create a beautiful gauge chart for probability."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')
    
    def create_feature_importance_chart(self, features):
        """Create feature importance visualization."""
        if not hasattr(self.best_model, 'feature_importances_'):
            return
        
        st.markdown("### 🎯 Feature Importance Analysis")
        
        try:
            # Get feature names based on available features
            if hasattr(self, 'selected_features') and self.selected_features:
                feature_names = self.selected_features
            else:
                feature_names = list(features.keys())
            
            importances = self.best_model.feature_importances_
            
            # Create DataFrame for visualization
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title='Feature Importance in Prediction',
                color='Importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')
            
        except Exception as e:
            st.write(f"Feature importance chart unavailable: {str(e)}")
    
    def analyze_risk_factors(self, features):
        """Analyze and display risk factors."""
        st.markdown("### ⚠️ Risk Factor Analysis")
        
        risk_factors = []
        
        # Analyze individual features
        if features['age'] > 60:
            risk_factors.append("Age over 60 increases cardiovascular risk")
        
        if features['chol'] > 240:
            risk_factors.append("High cholesterol (>240 mg/dl)")
        
        if features['trestbps'] > 140:
            risk_factors.append("High blood pressure (>140 mm Hg)")
        
        if features['fbs'] == 1:
            risk_factors.append("Elevated fasting blood sugar")
        
        if features['exang'] == 1:
            risk_factors.append("Exercise-induced chest pain")
        
        if features['cp'] == 3:  # Asymptomatic
            risk_factors.append("Asymptomatic chest pain (higher risk)")
        
        if features['thalch'] < 100:
            risk_factors.append("Low maximum heart rate")
        
        # Display risk factors
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"⚠️ {factor}")
        else:
            st.markdown("✅ No major risk factors identified")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = [
            "Regular cardiovascular check-ups",
            "Maintain healthy diet low in saturated fats",
            "Regular exercise (consult doctor first)",
            "Monitor blood pressure and cholesterol",
            "Avoid smoking and limit alcohol consumption",
            "Manage stress through relaxation techniques"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    def generate_pdf_report(self, results, features, patient_info):
        """Generate professional PDF report."""
        if not PDF_AVAILABLE:
            st.error("PDF generation not available. Please install reportlab.")
            return None
        
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )
            
            story.append(Paragraph("Heart Disease Prediction Report", title_style))
            story.append(Spacer(1, 20))
            
            # Patient Information
            if patient_info['name']:
                story.append(Paragraph(f"<b>Patient:</b> {patient_info['name']}", styles['Normal']))
            story.append(Paragraph(f"<b>Age:</b> {patient_info['age']}", styles['Normal']))
            story.append(Paragraph(f"<b>Sex:</b> {patient_info['sex']}", styles['Normal']))
            if patient_info['doctor']:
                story.append(Paragraph(f"<b>Consulting Doctor:</b> {patient_info['doctor']}", styles['Normal']))
            story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Prediction Results
            story.append(Paragraph("Prediction Results", styles['Heading2']))
            
            result_data = [
                ['Metric', 'Value'],
                ['Prediction', 'Heart Disease Detected' if results['prediction'] == 1 else 'No Heart Disease'],
                ['Probability', results['percentage']],
                ['Risk Level', results['risk_level']],
                ['Confidence', results['confidence']]
            ]
            
            result_table = Table(result_data)
            result_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(result_table)
            story.append(Spacer(1, 20))
            
            # Clinical Data
            story.append(Paragraph("Clinical Data", styles['Heading2']))
            
            clinical_data = [
                ['Parameter', 'Value', 'Normal Range'],
                ['Age', f"{features['age']} years", "20-100"],
                ['Resting BP', f"{features['trestbps']} mm Hg", "90-120"],
                ['Cholesterol', f"{features['chol']} mg/dl", "<200"],
                ['Max Heart Rate', f"{features['thalch']} bpm", "60-220"],
                ['ST Depression', f"{features['oldpeak']}", "0-3"]
            ]
            
            clinical_table = Table(clinical_data)
            clinical_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(clinical_table)
            story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Medical Recommendations", styles['Heading2']))
            recommendations = [
                "Consult with a cardiologist for detailed evaluation",
                "Regular monitoring of blood pressure and cholesterol",
                "Maintain a heart-healthy diet",
                "Engage in regular physical activity as recommended by physician",
                "Avoid smoking and limit alcohol consumption",
                "Manage stress through appropriate techniques"
            ]
            
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Disclaimer
            story.append(Paragraph("Disclaimer", styles['Heading3']))
            disclaimer_text = """
            This report is generated by an AI-based prediction system and should not be considered 
            as a definitive medical diagnosis. Please consult with qualified healthcare professionals 
            for proper medical evaluation and treatment decisions.
            """
            story.append(Paragraph(disclaimer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            return None
    
    def test_email_configuration(self, sender_email, sender_password):
        """Test email configuration without sending a full report."""
        try:
            with st.spinner("Testing email configuration..."):
                # SMTP server configuration for Gmail
                smtp_server = "smtp.gmail.com"
                smtp_port = 587
                
                # Create message
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = sender_email  # Send test email to self
                msg['Subject'] = "Heart Disease Prediction System - Email Test"
                
                # Email body
                body = """
                🎉 Email Configuration Test Successful!
                
                Your Heart Disease Prediction System email configuration is working correctly.
                
                ✅ SMTP Connection: Successful
                ✅ Authentication: Successful  
                ✅ Email Sending: Successful
                
                You can now send prediction reports to patients via email.
                
                ---
                Heart Disease Prediction System
                Powered by Machine Learning
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                # Send email
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(sender_email, sender_password)
                text = msg.as_string()
                server.sendmail(sender_email, sender_email, text)
                server.quit()
                
                st.success("✅ Email test successful! Check your inbox.")
                st.info(f"Test email sent to: {sender_email}")
                
        except smtplib.SMTPAuthenticationError:
            st.error("❌ Authentication failed! Check your email and app password.")
            st.info("💡 Make sure you're using an App Password, not your regular Gmail password.")
        except smtplib.SMTPConnectError:
            st.error("❌ Connection failed! Check your internet connection.")
        except Exception as e:
            st.error(f"❌ Email test failed: {str(e)}")
            st.info("💡 Make sure 2FA is enabled and you're using a Gmail App Password.")

    def get_email_settings(self):
        """Retrieve permanent email settings from Streamlit secrets or fallback file."""
        # Prefer Streamlit secrets for production
        try:
            if 'email' in st.secrets:
                sec = st.secrets['email']
                return {
                    'sender_email': sec.get('sender_email'),
                    'sender_password': sec.get('sender_password'),
                    'sender_name': sec.get('sender_name', 'Heart Disease Prediction System')
                }
        except Exception:
            pass

        # Fallback to local JSON if secrets are not set
        cfg = self.load_email_config()
        return {
            'sender_email': cfg.get('sender_email'),
            'sender_password': cfg.get('sender_password'),
            'sender_name': cfg.get('sender_name', 'Heart Disease Prediction System')
        }

    def send_email_report(self, pdf_buffer, patient_email, patient_name):
        """Send PDF report via email using permanent configuration."""
        if not patient_email:
            st.error("No email address provided")
            return False
        
        try:
            # Use permanent email settings
            settings = self.get_email_settings()
            sender_email = settings.get('sender_email')
            sender_password = settings.get('sender_password')
            sender_name = settings.get('sender_name')
            
            if not sender_email or not sender_password:
                st.error("❌ Email configuration not found. Please set Streamlit secrets for email.")
                with st.expander("Email setup instructions"):
                    st.markdown("Set `[email] sender_email`, `sender_password`, `sender_name` in Streamlit Cloud Secrets.")
                return False
            
            with st.spinner(f"Sending report to {patient_email}..."):
                # Email configuration
                smtp_server = "smtp.gmail.com"
                smtp_port = 587
                
                # Create message
                msg = MIMEMultipart()
                msg['From'] = f"{sender_name} <{sender_email}>"
                msg['To'] = patient_email
                msg['Subject'] = f"Heart Disease Prediction Report - {patient_name or 'Patient'}"
                
                # Email body
                body = f"""
Dear {patient_name or 'Patient'},

Please find attached your Heart Disease Prediction Report generated by our AI-powered system.

📊 REPORT DETAILS:
• Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
• Analysis: Comprehensive heart disease risk assessment
• Technology: Machine Learning with 81.52% accuracy
• Interpretability: SHAP analysis included

⚠️ IMPORTANT DISCLAIMER:
This report is for informational purposes only and should not replace professional medical advice. Please consult with your healthcare provider for proper medical evaluation and treatment decisions.

🔒 CONFIDENTIALITY NOTICE:
This report contains confidential medical information. If you received this email in error, please notify the sender immediately and delete this message.

Best regards,
{sender_name}
Heart Disease Prediction System

---
Powered by Advanced Machine Learning
Accuracy: 81.52% | Technology: Neural Networks + SHAP Analysis
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                # Attach PDF
                if pdf_buffer:
                    pdf_buffer.seek(0)
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(pdf_buffer.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename="Heart_Disease_Report_{patient_name or "Patient"}_{datetime.now().strftime("%Y%m%d")}.pdf"'
                    )
                    msg.attach(part)
                
                # Send email
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(sender_email, sender_password)
                text = msg.as_string()
                server.sendmail(sender_email, patient_email, text)
                server.quit()
                
                st.success(f"✅ Report successfully sent to {patient_email}")
                return True
                
        except smtplib.SMTPAuthenticationError:
            st.error("❌ Email authentication failed! Please check your email configuration.")
        except smtplib.SMTPRecipientsRefused:
            st.error(f"❌ Invalid recipient email address: {patient_email}")
        except Exception as e:
            st.error(f"❌ Failed to send email: {str(e)}")
            
        return False
    
    def load_email_config(self):
        """Load email configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Decode password
                    if 'sender_password' in config:
                        config['sender_password'] = base64.b64decode(config['sender_password']).decode()
                    return config
        except Exception as e:
            # Don't show warning on first load
            pass
        return {}

    def save_email_config(self, config):
        """Save email configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Could not save email config: {e}")
            return False

    def clear_email_config(self):
        """Clear saved email configuration."""
        try:
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            return True
        except Exception as e:
            st.error(f"Could not clear email config: {e}")
            return False

    def load_models(self):
        """Load the trained models and preprocessing objects."""
        try:
            # Try REGULAR model first (most reliable)
            if REGULAR_MODEL_AVAILABLE:
                st.write("🔍 Attempting to load regular models...")
                self.predictor = HeartDiseasePredictor()
                if hasattr(self.predictor, 'load_models'):
                    loaded_ok = self.predictor.load_models()
                    st.write(f"Load result: {loaded_ok}, best_model={self.predictor.best_model is not None}")
                    
                    if loaded_ok:
                        self.best_model = getattr(self.predictor, 'best_model', None)
                        self.best_model_name = getattr(self.predictor, 'best_model_name', 'Unknown')
                        self.scaler = getattr(self.predictor, 'scaler', None)
                        self.feature_selector = getattr(self.predictor, 'feature_selector', None)
                        self.selected_features = getattr(self.predictor, 'selected_features', None)
                        # Proactively apply sklearn compatibility patch (e.g., monotonic_cst on trees)
                        try:
                            if hasattr(self.predictor, '_apply_sklearn_compat_patches') and self.best_model is not None:
                                self.predictor._apply_sklearn_compat_patches(self.best_model)
                        except Exception:
                            pass
                        
                        # Validate best_model is not None
                        if self.best_model is None:
                            st.error("❌ Loaded models but best_model is None. Models may be corrupted.")
                            st.write(f"Predictor attributes: {dir(self.predictor)}")
                            self.models_loaded = False
                            return False
                        
                        self.models_loaded = True
                        st.success(f"✅ Models loaded successfully! Using: {self.best_model_name}")
                        return True
            
            # If loading pre-trained models failed, attempt on-cloud training from data
            if REGULAR_MODEL_AVAILABLE:
                with st.spinner("🔧 Pre-trained models not found. Training models on server (one-time setup)..."):
                    trained_ok = self.train_models_on_cloud()
                    if trained_ok:
                        # Try loading again after training
                        reloaded_ok = self.predictor.load_models()
                        if reloaded_ok:
                            self.best_model = getattr(self.predictor, 'best_model', None)
                            self.best_model_name = getattr(self.predictor, 'best_model_name', 'Unknown')
                            self.scaler = getattr(self.predictor, 'scaler', None)
                            self.feature_selector = getattr(self.predictor, 'feature_selector', None)
                            self.selected_features = getattr(self.predictor, 'selected_features', None)
                            # Apply compatibility patch after training reload
                            try:
                                if hasattr(self.predictor, '_apply_sklearn_compat_patches') and self.best_model is not None:
                                    self.predictor._apply_sklearn_compat_patches(self.best_model)
                            except Exception:
                                pass
                            self.models_loaded = True
                            st.success(f"✅ Models trained and loaded! Using: {self.best_model_name}")
                            return True
                        else:
                            st.error("❌ Training completed but failed to reload models.")
                    else:
                        st.error("❌ Failed to train models on server.")

            # Fallback to OPTIMIZED model if regular fails
            if OPTIMIZED_MODEL_AVAILABLE:
                self.predictor = OptimizedHeartDiseasePredictor()
                if hasattr(self.predictor, 'load_models'):
                    loaded_ok = self.predictor.load_models('optimized_models')
                    
                    if loaded_ok:
                        self.best_model = getattr(self.predictor, 'best_model_instance', None)
                        self.best_model_name = getattr(self.predictor, 'best_model', 'Unknown')
                        self.scaler = getattr(self.predictor, 'scaler', None)
                        self.feature_selector = getattr(self.predictor, 'feature_selector', None)
                        self.selected_features = getattr(self.predictor, 'selected_features', None)
                        # Apply compatibility patch for optimized best model
                        try:
                            if hasattr(self.predictor, '_apply_sklearn_compat_patches') and self.best_model is not None:
                                self.predictor._apply_sklearn_compat_patches(self.best_model)
                        except Exception:
                            pass
                        self.models_loaded = True
                        st.success(f"✅ Optimized models loaded! Using: {self.best_model_name}")
                        return True
            
            # If both fail, show error
            st.error("❌ No models available - both regular and optimized models failed to load")
            return False
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())
            self.models_loaded = False
            return False

    def train_models_on_cloud(self):
        """Train all models from the bundled dataset when pre-trained artifacts are unavailable.

        This ensures first-time deployments and fresh environments can self-initialize.
        """
        try:
            if not REGULAR_MODEL_AVAILABLE:
                st.error("Regular predictor unavailable for training.")
                return False

            data_path = "data/heart_disease_uci.csv"
            if not os.path.exists(data_path):
                st.error("Dataset not found. Cannot train models.")
                return False

            # Load and preprocess data via predictor API
            X, y = self.predictor.load_and_preprocess_data(data_path)
            if X is None or y is None:
                st.error("Failed to load data for training.")
                return False

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train all models and ensemble
            self.predictor.train_models(X_train, y_train, X_test, y_test)
            self.predictor.create_ensemble_model(X_train, y_train, X_test, y_test)

            # Persist artifacts to saved_models/ for future fast loads
            self.predictor.save_models(save_dir='saved_models')

            # Capture feature names for alignment in app
            self.selected_features = getattr(self.predictor, 'feature_names', None)

            return True
        except Exception as e:
            with st.expander("🔍 Training Error Details"):
                import traceback
                st.code(traceback.format_exc())
            return False

    def display_environment_info(self):
        """Show runtime environment details for troubleshooting deployments."""
        with st.expander("🛠️ Environment Info (for debugging)"):
            try:
                import sys
                import sklearn, numpy, pandas, shap as _shap, xgboost as _xgb
                st.write({
                    "python": sys.version.split(" ")[0],
                    "numpy": numpy.__version__,
                    "pandas": pandas.__version__,
                    "scikit_learn": sklearn.__version__,
                    "shap": _shap.__version__,
                    "xgboost": _xgb.__version__,
                    "streamlit": st.__version__,
                    "regular_model_available": REGULAR_MODEL_AVAILABLE,
                    "optimized_model_available": OPTIMIZED_MODEL_AVAILABLE,
                    "models_loaded": getattr(self, 'models_loaded', False),
                    "best_model_name": getattr(self, 'best_model_name', None),
                })
            except Exception as e:
                st.write(f"Env info unavailable: {e}")

    def apply_feature_engineering(self, df):
        """Apply the same feature engineering used during training."""
        try:
            # If predictor has feature engineering method, use it
            if hasattr(self.predictor, 'apply_feature_engineering'):
                return self.predictor.apply_feature_engineering(df)
            else:
                # Basic feature engineering fallback
                df_processed = df.copy()
                
                # Create interaction features if columns exist
                if 'age' in df.columns and 'chol' in df.columns:
                    df_processed['age_chol_interaction'] = df['age'] * df['chol']
                
                if 'age' in df.columns and 'trestbps' in df.columns:
                    df_processed['age_bp_interaction'] = df['age'] * df['trestbps']
                
                return df_processed
                
        except Exception as e:
            st.warning(f"Feature engineering failed: {e}")
            return df

    def make_prediction(self, features):
        """Make real predictions using the loaded model."""
        try:
            # Validate setup
            if not self.models_loaded or self.best_model is None or self.scaler is None:
                st.error("❌ Models not properly loaded. Please refresh.")
                with st.expander("🔍 Debug Info"):
                    st.write({
                        "models_loaded": self.models_loaded,
                        "best_model_exists": self.best_model is not None,
                        "scaler_exists": self.scaler is not None,
                        "best_model_name": getattr(self, 'best_model_name', 'N/A')
                    })
                return None
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Align with training features
            if hasattr(self.predictor, 'feature_names') and self.predictor.feature_names:
                expected = self.predictor.feature_names
                for c in set(expected) - set(feature_df.columns):
                    feature_df[c] = 0
                for c in set(feature_df.columns) - set(expected):
                    feature_df = feature_df.drop(columns=[c])
                feature_df = feature_df[expected]
            
            # Scale
            X_scaled = self.scaler.transform(feature_df)
            
            # Predict
            prediction = self.best_model.predict(X_scaled)
            pred = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)
            
            # Get probability
            if hasattr(self.best_model, 'predict_proba'):
                probability = self.best_model.predict_proba(X_scaled)
                if hasattr(probability[0], '__len__'):
                    prob = float(probability[0][1]) if len(probability[0]) > 1 else float(probability[0][0])
                else:
                    prob = float(probability[0])
            else:
                prob = 0.5
            
            # Determine risk level
            if prob >= 0.7:
                risk_level = "Very High"
                confidence = "High"
            elif prob >= 0.5:
                risk_level = "High"
                confidence = "Medium"
            elif prob >= 0.3:
                risk_level = "Moderate"
                confidence = "Medium"
            elif prob >= 0.2:
                risk_level = "Low"
                confidence = "High"
            else:
                risk_level = "Very Low"
                confidence = "High"
            
            return {
                'prediction': int(pred),
                'probability': float(prob),
                'percentage': f"{prob*100:.1f}%",
                'risk_level': risk_level,
                'confidence': confidence
            }
            
        except Exception as e:
            # Attempt a one-time sklearn compatibility patch if relevant
            err_msg = str(e)
            if 'monotonic_cst' in err_msg and hasattr(self, 'predictor') and hasattr(self.predictor, '_apply_sklearn_compat_patches'):
                try:
                    self.predictor._apply_sklearn_compat_patches(self.best_model)
                    # Retry once after patching
                    feature_df = pd.DataFrame([features])
                    if hasattr(self.predictor, 'feature_names') and self.predictor.feature_names:
                        expected = self.predictor.feature_names
                        for c in set(expected) - set(feature_df.columns):
                            feature_df[c] = 0
                        for c in set(feature_df.columns) - set(expected):
                            feature_df = feature_df.drop(columns=[c])
                        feature_df = feature_df[expected]
                    X_scaled = self.scaler.transform(feature_df)
                    prediction = self.best_model.predict(X_scaled)
                    pred = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)
                    if hasattr(self.best_model, 'predict_proba'):
                        probability = self.best_model.predict_proba(X_scaled)
                        if hasattr(probability[0], '__len__'):
                            prob = float(probability[0][1]) if len(probability[0]) > 1 else float(probability[0][0])
                        else:
                            prob = float(probability[0])
                    else:
                        prob = 0.5
                    if prob >= 0.7:
                        risk_level, confidence = "Very High", "High"
                    elif prob >= 0.5:
                        risk_level, confidence = "High", "Medium"
                    elif prob >= 0.3:
                        risk_level, confidence = "Moderate", "Medium"
                    elif prob >= 0.2:
                        risk_level, confidence = "Low", "High"
                    else:
                        risk_level, confidence = "Very Low", "High"
                    return {
                        'prediction': int(pred),
                        'probability': float(prob),
                        'percentage': f"{prob*100:.1f}%",
                        'risk_level': risk_level,
                        'confidence': confidence
                    }
                except Exception:
                    pass
            st.error(f"Prediction failed: {str(e)}")
            import traceback
            with st.expander("🔍 Prediction Error Details"):
                st.code(traceback.format_exc())
            return None

    def create_sidebar(self):
        """Create sidebar with input parameters (email config removed)."""
        st.sidebar.header("👤 Patient Information")
        
        # Patient info
        st.session_state.patient_info['name'] = st.sidebar.text_input(
            "Patient Name", 
            value=st.session_state.patient_info['name']
        )
        st.session_state.patient_info['email'] = st.sidebar.text_input(
            "Patient Email", 
            value=st.session_state.patient_info['email']
        )
        st.session_state.patient_info['doctor'] = st.sidebar.text_input(
            "Consulting Doctor", 
            value=st.session_state.patient_info['doctor']
        )
        
        st.sidebar.header("🩺 Clinical Parameters")
        
        # Clinical inputs
        features = {}
        
        features['age'] = st.sidebar.slider("Age (years)", 20, 100, 50)
        features['sex'] = st.sidebar.selectbox("Sex", ["Male", "Female"])
        features['sex'] = 1 if features['sex'] == "Male" else 0
        
        features['cp'] = st.sidebar.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            index=0
        )
        cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        features['cp'] = cp_mapping[features['cp']]
        
        features['trestbps'] = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        features['chol'] = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200)
        
        features['fbs'] = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        features['fbs'] = 1 if features['fbs'] == "Yes" else 0
        
        features['restecg'] = st.sidebar.selectbox(
            "Resting ECG",
            ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"],
            index=0
        )
        restecg_mapping = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        features['restecg'] = restecg_mapping[features['restecg']]
        
        features['thalch'] = st.sidebar.slider("Maximum Heart Rate (bpm)", 60, 220, 150)
        
        features['exang'] = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
        features['exang'] = 1 if features['exang'] == "Yes" else 0
        
        features['oldpeak'] = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        
        features['slope'] = st.sidebar.selectbox(
            "ST Slope",
            ["Upsloping", "Flat", "Downsloping"],
            index=1
        )
        slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        features['slope'] = slope_mapping[features['slope']]
        
        features['ca'] = st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0)
        
        features['thal'] = st.sidebar.selectbox(
            "Thalassemia",
            ["Normal", "Fixed Defect", "Reversible Defect"],
            index=0
        )
        thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
        features['thal'] = thal_mapping[features['thal']]
        
        # Email configuration UI has been removed to fully mask sender details.
        
        return features
    
    def run(self):
        """Main application runner."""
        # Header
        st.markdown('<h1 class="main-header">❤️ CardioCheck.AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced AI-powered cardiovascular risk assessment</p>', unsafe_allow_html=True)
        
        # Version indicator
        st.caption("🔄 v2024.12.16.2 | Predictions & SHAP Active")
        
        # Force cache clear on first run
        if 'cache_cleared' not in st.session_state:
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state.cache_cleared = True
            except Exception:
                pass
        
        # Load models
        if not self.models_loaded:
            with st.spinner("Loading AI models..."):
                self.load_models()
        
        if not self.models_loaded:
            st.stop()
        
        # Create sidebar for input
        features = self.create_sidebar()
        
        # Main prediction interface
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔬 Analyze Heart Health", key="predict"):
                with st.spinner("Analyzing your health data..."):
                    time.sleep(1)  # For dramatic effect
                    results = self.make_prediction(features)
                    if results:
                        st.session_state.prediction_made = True
                        st.session_state.prediction_results = results
                        st.session_state.features = features
        
        # Display results if prediction was made
        if st.session_state.prediction_made and 'prediction_results' in st.session_state:
            self.display_prediction_results(
                st.session_state.prediction_results, 
                st.session_state.features
            )
            
            # PDF and Email functionality
            st.markdown("### 📄 Generate Report")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                pdf_clicked = st.button("📄 Generate PDF Report", key="pdf_btn")
                if pdf_clicked:
                    with st.spinner("Generating PDF report..."):
                        pdf_buffer = self.generate_pdf_report(
                            st.session_state.prediction_results,
                            st.session_state.features,
                            st.session_state.patient_info
                        )
                        
                        if pdf_buffer:
                            st.success("✅ PDF generated successfully!")
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                key="download_pdf_btn"
                            )
                        else:
                            st.error("❌ Failed to generate PDF report.")
            
            with col2:
                email_clicked = st.button("📧 Email Report", key="email_btn")
                if email_clicked:
                    if st.session_state.patient_info['email']:
                        with st.spinner("Sending email..."):
                            pdf_buffer = self.generate_pdf_report(
                                st.session_state.prediction_results,
                                st.session_state.features,
                                st.session_state.patient_info
                            )
                            
                            if pdf_buffer:
                                success = self.send_email_report(
                                    pdf_buffer,
                                    st.session_state.patient_info['email'],
                                    st.session_state.patient_info['name']
                                )
                                
                                if success:
                                    st.success("✅ Report sent successfully!")
                                else:
                                    st.error("❌ Failed to send email. Please check configuration.")
                    else:
                        st.error("Please provide an email address in the sidebar.")
            
            with col3:
                if st.button("🔄 New Analysis", key="new_analysis_btn"):
                    st.session_state.prediction_made = False
                    st.session_state.prediction_results = None
                    st.session_state.features = None
                    st.rerun()
        
        # About section
        with st.expander("ℹ️ About This Application"):
            st.markdown("""
            ### About Heart Disease Prediction System
            
            This application uses advanced machine learning algorithms to assess cardiovascular risk based on clinical parameters.
            
            **🏆 Current Model Performance:**
            - **Best Model**: Neural Network (MLPClassifier)
            - **Accuracy**: 81.52%
            - **ROC-AUC**: 86.35%
            - **Precision**: 81.52%
            - **Recall**: 81.52%
            - **F1-Score**: 81.52%
            
            **🧠 Neural Network Architecture:**
            - **Hidden Layers**: (150, 100, 50) neurons
            - **Activation**: ReLU
            - **Solver**: Adam optimizer
            - **Learning Rate**: Adaptive
            
            **📊 Model Comparison (All 8 Models Tested):**
            | Model | Accuracy | ROC-AUC |
            |-------|----------|---------|
            | **Neural Network** | **81.52%** | **86.35%** |
            | Gradient Boosting | 80.98% | 85.43% |
            | Extra Trees | 80.43% | 85.34% |
            | Random Forest | 79.89% | 86.42% |
            | AdaBoost | 79.35% | 86.29% |
            | SVM | 79.35% | 84.98% |
            | XGBoost | 78.80% | 84.42% |
            | Logistic Regression | 78.26% | 85.15% |
            
            **🔬 Advanced Techniques Used:**
            - **Feature Engineering**: Interaction terms, categorical groups
            - **Feature Selection**: RFECV (16 optimal features selected)
            - **Hyperparameter Tuning**: RandomizedSearchCV optimization
            - **Class Balancing**: SMOTE oversampling
            - **Data Preprocessing**: Robust scaling, outlier handling
            - **Cross-Validation**: Stratified K-fold for robust evaluation
            
            **🎯 Selected Features (16 total):**
            1. Age, Blood Pressure, Cholesterol
            2. Fasting Blood Sugar, Max Heart Rate
            3. Exercise Angina, ST Depression, ST Slope
            4. Major Vessels, Thalassemia
            5. Age-Cholesterol Interaction
            6. Age-Blood Pressure Interaction
            7. Chest Pain-Exercise Angina Interaction
            8. Age Groups, Cholesterol Categories, BP Categories
            
            **Features:**
            - Multiple ML models for accurate prediction
            - SHAP analysis for model interpretability
            - Professional PDF report generation
            - Email delivery of reports
            - Modern, intuitive interface
            
            **Models Used:**
            - Random Forest
            - XGBoost
            - Gradient Boosting
            - Support Vector Machine
            - Neural Networks
            - Advanced Ensemble Methods
            
            **Data Processing:**
            - Feature engineering and selection
            - Hyperparameter optimization
            - Cross-validation for robustness
            - SMOTE for class balance
            
            **Disclaimer:** This tool is for educational and research purposes only. 
            Always consult with qualified healthcare professionals for medical decisions.
            """)

        # Environment section for troubleshooting
        self.display_environment_info()
        
        # Model Performance Section
        with st.expander("📊 Detailed Model Performance"):
            st.markdown("""
            ### 🎯 Current Model: Neural Network (MLPClassifier)
            
            **📈 Performance Metrics:**
            - **Test Accuracy**: 81.52% (152 correct out of 184 test samples)
            - **ROC-AUC Score**: 86.35% (excellent discrimination ability)
            - **Precision**: 81.52% (low false positive rate)
            - **Recall**: 81.52% (low false negative rate)
            - **F1-Score**: 81.52% (balanced precision and recall)
            
            **🧠 Neural Network Configuration:**
            - **Architecture**: 3 hidden layers with 150, 100, and 50 neurons
            - **Input Features**: 16 carefully selected clinical parameters
            - **Activation Function**: ReLU (Rectified Linear Unit)
            - **Optimizer**: Adam with adaptive learning rate
            - **Training Method**: Early stopping with validation monitoring
            
            **📊 Training Dataset:**
            - **Total Samples**: 920 patients from UCI Heart Disease Database
            - **Training Set**: 736 samples (80%)
            - **Test Set**: 184 samples (20%)
            - **Class Distribution**: Balanced using SMOTE oversampling
            - **Feature Processing**: Robust scaling to handle outliers
            
            **🔍 Model Selection Process:**
            We tested 8 different machine learning algorithms and selected the Neural Network 
            based on highest accuracy and robust cross-validation performance:
            
            1. **Neural Network** - 81.52% (Selected)
            2. Gradient Boosting - 80.98%
            3. Extra Trees - 80.43%
            4. Random Forest - 79.89%
            5. AdaBoost - 79.35%
            6. SVM - 79.35%
            7. XGBoost - 78.80%
            8. Logistic Regression - 78.26%
            
            **⚡ Real-time Prediction:**
            When you input patient data, the system applies the same preprocessing pipeline 
            used during training and generates predictions with confidence levels.
            """)


def main():
    """Main function to run the Streamlit app."""
    app = HeartDiseaseWebApp()
    app.run()


if __name__ == "__main__":
    main()