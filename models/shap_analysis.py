"""
SHAP Analysis for Heart Disease Prediction Models
================================================

This module provides comprehensive SHAP (SHapley Additive exPlanations) analysis
for model interpretability and feature importance analysis.

Author: Heart Disease Prediction Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

warnings.filterwarnings('ignore')

class SHAPAnalyzer:
    """
    SHAP analysis for heart disease prediction models.
    Provides comprehensive interpretability analysis including:
    - Feature importance rankings
    - SHAP summary plots
    - Waterfall plots for individual predictions
    - Partial dependence plots
    - Feature interaction analysis
    """
    
    def __init__(self, model=None, X_train=None, X_test=None, feature_names=None):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained ML model
            X_train: Training features
            X_test: Testing features  
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, model_type='tree'):
        """
        Create SHAP explainer based on model type.
        
        Args:
            model_type (str): Type of explainer ('tree', 'linear', 'kernel', 'deep')
        """
        print(f"🔍 Creating SHAP {model_type} explainer...")
        
        try:
            if model_type == 'tree':
                # For tree-based models (Random Forest, XGBoost, etc.)
                self.explainer = shap.TreeExplainer(self.model)
                
            elif model_type == 'linear':
                # For linear models (Logistic Regression, SVM with linear kernel)
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
                
            elif model_type == 'kernel':
                # For any model (slower but works with any model)
                # Use a sample of training data for background
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
                
            elif model_type == 'deep':
                # For neural networks
                self.explainer = shap.DeepExplainer(self.model, self.X_train)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            print("✅ SHAP explainer created successfully")
            
        except Exception as e:
            print(f"❌ Error creating SHAP explainer: {str(e)}")
            print("🔄 Falling back to KernelExplainer...")
            
            # Fallback to KernelExplainer
            try:
                background = shap.sample(self.X_train, min(50, len(self.X_train)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
                print("✅ Fallback explainer created successfully")
            except Exception as e2:
                print(f"❌ Fallback explainer also failed: {str(e2)}")
                return False
                
        return True
    
    def calculate_shap_values(self, sample_size=None):
        """
        Calculate SHAP values for the test set.
        
        Args:
            sample_size (int): Number of samples to analyze (None for all)
        """
        if self.explainer is None:
            print("❌ No explainer available. Create explainer first.")
            return
        
        print("🧮 Calculating SHAP values...")
        
        # Use subset of test data if specified
        if sample_size and sample_size < len(self.X_test):
            test_sample = self.X_test.sample(n=sample_size, random_state=42)
        else:
            test_sample = self.X_test
            
        try:
            # Calculate SHAP values
            if hasattr(self.explainer, 'shap_values'):
                # For older SHAP versions or specific explainers
                self.shap_values = self.explainer.shap_values(test_sample)
                
                # Handle binary classification case
                if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                    self.shap_values = self.shap_values[1]  # Use positive class
                    
            else:
                # For newer SHAP versions
                self.shap_values = self.explainer(test_sample)
                
            print(f"✅ SHAP values calculated for {len(test_sample)} samples")
            return True
            
        except Exception as e:
            print(f"❌ Error calculating SHAP values: {str(e)}")
            return False
    
    def plot_summary(self, plot_type='dot', max_display=20):
        """
        Create SHAP summary plot.
        
        Args:
            plot_type (str): Type of plot ('dot', 'bar', 'violin')
            max_display (int): Maximum number of features to display
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Calculate SHAP values first.")
            return
        
        print(f"📊 Creating SHAP summary plot ({plot_type})...")
        
        plt.figure(figsize=(12, 8))
        
        try:
            if hasattr(self.shap_values, 'values'):
                # For newer SHAP versions (Explanation objects)
                shap.summary_plot(
                    self.shap_values.values, 
                    self.shap_values.data,
                    feature_names=self.feature_names,
                    plot_type=plot_type,
                    max_display=max_display,
                    show=False
                )
            else:
                # For older SHAP versions (numpy arrays)
                shap.summary_plot(
                    self.shap_values, 
                    self.X_test,
                    feature_names=self.feature_names,
                    plot_type=plot_type,
                    max_display=max_display,
                    show=False
                )
                
            plt.title(f'SHAP Summary Plot - {plot_type.title()}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating summary plot: {str(e)}")
    
    def plot_feature_importance(self, max_display=20):
        """
        Create SHAP feature importance plot.
        
        Args:
            max_display (int): Maximum number of features to display
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Calculate SHAP values first.")
            return
        
        print("📊 Creating SHAP feature importance plot...")
        
        try:
            plt.figure(figsize=(10, 8))
            
            if hasattr(self.shap_values, 'values'):
                # For newer SHAP versions
                shap.plots.bar(self.shap_values, max_display=max_display, show=False)
            else:
                # For older SHAP versions
                shap.summary_plot(
                    self.shap_values, 
                    self.X_test,
                    feature_names=self.feature_names,
                    plot_type='bar',
                    max_display=max_display,
                    show=False
                )
                
            plt.title('Feature Importance (SHAP Values)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating feature importance plot: {str(e)}")
    
    def plot_waterfall(self, sample_idx=0):
        """
        Create waterfall plot for individual prediction.
        
        Args:
            sample_idx (int): Index of sample to analyze
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Calculate SHAP values first.")
            return
        
        print(f"📊 Creating waterfall plot for sample {sample_idx}...")
        
        try:
            plt.figure(figsize=(12, 8))
            
            if hasattr(self.shap_values, 'values'):
                # For newer SHAP versions
                shap.plots.waterfall(self.shap_values[sample_idx], show=False)
            else:
                # For older SHAP versions
                shap.waterfall_plot(
                    shap.Explanation(
                        values=self.shap_values[sample_idx],
                        base_values=self.explainer.expected_value,
                        data=self.X_test.iloc[sample_idx].values,
                        feature_names=self.feature_names
                    ),
                    show=False
                )
                
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating waterfall plot: {str(e)}")
    
    def plot_force_plot(self, sample_idx=0):
        """
        Create force plot for individual prediction.
        
        Args:
            sample_idx (int): Index of sample to analyze
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Calculate SHAP values first.")
            return
        
        print(f"📊 Creating force plot for sample {sample_idx}...")
        
        try:
            if hasattr(self.shap_values, 'values'):
                # For newer SHAP versions
                shap.plots.force(self.shap_values[sample_idx])
            else:
                # For older SHAP versions
                shap.force_plot(
                    self.explainer.expected_value,
                    self.shap_values[sample_idx],
                    self.X_test.iloc[sample_idx],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.show()
                
        except Exception as e:
            print(f"❌ Error creating force plot: {str(e)}")
    
    def plot_partial_dependence(self, feature_idx=0):
        """
        Create partial dependence plot for a specific feature.
        
        Args:
            feature_idx (int): Index of feature to analyze
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Calculate SHAP values first.")
            return
        
        feature_name = self.feature_names[feature_idx] if self.feature_names else f"Feature_{feature_idx}"
        print(f"📊 Creating partial dependence plot for {feature_name}...")
        
        try:
            plt.figure(figsize=(10, 6))
            
            if hasattr(self.shap_values, 'values'):
                # For newer SHAP versions
                shap.plots.partial_dependence(
                    feature_idx, 
                    self.model.predict, 
                    self.X_test, 
                    ice=False,
                    model_expected_value=True,
                    feature_expected_value=True,
                    show=False
                )
            else:
                # Manual partial dependence plot
                feature_values = self.X_test.iloc[:, feature_idx]
                shap_values_feature = self.shap_values[:, feature_idx]
                
                plt.scatter(feature_values, shap_values_feature, alpha=0.6)
                plt.xlabel(feature_name)
                plt.ylabel('SHAP Value')
                plt.title(f'Partial Dependence Plot - {feature_name}')
                
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating partial dependence plot: {str(e)}")
    
    def analyze_feature_interactions(self, max_display=10):
        """
        Analyze and visualize feature interactions.
        
        Args:
            max_display (int): Maximum number of interaction pairs to display
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Calculate SHAP values first.")
            return
        
        print("🔍 Analyzing feature interactions...")
        
        try:
            # Calculate interaction values
            if hasattr(self.explainer, 'shap_interaction_values'):
                interaction_values = self.explainer.shap_interaction_values(
                    self.X_test.head(min(100, len(self.X_test)))
                )
                
                # Create interaction summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    interaction_values, 
                    self.X_test.head(min(100, len(self.X_test))),
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False
                )
                plt.title('Feature Interaction Analysis', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                
            else:
                print("⚠️  Feature interaction analysis not available for this explainer type")
                
        except Exception as e:
            print(f"❌ Error analyzing feature interactions: {str(e)}")
    
    def get_top_features(self, n_features=10):
        """
        Get top n most important features based on SHAP values.
        
        Args:
            n_features (int): Number of top features to return
            
        Returns:
            pandas.DataFrame: Top features with their importance scores
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Calculate SHAP values first.")
            return None
        
        try:
            if hasattr(self.shap_values, 'values'):
                # For newer SHAP versions
                shap_vals = self.shap_values.values
            else:
                # For older SHAP versions
                shap_vals = self.shap_values
            
            # Calculate mean absolute SHAP values for each feature
            feature_importance = np.abs(shap_vals).mean(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(n_features)
            
        except Exception as e:
            print(f"❌ Error getting top features: {str(e)}")
            return None
    
    def generate_shap_report(self, save_plots=True, plots_dir='shap_plots'):
        """
        Generate comprehensive SHAP analysis report.
        
        Args:
            save_plots (bool): Whether to save plots to files
            plots_dir (str): Directory to save plots
        """
        print("\n📋 Generating Comprehensive SHAP Analysis Report")
        print("=" * 60)
        
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            print(f"📁 Plots will be saved to: {plots_dir}/")
        
        # 1. Feature Importance
        print("\n1️⃣ Feature Importance Analysis")
        top_features = self.get_top_features(n_features=10)
        if top_features is not None:
            print("\n🔝 Top 10 Most Important Features:")
            print(top_features.to_string(index=False, float_format='%.4f'))
        
        # 2. Summary Plots
        print("\n2️⃣ Creating Summary Plots...")
        self.plot_summary(plot_type='dot')
        if save_plots:
            plt.savefig(f'{plots_dir}/shap_summary_dot.png', dpi=300, bbox_inches='tight')
        
        self.plot_summary(plot_type='bar')
        if save_plots:
            plt.savefig(f'{plots_dir}/shap_summary_bar.png', dpi=300, bbox_inches='tight')
        
        # 3. Feature Importance Plot
        print("\n3️⃣ Creating Feature Importance Plot...")
        self.plot_feature_importance()
        if save_plots:
            plt.savefig(f'{plots_dir}/shap_feature_importance.png', dpi=300, bbox_inches='tight')
        
        # 4. Individual Prediction Analysis
        print("\n4️⃣ Individual Prediction Analysis...")
        for i in range(min(3, len(self.X_test))):
            print(f"   Analyzing sample {i}...")
            self.plot_waterfall(sample_idx=i)
            if save_plots:
                plt.savefig(f'{plots_dir}/shap_waterfall_sample_{i}.png', dpi=300, bbox_inches='tight')
        
        # 5. Partial Dependence Analysis
        if top_features is not None and len(top_features) > 0:
            print("\n5️⃣ Partial Dependence Analysis for Top Features...")
            for i, feature_name in enumerate(top_features['feature'].head(3)):
                feature_idx = self.feature_names.index(feature_name)
                self.plot_partial_dependence(feature_idx=feature_idx)
                if save_plots:
                    plt.savefig(f'{plots_dir}/shap_partial_dependence_{feature_name}.png', 
                              dpi=300, bbox_inches='tight')
        
        # 6. Feature Interactions
        print("\n6️⃣ Feature Interaction Analysis...")
        self.analyze_feature_interactions()
        if save_plots:
            plt.savefig(f'{plots_dir}/shap_feature_interactions.png', dpi=300, bbox_inches='tight')
        
        print("\n✅ SHAP Analysis Report Complete!")
        if save_plots:
            print(f"📊 All plots saved to: {plots_dir}/")


def run_shap_analysis_for_model(model, X_train, X_test, y_test, feature_names, model_name):
    """
    Run comprehensive SHAP analysis for a specific model.
    
    Args:
        model: Trained ML model
        X_train: Training features
        X_test: Test features
        y_test: Test targets
        feature_names: List of feature names
        model_name: Name of the model
    """
    print(f"\n🔍 Running SHAP Analysis for {model_name}")
    print("=" * 50)
    
    # Initialize SHAP analyzer
    analyzer = SHAPAnalyzer(model, X_train, X_test, feature_names)
    
    # Determine explainer type based on model
    explainer_type = 'kernel'  # Default fallback
    
    if hasattr(model, 'tree_'):
        explainer_type = 'tree'
    elif 'RandomForest' in str(type(model)) or 'XGB' in str(type(model)):
        explainer_type = 'tree'
    elif 'Logistic' in str(type(model)) or 'Linear' in str(type(model)):
        explainer_type = 'linear'
    elif 'MLP' in str(type(model)):
        explainer_type = 'kernel'  # Neural networks work better with kernel
    
    print(f"🎯 Using {explainer_type} explainer for {model_name}")
    
    # Create explainer
    if not analyzer.create_explainer(model_type=explainer_type):
        print(f"❌ Failed to create explainer for {model_name}")
        return None
    
    # Calculate SHAP values
    if not analyzer.calculate_shap_values(sample_size=200):  # Limit for performance
        print(f"❌ Failed to calculate SHAP values for {model_name}")
        return None
    
    # Generate comprehensive report
    plots_dir = f'shap_plots_{model_name.replace(" ", "_").lower()}'
    analyzer.generate_shap_report(save_plots=True, plots_dir=plots_dir)
    
    return analyzer


def main():
    """Main function to demonstrate SHAP analysis."""
    print("🚀 SHAP Analysis for Heart Disease Prediction")
    print("=" * 50)
    
    # This would typically be called from the main prediction script
    # For demonstration, we'll show how to use it
    
    print("📝 This module provides SHAP analysis capabilities.")
    print("🔧 To use it, import and call run_shap_analysis_for_model()")
    print("💡 Example usage:")
    print("""
    from models.shap_analysis import run_shap_analysis_for_model
    
    # After training your model
    analyzer = run_shap_analysis_for_model(
        model=your_trained_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        model_name="Random Forest"
    )
    """)


if __name__ == "__main__":
    main()