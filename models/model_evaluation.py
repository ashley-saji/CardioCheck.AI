"""
Model Evaluation System for Heart Disease Prediction
==================================================

This module provides comprehensive model evaluation tools including:
- Advanced metrics calculation
- Cross-validation analysis
- ROC and Precision-Recall curves
- Learning curves
- Model comparison visualizations

Author: Heart Disease Prediction Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import (
    cross_val_score, validation_curve, learning_curve,
    StratifiedKFold
)
import warnings

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation system for heart disease prediction.
    """
    
    def __init__(self, models_dict, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Initialize the model evaluator.
        
        Args:
            models_dict (dict): Dictionary of trained models
            X_train, X_test: Training and testing features
            y_train, y_test: Training and testing targets
            feature_names (list): List of feature names
        """
        self.models_dict = models_dict
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.detailed_results = {}
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # Add precision, recall, f1 for each class
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            metrics[f'precision_class_{i}'] = p
            metrics[f'recall_class_{i}'] = r
            metrics[f'f1_class_{i}'] = f
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            try:
                # Ensure binary classification for ROC metrics
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                    metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
                else:
                    # Multi-class case - use macro average
                    if y_pred_proba.ndim > 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                    else:
                        metrics['roc_auc'] = np.nan
                    metrics['average_precision'] = np.nan
            except (ValueError, IndexError):
                # Handle case where y_true has only one class or other issues
                metrics['roc_auc'] = np.nan
                metrics['average_precision'] = np.nan
        
        return metrics
    
    def evaluate_all_models(self):
        """
        Evaluate all models with comprehensive metrics.
        """
        print("📊 Evaluating all models with comprehensive metrics...")
        
        for name, model in self.models_dict.items():
            if model is None:
                continue
                
            print(f"\n🔍 Evaluating {name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                metrics = self.calculate_comprehensive_metrics(
                    self.y_test, y_pred, y_pred_proba
                )
                
                # Store detailed results
                self.detailed_results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'metrics': metrics,
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                    'classification_report': classification_report(
                        self.y_test, y_pred, output_dict=True
                    )
                }
                
                print(f"✅ {name}: Accuracy = {metrics['accuracy']:.4f}, "
                      f"F1 = {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {name}: {str(e)}")
    
    def plot_roc_curves(self, figsize=(12, 8)):
        """
        Plot ROC curves for all models.
        """
        plt.figure(figsize=figsize)
        
        # Check if we have binary classification
        if len(np.unique(self.y_test)) != 2:
            print("⚠️  ROC curves are only supported for binary classification")
            return
        
        curves_plotted = False
        for name, results in self.detailed_results.items():
            if results['probabilities'] is not None and not np.isnan(results['metrics'].get('roc_auc', np.nan)):
                try:
                    fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
                    auc_score = results['metrics']['roc_auc']
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
                    curves_plotted = True
                except Exception as e:
                    print(f"⚠️  Could not plot ROC curve for {name}: {str(e)}")
        
        if curves_plotted:
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️  No ROC curves could be plotted")
    
    def plot_precision_recall_curves(self, figsize=(12, 8)):
        """
        Plot Precision-Recall curves for all models.
        """
        plt.figure(figsize=figsize)
        
        # Check if we have binary classification
        if len(np.unique(self.y_test)) != 2:
            print("⚠️  Precision-Recall curves are only supported for binary classification")
            return
        
        curves_plotted = False
        for name, results in self.detailed_results.items():
            if results['probabilities'] is not None and not np.isnan(results['metrics'].get('average_precision', np.nan)):
                try:
                    precision, recall, _ = precision_recall_curve(
                        self.y_test, results['probabilities']
                    )
                    ap_score = results['metrics']['average_precision']
                    plt.plot(recall, precision, label=f'{name} (AP = {ap_score:.3f})', linewidth=2)
                    curves_plotted = True
                except Exception as e:
                    print(f"⚠️  Could not plot PR curve for {name}: {str(e)}")
        
        if curves_plotted:
            # Baseline (random classifier)
            baseline = np.sum(self.y_test) / len(self.y_test)
            plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.6, 
                       label=f'Random Classifier (AP = {baseline:.3f})')
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️  No Precision-Recall curves could be plotted")
    
    def plot_detailed_confusion_matrices(self, figsize=(15, 10)):
        """
        Plot detailed confusion matrices for all models.
        """
        n_models = len(self.detailed_results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle('Detailed Confusion Matrices', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, results) in enumerate(self.detailed_results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = results['confusion_matrix']
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create combined labels (count and percentage)
            labels = np.array([[f'{count}\n({percent:.1f}%)' 
                              for count, percent in zip(row_counts, row_percents)]
                             for row_counts, row_percents in zip(cm, cm_percent)])
            
            sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'})
            
            ax.set_title(f'{name}\nAccuracy: {results["metrics"]["accuracy"]:.3f}')
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
    
    def plot_learning_curves(self, model_names=None, cv=5, figsize=(15, 10)):
        """
        Plot learning curves for specified models.
        
        Args:
            model_names (list): List of model names to plot (None for all)
            cv (int): Number of cross-validation folds
            figsize (tuple): Figure size
        """
        if model_names is None:
            model_names = list(self.models_dict.keys())
        
        models_to_plot = {name: self.models_dict[name] for name in model_names 
                         if name in self.models_dict and self.models_dict[name] is not None}
        
        n_models = len(models_to_plot)
        cols = 2
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle('Learning Curves', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1 and n_models > 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, model) in enumerate(models_to_plot.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col] if n_models > 1 else axes
            
            try:
                # Calculate learning curve
                train_sizes, train_scores, val_scores = learning_curve(
                    model, self.X_train, self.y_train,
                    cv=cv, n_jobs=-1, 
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='accuracy'
                )
                
                # Calculate mean and std
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                # Plot
                ax.plot(train_sizes, train_mean, 'o-', color='blue', 
                       label='Training Score')
                ax.fill_between(train_sizes, train_mean - train_std,
                               train_mean + train_std, alpha=0.1, color='blue')
                
                ax.plot(train_sizes, val_mean, 'o-', color='red', 
                       label='Validation Score')
                ax.fill_between(train_sizes, val_mean - val_std,
                               val_mean + val_std, alpha=0.1, color='red')
                
                ax.set_title(f'{name}')
                ax.set_xlabel('Training Set Size')
                ax.set_ylabel('Accuracy Score')
                ax.legend()
                ax.grid(alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{name} - Error')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            if rows > 1:
                row, col = idx // cols, idx % cols
                axes[row, col].axis('off')
            elif cols > n_models and n_models > 1:
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def cross_validate_models(self, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1']):
        """
        Perform cross-validation on all models.
        
        Args:
            cv (int): Number of cross-validation folds
            scoring (list): List of scoring metrics
        """
        print(f"\n🔄 Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        
        for name, model in self.models_dict.items():
            if model is None:
                continue
                
            print(f"   Cross-validating {name}...")
            
            try:
                model_cv_results = {}
                
                for metric in scoring:
                    scores = cross_val_score(
                        model, self.X_train, self.y_train,
                        cv=cv, scoring=metric, n_jobs=-1
                    )
                    
                    model_cv_results[metric] = {
                        'scores': scores,
                        'mean': np.mean(scores),
                        'std': np.std(scores)
                    }
                
                cv_results[name] = model_cv_results
                
            except Exception as e:
                print(f"❌ Error in cross-validation for {name}: {str(e)}")
        
        self.cv_results = cv_results
        return cv_results
    
    def plot_cross_validation_results(self, metric='accuracy'):
        """
        Plot cross-validation results for specified metric.
        
        Args:
            metric (str): Metric to plot
        """
        if not hasattr(self, 'cv_results'):
            print("❌ No cross-validation results available. Run cross_validate_models() first.")
            return
        
        models = []
        means = []
        stds = []
        
        for name, results in self.cv_results.items():
            if metric in results:
                models.append(name)
                means.append(results[metric]['mean'])
                stds.append(results[metric]['std'])
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot with error bars
        bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.8)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Cross-Validation Results - {metric.title()}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel(f'{metric.title()} Score')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive evaluation report.
        """
        print("\n📋 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 80)
        
        if not self.detailed_results:
            print("❌ No evaluation results available. Run evaluate_all_models() first.")
            return
        
        # 1. Overall Performance Summary
        print("\n1️⃣ OVERALL PERFORMANCE SUMMARY")
        print("-" * 40)
        
        summary_df = pd.DataFrame({
            'Model': list(self.detailed_results.keys()),
            'Accuracy': [r['metrics']['accuracy'] for r in self.detailed_results.values()],
            'Precision': [r['metrics']['precision'] for r in self.detailed_results.values()],
            'Recall': [r['metrics']['recall'] for r in self.detailed_results.values()],
            'F1-Score': [r['metrics']['f1_score'] for r in self.detailed_results.values()],
            'ROC-AUC': [r['metrics'].get('roc_auc', np.nan) for r in self.detailed_results.values()],
            'Matthews Corr': [r['metrics']['matthews_corrcoef'] for r in self.detailed_results.values()]
        }).sort_values('Accuracy', ascending=False)
        
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # 2. Best Model Analysis
        best_model_name = summary_df.iloc[0]['Model']
        best_results = self.detailed_results[best_model_name]
        
        print(f"\n2️⃣ BEST MODEL ANALYSIS: {best_model_name}")
        print("-" * 40)
        print(f"🎯 Accuracy: {best_results['metrics']['accuracy']:.4f}")
        print(f"📊 F1-Score: {best_results['metrics']['f1_score']:.4f}")
        print(f"🔍 Matthews Correlation: {best_results['metrics']['matthews_corrcoef']:.4f}")
        
        if best_results['metrics'].get('roc_auc'):
            print(f"📈 ROC-AUC: {best_results['metrics']['roc_auc']:.4f}")
        
        # 3. Classification Report
        print(f"\n3️⃣ DETAILED CLASSIFICATION REPORT - {best_model_name}")
        print("-" * 40)
        report_df = pd.DataFrame(best_results['classification_report']).transpose()
        print(report_df.to_string(float_format='%.4f'))
        
        # 4. Model Comparison Visualizations
        print("\n4️⃣ GENERATING VISUALIZATIONS...")
        print("-" * 40)
        
        # ROC Curves
        self.plot_roc_curves()
        
        # Precision-Recall Curves
        self.plot_precision_recall_curves()
        
        # Detailed Confusion Matrices
        self.plot_detailed_confusion_matrices()
        
        # Cross-validation if available
        if hasattr(self, 'cv_results'):
            print("\n5️⃣ CROSS-VALIDATION RESULTS")
            print("-" * 40)
            self.plot_cross_validation_results('accuracy')
            self.plot_cross_validation_results('f1')
        
        print("\n✅ COMPREHENSIVE EVALUATION COMPLETE!")
        return summary_df


def main():
    """Main function to demonstrate model evaluation."""
    print("🚀 Model Evaluation System for Heart Disease Prediction")
    print("=" * 60)
    
    print("📝 This module provides comprehensive model evaluation capabilities.")
    print("🔧 To use it, import and initialize with your trained models:")
    print("""
    from models.model_evaluation import ModelEvaluator
    
    # After training your models
    evaluator = ModelEvaluator(
        models_dict=your_models_dict,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names
    )
    
    # Run comprehensive evaluation
    evaluator.evaluate_all_models()
    evaluator.cross_validate_models()
    evaluator.generate_comprehensive_report()
    """)


if __name__ == "__main__":
    main()