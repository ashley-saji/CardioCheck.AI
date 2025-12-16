# ❤️ Heart Disease Prediction Web Application

A comprehensive AI-powered heart disease prediction system with interactive web interface, advanced machine learning models, and professional reporting capabilities.

## 🌟 Features

### 🤖 Advanced Machine Learning
- **Multiple ML Models**: Random Forest, XGBoost, Gradient Boosting, SVM, Neural Networks, Extra Trees, AdaBoost
- **Hyperparameter Optimization**: GridSearchCV and RandomizedSearchCV for maximum accuracy
- **Feature Engineering**: Advanced feature creation, interaction terms, and categorical encoding
- **Class Balance**: SMOTE oversampling for balanced training data
- **Feature Selection**: Recursive Feature Elimination with Cross-Validation (RFECV)
- **Ensemble Methods**: Advanced voting classifiers and bagging techniques

### 🎯 Model Performance
- **Best Accuracy**: 81.52% with Neural Network model
- **Robust Evaluation**: Cross-validation, ROC-AUC analysis, precision, recall, F1-score
- **Binary Classification**: Optimized for disease vs. no disease prediction
- **Confidence Intervals**: Probability-based risk level assessment

### 🌐 Interactive Web Interface
- **Modern UI/UX**: Beautiful gradient design with responsive layout
- **Real-time Predictions**: Instant analysis with slider inputs
- **Risk Assessment**: Visual probability gauges and risk level indicators
- **Feature Importance**: Interactive charts showing model decision factors
- **Clinical Guidelines**: Risk factor analysis and medical recommendations

### 📊 Advanced Visualizations
- **Probability Gauges**: Interactive Plotly charts for disease likelihood
- **Feature Importance**: Horizontal bar charts showing key predictors
- **Risk Factor Analysis**: Clinical parameter evaluation
- **Model Comparison**: Performance metrics across all algorithms

### 📄 Professional Reporting
- **PDF Generation**: Comprehensive medical reports with ReportLab
- **Patient Information**: Customizable patient details and doctor information
- **Clinical Data**: Structured tables with normal ranges and current values
- **Medical Recommendations**: Evidence-based lifestyle and treatment suggestions
- **Professional Layout**: Clean, medical-grade document design

### 📧 Email Integration
- **SMTP Delivery**: Automatic email sending with PDF attachments
- **Patient Communication**: Direct report delivery to patient email addresses
- **Secure Transmission**: Professional email templates with disclaimers

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ (tested with Python 3.13.7)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "web app final"
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Training Models

1. **Run the optimized training script**:
   ```bash
   python models/heart_disease_prediction_optimized.py
   ```

2. **View training results**:
   - Models saved to `optimized_models/` directory
   - Best model automatically selected and saved
   - Performance metrics displayed in console

### Running the Web Application

1. **Start Streamlit server**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the application**:
   - Open browser to `http://localhost:8501`
   - Use the sidebar to input patient data
   - Click "Analyze Heart Health" for predictions

## 📁 Project Structure

```
web app final/
├── 📊 data/
│   └── heart_disease_uci.csv          # UCI Heart Disease dataset
├── 🤖 models/
│   ├── heart_disease_prediction_optimized.py  # Advanced ML pipeline
│   ├── heart_disease_prediction.py            # Original ML models
│   ├── model_evaluation.py                    # Evaluation utilities
│   └── shap_analysis.py                       # Model interpretability
├── 💾 optimized_models/                # Trained model artifacts
│   ├── best_model.joblib               # Best performing model
│   ├── scaler.joblib                   # Feature scaler
│   ├── feature_selector.joblib         # Feature selection
│   └── *_model.joblib                  # Individual trained models
├── 📊 notebooks/
│   └── exploration.ipynb               # Data exploration
├── 🖼️ shap_plots_*/                    # SHAP visualization outputs
├── 🌐 streamlit_app.py                 # Web application
├── 📋 requirements.txt                 # Python dependencies
└── 📖 README.md                        # This file
```

## 🔧 Configuration

### Email Setup (Optional)
To enable email functionality, configure SMTP settings in `streamlit_app.py`:

```python
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = "your-email@gmail.com"
sender_password = "your-app-password"
```

### Model Configuration
Adjust hyperparameters in `models/heart_disease_prediction_optimized.py`:
- Model parameters in `_initialize_optimized_models()`
- Feature engineering in `advanced_feature_engineering()`
- Evaluation metrics in `display_results()`

## 📊 Dataset Information

### UCI Heart Disease Dataset
- **Samples**: 920 patients
- **Features**: 16 clinical parameters
- **Target**: Multi-class (0-4) converted to binary (0/1)
- **Source**: UCI Machine Learning Repository

### Key Features
- **Age**: Patient age in years
- **Sex**: Male (1) or Female (0)
- **Chest Pain Type**: 4 categories of chest pain
- **Resting BP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol (mg/dl)
- **Fasting Blood Sugar**: >120 mg/dl (boolean)
- **Resting ECG**: Electrocardiogram results
- **Max Heart Rate**: Maximum heart rate achieved
- **Exercise Angina**: Exercise-induced angina (boolean)
- **ST Depression**: Exercise-induced depression
- **ST Slope**: Slope of peak exercise ST segment
- **Major Vessels**: Number of major vessels (0-3)
- **Thalassemia**: Blood disorder type

## 🎯 Model Performance

### Best Results (Neural Network)
- **Accuracy**: 81.52%
- **Precision**: 81.52%
- **Recall**: 81.52%
- **F1-Score**: 81.52%
- **ROC-AUC**: 86.35%

### Model Comparison
| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| Neural Network | 81.52% | 86.35% | 81.52% |
| Gradient Boosting | 80.98% | 85.43% | 80.94% |
| Extra Trees | 80.43% | 85.34% | 80.38% |
| Random Forest | 79.89% | 86.42% | 79.88% |
| AdaBoost | 79.35% | 86.29% | 79.40% |
| SVM | 79.35% | 84.98% | 79.39% |
| XGBoost | 78.80% | 84.42% | 78.72% |
| Logistic Regression | 78.26% | 85.15% | 78.32% |

## 🧪 Model Interpretability

### SHAP Analysis
- **Feature Importance**: Identifies key predictors
- **Waterfall Plots**: Individual prediction explanations
- **Summary Plots**: Global model behavior
- **Partial Dependence**: Feature impact visualization

### Risk Assessment
- **Very High Risk**: >80% probability
- **High Risk**: 60-80% probability
- **Moderate Risk**: 40-60% probability
- **Low Risk**: 20-40% probability
- **Very Low Risk**: <20% probability

## 🔬 Advanced Techniques

### Feature Engineering
- **Interaction Terms**: Age × Cholesterol, Age × BP, Chest Pain × Angina
- **Categorical Groups**: Age groups, cholesterol categories, BP categories
- **Outlier Handling**: IQR-based capping for continuous variables
- **Missing Value Imputation**: Median for numerical, mode for categorical

### Model Optimization
- **Hyperparameter Tuning**: RandomizedSearchCV for efficiency
- **Cross-Validation**: Stratified K-fold for robust evaluation
- **Feature Selection**: RFECV with Random Forest estimator
- **Class Balancing**: SMOTE oversampling for minority class
- **Ensemble Methods**: Voting classifiers with best models

## 🚨 Medical Disclaimer

This application is designed for educational and research purposes only. The predictions and recommendations provided should not be considered as professional medical advice, diagnosis, or treatment recommendations.

**Important Guidelines:**
- Always consult qualified healthcare professionals for medical decisions
- This tool is not a substitute for professional medical evaluation
- The AI models are trained on historical data and may not account for individual circumstances
- Regular medical check-ups are essential regardless of prediction results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the heart disease dataset
- **scikit-learn** for machine learning algorithms
- **Streamlit** for the web application framework
- **SHAP** for model interpretability
- **Plotly** for interactive visualizations
- **ReportLab** for PDF generation

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation and examples

---

**Built with ❤️ for advancing cardiovascular health through AI**