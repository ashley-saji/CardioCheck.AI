# Heart Disease Prediction Web App

A comprehensive machine learning system for predicting heart disease using multiple algorithms with SHAP interpretability analysis.

## Project Overview

This project implements and compares multiple machine learning models to achieve maximum accuracy in heart disease prediction. The system includes:

- **Multiple ML Models**: Random Forest, XGBoost, SVM, Neural Networks, and Ensemble methods
- **SHAP Analysis**: For model interpretability and feature importance
- **Comprehensive Evaluation**: Detailed performance metrics and comparisons
- **Data Preprocessing**: Feature engineering and data preparation

## Features

- 🤖 **Multiple ML Algorithms**: Implements 8+ different machine learning models
- 📊 **Model Comparison**: Side-by-side performance analysis
- 🔍 **SHAP Interpretability**: Understand model decisions and feature importance
- 📈 **Visualization**: Comprehensive plots for model performance and feature analysis
- 🎯 **Accuracy Optimization**: Hyperparameter tuning for maximum performance

## Models Implemented

1. **Random Forest Classifier**
2. **XGBoost Classifier**
3. **Support Vector Machine (SVM)**
4. **Neural Network (MLPClassifier)**
5. **Logistic Regression**
6. **Decision Tree Classifier**
7. **Gradient Boosting Classifier**
8. **Ensemble Voting Classifier**

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your heart disease dataset in the `data/` directory
2. Run the main analysis script:
```bash
python models/heart_disease_prediction.py
```
3. View results in the generated plots and console output

## Project Structure

```
/
├── .github/
│   └── copilot-instructions.md
├── models/
│   ├── heart_disease_prediction.py    # Main ML script
│   └── model_evaluation.py            # Evaluation utilities
├── data/
│   └── [your_dataset.csv]            # Heart disease dataset
├── notebooks/
│   └── exploration.ipynb             # Data exploration notebook
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- scikit-learn
- XGBoost
- SHAP
- pandas
- numpy
- matplotlib
- seaborn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.