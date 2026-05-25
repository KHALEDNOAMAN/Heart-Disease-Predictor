# 🫀 Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **A machine learning system that predicts the likelihood of heart disease using patient health data. Compares 5 ML algorithms and achieves 95% accuracy with Gradient Boosting.**

---

## 🎯 Overview

This project uses the UCI Heart Disease dataset (303 patients, 13 health features) to predict whether a patient has heart disease. It implements **5 different ML algorithms**, performs comprehensive **Exploratory Data Analysis (EDA)**, and generates **8 professional visualizations**.

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 83.6% | 83.3% | 68.2% | 75.0% |
| Random Forest | 80.3% | 81.3% | 59.1% | 68.4% |
| **Gradient Boosting** | **95.1%** | **95.2%** | **90.9%** | **93.0%** |
| SVM (RBF Kernel) | 82.0% | 86.7% | 59.1% | 70.3% |
| K-Nearest Neighbors | 73.8% | 66.7% | 54.6% | 60.0% |

**Best Model: Gradient Boosting** with 95.1% accuracy and 93.0% F1 score.

## ✨ Features

- 📊 **Exploratory Data Analysis** — Statistical summaries, distributions, correlations
- 🤖 **5 ML Algorithms** — Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN
- 📈 **8 Professional Visualizations** — ROC curves, confusion matrices, feature importance
- 🔄 **Cross-Validation** — 5-fold CV for robust performance evaluation
- 📉 **Model Comparison** — Side-by-side metrics comparison across all models
- 🎯 **Feature Importance** — Identifies which health metrics matter most

## 📊 Visualizations

The system generates 8 publication-quality charts:

| # | Visualization | Purpose |
|---|--------------|---------|
| 1 | Target Distribution | Disease vs No Disease breakdown |
| 2 | Age Distribution | Age patterns by diagnosis |
| 3 | Correlation Heatmap | Feature relationships |
| 4 | Feature Comparison | Key metrics: Disease vs Healthy |
| 5 | Model Comparison | All 5 models side-by-side |
| 6 | Confusion Matrices | Prediction accuracy per model |
| 7 | ROC Curves | True vs False positive rates |
| 8 | Feature Importance | Which features predict disease |

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Core language |
| Scikit-learn | ML algorithms, metrics, preprocessing |
| Pandas | Data manipulation & analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |

## 🚀 Getting Started

### Prerequisites
```bash
Python >= 3.8
```

### Installation & Run
```bash
# Clone the repository
git clone https://github.com/KHALEDNOAMAN/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor

# Install dependencies
pip install -r requirements.txt

# Run the prediction system
python heart_disease_predictor.py
```

### Output
The system will:
1. Load and analyze the dataset
2. Create 8 visualizations in the `output/` folder
3. Train 5 ML models
4. Print a performance comparison table
5. Identify the best-performing model

## 📁 Project Structure

```
Heart-Disease-Predictor/
├── heart_disease_predictor.py   # Main ML pipeline
├── requirements.txt             # Python dependencies
├── output/                      # Generated visualizations
│   ├── 01_target_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_feature_comparison.png
│   ├── 05_model_comparison.png
│   ├── 06_confusion_matrices.png
│   ├── 07_roc_curves.png
│   └── 08_feature_importance.png
├── LICENSE
└── README.md
```

## 🔬 How It Works

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Load Data   │───▶│     EDA      │───▶│ Preprocess   │
│  (303 rows)  │    │ (Visualize)  │    │ (Scale/Split)│
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
        ┌───────────────────────┐
        │   Train 5 ML Models   │
        ├───────────────────────┤
        │ • Logistic Regression │
        │ • Random Forest       │
        │ • Gradient Boosting   │
        │ • SVM (RBF)          │
        │ • K-Nearest Neighbors │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Evaluate & Compare   │
        │  • Accuracy/F1/ROC   │
        │  • Cross-Validation  │
        │  • Visualizations    │
        └───────────────────────┘
```

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Khaled Noaman** — Computer Engineering Student at Istanbul Arel University

- [GitHub](https://github.com/KhaledNoaman)
- [LinkedIn](https://www.linkedin.com/in/khalednoaman1/)
