"""
Heart Disease Prediction System
================================
A machine learning project that predicts the likelihood of heart disease
using patient health metrics. Uses multiple ML algorithms and compares
their performance.

Author: Khaled Noaman
Technologies: Python, Scikit-learn, Pandas, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================
# Configuration
# ============================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================
# STEP 1: Load & Explore the Dataset
# ============================================
def load_data():
    """
    Load the Heart Disease dataset.
    Uses the UCI Cleveland Heart Disease dataset.
    
    Features:
    - age: Age of patient
    - sex: Gender (1=male, 0=female)
    - cp: Chest pain type (0-3)
    - trestbps: Resting blood pressure (mm Hg)
    - chol: Serum cholesterol (mg/dl)
    - fbs: Fasting blood sugar > 120 mg/dl (1=true, 0=false)
    - restecg: Resting ECG results (0-2)
    - thalach: Maximum heart rate achieved
    - exang: Exercise induced angina (1=yes, 0=no)
    - oldpeak: ST depression induced by exercise
    - slope: Slope of peak exercise ST segment
    - ca: Number of major vessels colored by fluoroscopy (0-3)
    - thal: Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)
    - target: Heart disease (1=disease, 0=no disease)
    """
    print("=" * 60)
    print("  HEART DISEASE PREDICTION SYSTEM")
    print("  Using Machine Learning")
    print("=" * 60)
    print()
    
    # Use the built-in UCI Heart Disease dataset URL
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"[OK] Dataset loaded: {df.shape[0]} patients, {df.shape[1]} features")
    except Exception:
        # Fallback: generate synthetic data that mimics the heart disease dataset
        print("[INFO] Generating dataset locally...")
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            'age': np.random.randint(29, 77, n),
            'sex': np.random.choice([0, 1], n, p=[0.32, 0.68]),
            'cp': np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08]),
            'trestbps': np.random.normal(131, 17, n).astype(int),
            'chol': np.random.normal(246, 52, n).astype(int),
            'fbs': np.random.choice([0, 1], n, p=[0.85, 0.15]),
            'restecg': np.random.choice([0, 1, 2], n, p=[0.48, 0.49, 0.03]),
            'thalach': np.random.normal(149, 23, n).astype(int),
            'exang': np.random.choice([0, 1], n, p=[0.67, 0.33]),
            'oldpeak': np.abs(np.random.normal(1.04, 1.16, n)).round(1),
            'slope': np.random.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47]),
            'ca': np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07]),
            'thal': np.random.choice([0, 1, 2, 3], n, p=[0.01, 0.06, 0.54, 0.39]),
        })
        # Generate target based on features (simplified model)
        risk = (df['age'] > 55).astype(int) + (df['cp'] >= 2).astype(int) + \
               (df['thalach'] < 140).astype(int) + df['exang'] + \
               (df['oldpeak'] > 1.5).astype(int) + (df['ca'] > 0).astype(int)
        df['target'] = (risk >= 3).astype(int)
        print(f"[OK] Dataset generated: {df.shape[0]} patients, {df.shape[1]} features")
    
    return df


def explore_data(df):
    """Perform Exploratory Data Analysis (EDA)"""
    print("\n" + "=" * 60)
    print("  STEP 1: Exploratory Data Analysis")
    print("=" * 60)
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"\n📋 Feature Types:")
    print(df.dtypes.to_string())
    
    print(f"\n📈 Statistical Summary:")
    print(df.describe().round(2).to_string())
    
    print(f"\n🎯 Target Distribution:")
    target_counts = df['target'].value_counts()
    print(f"   No Disease (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   Disease    (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    print(f"\n🔍 Missing Values: {df.isnull().sum().sum()}")
    
    return df


# ============================================
# STEP 2: Data Visualization
# ============================================
def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 60)
    print("  STEP 2: Creating Visualizations")
    print("=" * 60)
    
    # --- Plot 1: Target Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#2ecc71', '#e74c3c']
    target_counts = df['target'].value_counts()
    axes[0].pie(target_counts, labels=['No Disease', 'Disease'], 
                colors=colors, autopct='%1.1f%%', startangle=90,
                explode=(0.05, 0.05), shadow=True,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')
    
    sns.countplot(data=df, x='target', ax=axes[1], palette=colors, edgecolor='black')
    axes[1].set_xticklabels(['No Disease', 'Disease'])
    axes[1].set_title('Patient Count by Diagnosis', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Count')
    
    # Add count labels on bars
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 01_target_distribution.png")
    
    # --- Plot 2: Age Distribution by Disease ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='target', kde=True, bins=20,
                palette=colors, alpha=0.7, ax=ax)
    ax.set_title('Age Distribution by Heart Disease Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(['No Disease', 'Disease'])
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_age_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 02_age_distribution.png")
    
    # --- Plot 3: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 03_correlation_heatmap.png")
    
    # --- Plot 4: Key Features Comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    features_to_plot = [
        ('thalach', 'Max Heart Rate'),
        ('chol', 'Cholesterol Level'),
        ('trestbps', 'Resting Blood Pressure'),
        ('oldpeak', 'ST Depression (Oldpeak)')
    ]
    
    for idx, (feat, title) in enumerate(features_to_plot):
        row, col = idx // 2, idx % 2
        sns.boxplot(data=df, x='target', y=feat, ax=axes[row][col], 
                   palette=colors, width=0.5)
        axes[row][col].set_xticklabels(['No Disease', 'Disease'])
        axes[row][col].set_title(f'{title} by Diagnosis', fontsize=12, fontweight='bold')
        axes[row][col].set_xlabel('')
    
    plt.suptitle('Key Health Metrics: Disease vs No Disease', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_feature_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 04_feature_comparison.png")


# ============================================
# STEP 3: Data Preprocessing
# ============================================
def preprocess_data(df):
    """Prepare data for machine learning"""
    print("\n" + "=" * 60)
    print("  STEP 3: Data Preprocessing")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set:  {X_test.shape[0]} samples")
    
    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("   [OK] Features scaled with StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


# ============================================
# STEP 4: Train Multiple ML Models
# ============================================
def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple ML algorithms"""
    print("\n" + "=" * 60)
    print("  STEP 4: Training Machine Learning Models")
    print("=" * 60)
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   🔄 Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"      Accuracy:  {acc:.4f}")
        print(f"      Precision: {prec:.4f}")
        print(f"      Recall:    {rec:.4f}")
        print(f"      F1 Score:  {f1:.4f}")
        print(f"      CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results


# ============================================
# STEP 5: Visualize Model Performance
# ============================================
def visualize_results(results, y_test):
    """Create performance comparison visualizations"""
    print("\n" + "=" * 60)
    print("  STEP 5: Visualizing Results")
    print("=" * 60)
    
    # --- Plot 5: Model Comparison Bar Chart ---
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), 
                     color=colors[i], edgecolor='black', linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.3, label='90% threshold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 05_model_comparison.png")
    
    # --- Plot 6: Confusion Matrices ---
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    
    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        axes[idx].set_title(f'{name}\nAcc: {res["accuracy"]:.2f}', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Actual' if idx == 0 else '')
        axes[idx].set_xlabel('Predicted')
    
    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 06_confusion_matrices.png")
    
    # --- Plot 7: ROC Curves ---
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_roc = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    for idx, (name, res) in enumerate(results.items()):
        if res['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors_roc[idx], lw=2,
                   label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 07_roc_curves.png")
    
    # --- Plot 8: Feature Importance (from Random Forest) ---
    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    indices = np.argsort(importances)[::-1]
    
    colors_feat = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(feature_names)))
    bars = ax.bar(range(len(feature_names)), importances[indices], 
                  color=colors_feat, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 08_feature_importance.png")


# ============================================
# STEP 6: Summary & Best Model
# ============================================
def print_summary(results):
    """Print final summary and best model"""
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    # Create results table
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV Score':>10}")
    print("-" * 77)
    
    best_model = None
    best_f1 = 0
    
    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
              f"{res['recall']:>10.4f} {res['f1']:>10.4f} {res['cv_mean']:>10.4f}")
        if res['f1'] > best_f1:
            best_f1 = res['f1']
            best_model = name
    
    print("-" * 77)
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   F1 Score: {results[best_model]['f1']:.4f}")
    print(f"   Accuracy: {results[best_model]['accuracy']:.4f}")
    print(f"   CV Score: {results[best_model]['cv_mean']:.4f} (+/- {results[best_model]['cv_std']:.4f})")
    
    print(f"\n📁 All visualizations saved to: ./{OUTPUT_DIR}/")
    print(f"   • 01_target_distribution.png")
    print(f"   • 02_age_distribution.png")
    print(f"   • 03_correlation_heatmap.png")
    print(f"   • 04_feature_comparison.png")
    print(f"   • 05_model_comparison.png")
    print(f"   • 06_confusion_matrices.png")
    print(f"   • 07_roc_curves.png")
    print(f"   • 08_feature_importance.png")
    
    return best_model


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Step 1: Load data
    df = load_data()
    df = explore_data(df)
    
    # Step 2: Visualize
    create_visualizations(df)
    
    # Step 3: Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Step 4: Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Visualize results
    visualize_results(results, y_test)
    
    # Step 6: Summary
    best = print_summary(results)
    
    print("\n" + "=" * 60)
    print("  DONE! Heart Disease Prediction System complete.")
    print("=" * 60)
