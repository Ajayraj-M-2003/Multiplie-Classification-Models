"""
BITS Pilani - Work Integrated Learning Programmes Division
M.Tech (AIML) - Machine Learning Assignment 2
Model Training & Artifacts Generation Pipeline
Dataset: UCI Default of Credit Card Clients Dataset
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)


def load_and_preprocess_data(file_path: str):
    """
    Loads the Excel file from UCI repository, strips ID column,
    renames the target variable cleanly, and splits into features and target.
    """
    print(f"[1/5] Loading raw data from: {file_path}")
    df = pd.read_excel(file_path, header=1, engine='calamine')
    
    # Drop ID column if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    # Standardize target column name
    # Original UCI column name is 'default payment next month'
    target_candidates = ['default payment next month', 'default', 'default_payment_next_month']
    target_col = None
    for cand in target_candidates:
        if cand in df.columns:
            target_col = cand
            break
            
    if target_col is None:
        raise ValueError(f"Could not identify target column among candidates {target_candidates}")
        
    df = df.rename(columns={target_col: 'default'})
    
    # Feature columns and target column
    feature_cols = [c for c in df.columns if c != 'default']
    X = df[feature_cols]
    y = df['default'].astype(int)
    
    print(f"      Total records: {len(df):,}")
    print(f"      Features ({len(feature_cols)}): {feature_cols}")
    print(f"      Target distribution: 0={int((y==0).sum())} ({((y==0).mean()*100):.1f}%), 1={int((y==1).sum())} ({((y==1).mean()*100):.1f}%)")
    
    return X, y, feature_cols


def create_train_test_splits(X, y, test_size=0.20, random_state=42):
    """
    Performs a stratified split to preserve minority class default ratio.
    """
    print(f"[2/5] Creating stratified {int((1-test_size)*100)}/{int(test_size*100)} train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"      Train samples: {len(X_train):,}")
    print(f"      Test samples:  {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def save_test_data(X_test, y_test, output_path: str):
    """
    Saves the holdout test set as test_data.csv in the project root.
    Includes all features + target column for out-of-the-box app evaluation.
    """
    test_df = X_test.copy()
    test_df['default'] = y_test.values
    test_df.to_csv(output_path, index=False)
    print(f"[3/5] Exported test dataset to: {output_path} ({len(test_df):,} rows, {len(test_df.columns)} columns)")


def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_dir: str):
    """
    Trains all 5 required ML models:
      1. Logistic Regression
      2. Decision Tree Classifier
      3. K-Nearest Neighbor Classifier
      4. Naive Bayes Classifier (Gaussian)
      5. Support Vector Machine (RBF kernel)
      
    Computes:
      - Accuracy
      - AUC Score
      - Precision
      - Recall
      - F1 Score
      - Matthews Correlation Coefficient (MCC)
      
    Saves trained pipeline models, preprocessors, and metric reports.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Fitted Scaler for continuous/ordinal features
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    # 5 Models Dictionary
    model_definitions = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
            'logistic_regression.joblib'
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=42),
            'decision_tree.joblib'
        ),
        'KNN': (
            KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
            'knn.joblib'
        ),
        'Naive Bayes': (
            GaussianNB(var_smoothing=1e-8),
            'naive_bayes.joblib'
        ),
        'Support Vector Machine': (
            SVC(kernel='rbf', C=1.0, random_state=42),
            'svm.joblib'
        )
    }
    
    metrics_list = []
    confusion_matrices = {}
    
    print(f"[4/5] Training 5 classification models...")
    for model_name, (classifier, filename) in model_definitions.items():
        print(f"      -> Training: {model_name}...")
        
        # Build end-to-end Pipeline (StandardScaler -> Classifier)
        # Note: Decision Tree is scale-invariant, but pipeline maintains uniform interface
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Save pipeline model
        save_path = os.path.join(model_dir, filename)
        joblib.dump(pipeline, save_path)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Ranking scores / Probabilities for ROC-AUC
        clf = pipeline.named_steps['classifier']
        if hasattr(clf, 'predict_proba'):
            y_scores = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(clf, 'decision_function'):
            y_scores = pipeline.decision_function(X_test)
        else:
            y_scores = y_pred
            
        # Metric Computations
        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_scores))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred).tolist()
        confusion_matrices[model_name] = cm
        
        metrics_list.append({
            'ML Model Name': model_name,
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4)
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    print("\n[5/5] Final Model Evaluation Metric Comparison Table:")
    print("=" * 85)
    print(metrics_df.to_string(index=False))
    print("=" * 85)
    
    # Save metrics table as CSV & JSON for application consumption
    metrics_df.to_csv(os.path.join(model_dir, 'metrics_comparison.csv'), index=False)
    with open(os.path.join(model_dir, 'metrics_summary.json'), 'w') as f:
        json.dump({
            'metrics': metrics_list,
            'confusion_matrices': confusion_matrices
        }, f, indent=2)
        
    print(f"All models and artifacts successfully saved to: {model_dir}/")
    return metrics_df


if __name__ == '__main__':
    raw_path = os.path.join('data_raw', 'default of credit card clients.xls')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Please download the UCI dataset first.")
        
    X, y, feature_cols = load_and_preprocess_data(raw_path)
    X_train, X_test, y_train, y_test = create_train_test_splits(X, y, test_size=0.20, random_state=42)
    
    # Save holdout test data in project root
    save_test_data(X_test, y_test, 'test_data.csv')
    
    # Train, evaluate, and persist models
    metrics_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, 'model')
