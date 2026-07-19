import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.helpers import setup_logger
log = setup_logger(__name__)

def evaluate_model():
    log.info("Starting model evaluation process...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, "data", "models")
    data_path = os.path.join(root_dir, "data", "raw", "resume_dataset.json")
    
    # Load Models
    log.info("Loading models...")
    clf = joblib.load(os.path.join(models_dir, "classifier.joblib"))
    tfidf = joblib.load(os.path.join(models_dir, "tfidf.joblib"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    ohe = joblib.load(os.path.join(models_dir, "ohe.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    
    # Load Data
    with open(data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    log.info(f"Loaded {len(df)} test records.")
    
    # Feature transform
    X_text = tfidf.transform(df['Skills'])
    X_edu = ohe.transform(df[['Education']])
    X_exp = scaler.transform(df[['YearsExperience']])
    X = hstack((X_text, X_edu, X_exp))
    y = le.transform(df['JobRole'])
    
    # Predictions
    y_pred = clf.predict(X)
    y_probs = clf.predict_proba(X)
    
    # Metrics
    overall_acc = accuracy_score(y, y_pred)
    
    # Top-3 Accuracy
    top3_preds = np.argsort(y_probs, axis=1)[:, -3:]
    top3_acc = np.mean([y[i] in top3_preds[i] for i in range(len(y))])
    
    print("\n--- Evaluation Results ---")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Top-3 Accuracy:   {top3_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    df_cm = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_cm)
    
    # CV Scores (on the full dataset to evaluate general stability)
    log.info("Running CV on full dataset...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"\nCross-Validation Scores (5-fold): {scores}")
    print(f"Mean CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores)*2:.4f})")

if __name__ == "__main__":
    evaluate_model()
