import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.helpers import setup_logger
log = setup_logger(__name__)

def train():
    log.info("Starting classifier training process...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, "data", "raw", "resume_dataset.json")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    log.info(f"Loaded {len(df)} records.")
    
    # Feature Engineering
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df['Skills'])
    
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_edu = ohe.fit_transform(df[['Education']])
    
    scaler = StandardScaler()
    X_exp = scaler.fit_transform(df[['YearsExperience']])
    
    X = hstack((X_text, X_edu, X_exp))
    
    le = LabelEncoder()
    y = le.fit_transform(df['JobRole'])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    log.info("Training XGBoost classifier...")
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        max_depth=6,
        n_estimators=150,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    log.info(f"CV Mean Accuracy: {np.mean(scores):.4f}")
    
    # Final Train
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)
    log.info("Model evaluation completed.")
    
    # Save models
    models_dir = os.path.join(root_dir, "data", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(clf, os.path.join(models_dir, "classifier.joblib"))
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf.joblib"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))
    joblib.dump(ohe, os.path.join(models_dir, "ohe.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    log.info(f"Saved models to {models_dir}")

if __name__ == "__main__":
    train()
