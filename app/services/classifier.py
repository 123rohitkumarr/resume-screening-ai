import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from typing import Optional, List, Tuple
from app.config import settings
from app.utils.helpers import setup_logger

log = setup_logger(__name__)


class RoleClassifier:
    """XGBoost-based Job Role Classifier using TF-IDF skills + OneHot education + Scaled experience."""

    def __init__(self):
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.ohe = None
        self.scaler = None

    def _load_models(self) -> bool:
        if self.model is not None and self.tfidf is not None and self.label_encoder is not None:
            return True
        try:
            if not os.path.exists(settings.classifier_path) or not os.path.exists(settings.tfidf_path):
                log.warning("Classifier model files not found on disk. Run python ml/train_classifier.py to train.")
                return False

            self.model = joblib.load(settings.classifier_path)
            self.tfidf = joblib.load(settings.tfidf_path)
            self.label_encoder = joblib.load(settings.label_encoder_path)

            # Load optional encoders if present
            if os.path.exists(settings.ohe_path):
                self.ohe = joblib.load(settings.ohe_path)
            if os.path.exists(settings.scaler_path):
                self.scaler = joblib.load(settings.scaler_path)

            log.info("Successfully loaded classifier models.")
            return True
        except Exception as e:
            log.warning(f"Failed to load classifier models: {e}")
            self.model = None
            self.tfidf = None
            self.label_encoder = None
            self.ohe = None
            self.scaler = None
            return False

    def _build_features(self, skills: List[str], education: Optional[str] = None, experience: Optional[float] = None):
        """Construct the combined sparse feature matrix matching train_classifier.py."""
        skill_text = " ".join(skills) if skills else ""
        X_text = self.tfidf.transform([skill_text])

        # If model expects 505 features (with OHE & Scaler)
        if self.ohe is not None and self.scaler is not None:
            edu_val = str(education).title() if education else "Unknown"
            exp_val = float(experience) if experience is not None else 0.0

            df_edu = pd.DataFrame([[edu_val]], columns=["Education"])
            df_exp = pd.DataFrame([[exp_val]], columns=["YearsExperience"])

            X_edu = self.ohe.transform(df_edu)
            X_exp = self.scaler.transform(df_exp)

            return hstack((X_text, X_edu, X_exp))

        return X_text

    def predict(self, skills: List[str], education: Optional[str] = None, experience: Optional[float] = None) -> Tuple[str, float]:
        if not self._load_models():
            return ("Unknown", 0.0)

        try:
            X = self._build_features(skills, education, experience)
            pred_idx = self.model.predict(X)[0]
            prob = float(self.model.predict_proba(X)[0].max())
            predicted_role = str(self.label_encoder.inverse_transform([pred_idx])[0])
            return predicted_role, round(prob, 4)
        except Exception as e:
            log.error(f"Prediction error in classifier: {e}", exc_info=True)
            return ("Unknown", 0.0)

    def predict_top_k(
        self, skills: List[str], education: Optional[str] = None, experience: Optional[float] = None, k: int = 3
    ) -> List[Tuple[str, float]]:
        if not self._load_models():
            return [("Unknown", 0.0)]

        try:
            X = self._build_features(skills, education, experience)
            probs = self.model.predict_proba(X)[0]
            top_k_idx = probs.argsort()[-k:][::-1]
            results = []
            for idx in top_k_idx:
                role = str(self.label_encoder.inverse_transform([idx])[0])
                results.append((role, round(float(probs[idx]), 4)))
            return results
        except Exception as e:
            log.error(f"Top-k prediction error in classifier: {e}", exc_info=True)
            return [("Unknown", 0.0)]


classifier = RoleClassifier()
