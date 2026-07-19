import joblib
from app.config import settings
from app.utils.helpers import setup_logger

log = setup_logger(__name__)

class RoleClassifier:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.label_encoder = None

    def _load_models(self):
        if self.model is not None:
            return True
        try:
            self.model = joblib.load(settings.classifier_path)
            self.tfidf = joblib.load(settings.tfidf_path)
            self.label_encoder = joblib.load(settings.label_encoder_path)
            log.info("Successfully loaded classifier models.")
            return True
        except FileNotFoundError as e:
            log.warning(f"Failed to load classifier models: {e}")
            self.model = None
            self.tfidf = None
            self.label_encoder = None
            return False

    def predict(self, skills: list[str], education: str = None, experience: float = None) -> tuple[str, float]:
        if not self._load_models():
            return ('Unknown', 0.0)

        feature_text = " ".join(skills)
        if education:
            feature_text += f" {education}"
        if experience is not None:
            feature_text += f" {experience}"

        X = self.tfidf.transform([feature_text])
        pred_idx = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0].max()
        
        predicted_role = self.label_encoder.inverse_transform([pred_idx])[0]
        return predicted_role, float(prob)

    def predict_top_k(self, skills: list[str], education: str = None, experience: float = None, k: int = 3) -> list[tuple[str, float]]:
        if not self._load_models():
            return [('Unknown', 0.0)]

        feature_text = " ".join(skills)
        if education:
            feature_text += f" {education}"
        if experience is not None:
            feature_text += f" {experience}"

        X = self.tfidf.transform([feature_text])
        probs = self.model.predict_proba(X)[0]
        
        top_k_idx = probs.argsort()[-k:][::-1]
        results = []
        for idx in top_k_idx:
            role = self.label_encoder.inverse_transform([idx])[0]
            results.append((role, float(probs[idx])))
            
        return results

classifier = RoleClassifier()
