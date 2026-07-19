import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from app.utils.helpers import setup_logger

log = setup_logger(__name__)

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.strip()]
    return ' '.join(processed_tokens)

def tokenize_skills(skills_text: str) -> list[str]:
    if not skills_text:
        return []
    skills = [s.strip().lower() for s in skills_text.split(',')]
    return [s for s in skills if s]

def extract_experience_years(text: str) -> float | None:
    if not text:
        return None
        
    text = text.lower()
    pattern = r'(\d+(?:\.\d+)?)\+?\s*(?:to|-)\s*(\d+(?:\.\d+)?)?\s*years?|(\d+(?:\.\d+)?)\+?\s*years?'
    matches = re.finditer(pattern, text)
    
    max_years = 0.0
    for match in matches:
        groups = match.groups()
        try:
            if groups[2]:
                val = float(groups[2])
            elif groups[1]:
                val = float(groups[1])
            elif groups[0]:
                val = float(groups[0])
            else:
                continue
            max_years = max(max_years, val)
        except ValueError:
            pass
            
    return max_years if max_years > 0 else None

def extract_education(text: str) -> str | None:
    if not text:
        return None
        
    text = text.lower()
    
    if re.search(r'\b(phd|doctorate|doctor of philosophy)\b', text):
        return 'PhD'
    elif re.search(r'\b(master|masters|ms|msc|ma|mba|m\.s\.|m\.a\.)\b', text):
        return 'Masters'
    elif re.search(r'\b(bachelor|bachelors|bs|bsc|ba|b\.s\.|b\.a\.)\b', text):
        return 'Bachelors'
    elif re.search(r'\b(associate|associates|a\.s\.|a\.a\.)\b', text):
        return 'Associate'
    elif re.search(r'\b(high school|ged)\b', text):
        return 'High School'
    
    return None

def build_feature_text(skills: list[str], education: str, experience: float) -> str:
    components = []
    if skills:
        components.extend(skills)
    if education:
        components.append(f"edu_{education.lower()}")
    if experience is not None:
        if experience > 10:
            components.append("exp_senior")
        elif experience > 3:
            components.append("exp_mid")
        else:
            components.append("exp_junior")
            
    return " ".join(components)
