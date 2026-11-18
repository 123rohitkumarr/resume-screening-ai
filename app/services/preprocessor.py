import re
from typing import Optional, List
from app.utils.helpers import setup_logger

log = setup_logger(__name__)

# Fallback English stopwords in case NLTK data is unavailable offline
FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours", "yourself", "yourselves"
}

_stop_words = None
_lemmatizer = None

def _get_nlp_resources():
    """Lazily load NLTK resources with graceful fallback for offline/sandboxed environments."""
    global _stop_words, _lemmatizer
    if _stop_words is not None and _lemmatizer is not None:
        return _stop_words, _lemmatizer

    try:
        import nltk
        try:
            nltk.download("punkt_tab", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
        except Exception:
            pass

        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        _stop_words = set(stopwords.words("english"))
        _lemmatizer = WordNetLemmatizer()
    except Exception as e:
        log.warning(f"Using fallback NLP resources: {e}")
        _stop_words = FALLBACK_STOPWORDS
        _lemmatizer = None

    return _stop_words, _lemmatizer


def preprocess_text(text: str) -> str:
    """Preprocess text for NLP feature extraction (lowercasing, cleaning, tokenizing, lemmatizing)."""
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    stop_words, lemmatizer = _get_nlp_resources()

    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except Exception:
        tokens = re.findall(r"\b[a-z0-9]+\b", text)

    processed_tokens = []
    for token in tokens:
        clean_tok = token.strip()
        if clean_tok and clean_tok not in stop_words:
            if lemmatizer is not None:
                try:
                    clean_tok = lemmatizer.lemmatize(clean_tok)
                except Exception:
                    pass
            processed_tokens.append(clean_tok)

    return " ".join(processed_tokens)


def tokenize_skills(skills_text: str) -> List[str]:
    """Tokenize a comma-delimited string of skills."""
    if not skills_text:
        return []
    skills = [s.strip().lower() for s in skills_text.split(",")]
    return [s for s in skills if s]


def extract_email(text: str) -> Optional[str]:
    """Extract candidate email address from text."""
    if not text:
        return None
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    """Extract candidate phone number from text."""
    if not text:
        return None
    pattern = r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?"
    match = re.search(pattern, text)
    if match and len(match.group(0).strip()) >= 10:
        return match.group(0).strip()
    return None


def extract_candidate_name(text: str) -> Optional[str]:
    """Attempt to extract candidate name from the top lines of a resume."""
    if not text:
        return None
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    ignore_words = {"resume", "curriculum", "vitae", "cv", "page", "profile", "contact", "summary"}

    for line in lines[:5]:
        line_clean = re.sub(r"[^a-zA-Z\s]", "", line).strip()
        words = line_clean.split()
        if 2 <= len(words) <= 4:
            if not any(w.lower() in ignore_words for w in words):
                return line_clean.title()
    return None


def extract_experience_years(text: str) -> Optional[float]:
    """
    Extract years of experience from resume text.
    Differentiates between:
    - Explicit zero experience (fresher, entry-level, intern, 0 years) -> returns 0.0
    - Quantified experience (X years, X+ years, X-Y years) -> returns float(X)
    - Unmentioned/Unknown experience -> returns None
    """
    if not text:
        return None

    text_lower = text.lower()

    # Check for explicit fresher / zero experience indicators
    zero_exp_patterns = [
        r"\b(?:0|zero)\s*(?:\.0)?\s*years?(?:\s*of)?\s*experience\b",
        r"\b(?:fresher|fresh\s*graduate|entry\s*level|intern|no\s*prior\s*experience)\b"
    ]
    for zpat in zero_exp_patterns:
        if re.search(zpat, text_lower):
            # Verify there isn't a higher experience number elsewhere
            has_positive = re.search(r"(\d+(?:\.\d+)?)\+?\s*years?(?:\s*of)?\s*experience", text_lower)
            if not has_positive or float(has_positive.group(1)) == 0.0:
                return 0.0

    # Pattern for "X to Y years", "X+ years", "X years of experience"
    pattern = r"(\d+(?:\.\d+)?)\+?\s*(?:to|-)\s*(\d+(?:\.\d+)?)?\s*years?|(\d+(?:\.\d+)?)\+?\s*years?"
    matches = re.finditer(pattern, text_lower)

    found_years = []
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
            # Filter unrealistic experience numbers (> 50 years is likely a year like 2020 or typo)
            if 0.0 <= val <= 50.0:
                found_years.append(val)
        except ValueError:
            pass

    if found_years:
        return max(found_years)

    return None


def extract_education(text: str) -> Optional[str]:
    """
    Extract highest education credential from resume text.
    Returns 'PhD', 'Masters', 'Bachelors', 'Associate', 'High School', or None if not found.
    """
    if not text:
        return None

    text_lower = text.lower()

    if re.search(r"\b(phd|doctorate|doctor of philosophy|ph\.d\.)\b", text_lower):
        return "PhD"
    elif re.search(r"\b(master|masters|msc|m\.sc\.|mba|m\.b\.a\.|m\.s\.|m\.a\.|mtech|m\.tech)\b", text_lower):
        return "Masters"
    elif re.search(r"\b(bachelor|bachelors|bsc|b\.sc\.|bs|b\.s\.|ba|b\.a\.|btech|b\.tech|be|b\.e\.)\b", text_lower):
        return "Bachelors"
    elif re.search(r"\b(associate|associates|a\.s\.|a\.a\.)\b", text_lower):
        return "Associate"
    elif re.search(r"\b(high school|ged|secondary education|diploma)\b", text_lower):
        return "High School"

    return None


def build_feature_text(skills: List[str], education: Optional[str], experience: Optional[float]) -> str:
    """Build space-separated feature tokens for text vectorization."""
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
