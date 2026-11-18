import re
from rapidfuzz import fuzz
from app.utils.helpers import setup_logger, normalize_skill
from app.models.schemas import SkillAnalysis

log = setup_logger(__name__)

SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "ruby",
        "php", "swift", "kotlin", "rust", "scala", "r", "matlab", "perl",
        "objective-c", "dart", "shell", "bash", "powershell"
    ],
    "web_frameworks": [
        "react", "angular", "vue", "django", "flask", "fastapi", "spring boot",
        "express", "node.js", "ruby on rails", "asp.net", "next.js", "nuxt.js",
        "svelte", "laravel", "symfony", "struts", "meteor", "ember"
    ],
    "databases": [
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
        "sqlite", "oracle", "sql server", "dynamodb", "couchbase", "neo4j",
        "mariadb", "firebase", "snowflake", "bigquery", "redshift", "cosmos db"
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "github actions", "gitlab ci", "circleci", "vagrant",
        "puppet", "chef", "helm", "prometheus", "grafana", "nginx", "apache"
    ],
    "data_science_ml": [
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "scikit-learn", "keras",
        "pandas", "numpy", "matplotlib", "seaborn", "scipy", "xgboost",
        "lightgbm", "opencv", "nltk", "spacy", "huggingface"
    ],
    "mobile": [
        "ios", "android", "react native", "flutter", "xamarin", "ionic",
        "cordova", "swiftui", "jetpack compose", "capacitor", "objective-c",
        "cocoa touch", "android sdk", "xcode", "android studio"
    ],
    "testing": [
        "jest", "mocha", "pytest", "unittest", "selenium", "cypress",
        "puppeteer", "playwright", "jasmine", "karma", "enzyme",
        "testing library", "rspec", "cucumber", "appium", "postman"
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "time management", "adaptability", "collaboration",
        "mentoring", "agile", "scrum", "kanban", "project management",
        "presentation", "negotiation", "conflict resolution"
    ]
}

SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "k8s": "kubernetes",
    "tf": "terraform",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "db": "database",
    "os": "operating system",
    "ui": "user interface",
    "ux": "user experience",
    "qa": "quality assurance",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "rn": "react native",
    "k8": "kubernetes",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "vuejs": "vue",
    "reactjs": "react"
}

def resolve_alias(skill: str) -> str:
    normalized = skill.lower().strip()
    return SKILL_ALIASES.get(normalized, normalized)

def extract_skills(text: str) -> list[str]:
    if not text:
        return []

    # Clean text preserving dots and plus/hashes for language names (e.g. C++, C#, .NET, Node.js)
    cleaned_text = re.sub(r'[/\\()\[\]{};:,\n\r\t|•–—]', ' ', text.lower())
    padded_text = f" {cleaned_text} "

    found_skills = set()

    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            pattern = r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])'
            if re.search(pattern, padded_text):
                found_skills.add(skill)

    for alias, full_name in SKILL_ALIASES.items():
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(alias) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, padded_text):
            found_skills.add(full_name)

    return sorted([normalize_skill(skill) for skill in found_skills])

def match_skills(resume_skills: list[str], required_skills: list[str], preferred_skills: list[str] = None) -> SkillAnalysis:
    if preferred_skills is None:
        preferred_skills = []
        
    resume_skills_norm = [normalize_skill(resolve_alias(s)) for s in resume_skills]
    required_skills_norm = [normalize_skill(resolve_alias(s)) for s in required_skills]
    preferred_skills_norm = [normalize_skill(resolve_alias(s)) for s in preferred_skills]
    
    matched = []
    missing = []
    bonus = []
    
    def is_match(skill, skill_list, threshold=80):
        for candidate in skill_list:
            if fuzz.ratio(skill, candidate) >= threshold:
                return True
        return False

    for req in required_skills_norm:
        if is_match(req, resume_skills_norm):
            matched.append(req)
        else:
            missing.append(req)
            
    pref_matched_count = 0
    for pref in preferred_skills_norm:
        if is_match(pref, resume_skills_norm):
            bonus.append(pref)
            pref_matched_count += 1
            
    for res in resume_skills_norm:
        if not is_match(res, required_skills_norm) and not is_match(res, preferred_skills_norm):
            bonus.append(res)
            
    match_percentage = (len(matched) / len(required_skills_norm) * 100) if required_skills_norm else 100.0
    
    return SkillAnalysis(
        matched=matched,
        missing=missing,
        bonus=list(set(bonus)),
        match_percentage=round(match_percentage, 2),
        preferred_skills_match_count=pref_matched_count
    )

def categorize_skills(skills: list[str]) -> dict[str, list[str]]:
    categorized = {cat: [] for cat in SKILL_TAXONOMY.keys()}
    categorized["other"] = []
    
    for skill in skills:
        norm_skill = normalize_skill(resolve_alias(skill))
        found = False
        for category, cat_skills in SKILL_TAXONOMY.items():
            if norm_skill in cat_skills:
                categorized[category].append(norm_skill)
                found = True
                break
        if not found:
            categorized["other"].append(norm_skill)
            
    return {k: v for k, v in categorized.items() if v}

def get_skill_importance(skill: str) -> str:
    norm_skill = normalize_skill(resolve_alias(skill))
    
    high_importance_cats = ["programming_languages", "data_science_ml", "cloud_devops"]
    medium_importance_cats = ["web_frameworks", "databases", "mobile", "testing"]
    
    for cat in high_importance_cats:
        if norm_skill in SKILL_TAXONOMY.get(cat, []):
            return "high"
            
    for cat in medium_importance_cats:
        if norm_skill in SKILL_TAXONOMY.get(cat, []):
            return "medium"
            
    return "low"
