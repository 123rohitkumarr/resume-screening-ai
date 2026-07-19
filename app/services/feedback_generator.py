"""
AI-powered feedback generator.

Produces structured improvement suggestions by combining rule-based
gap analysis with optional LLM-generated career advice.
"""

from typing import Optional
from app.models.schemas import (
    ScreeningResult, SkillAnalysis, ScoreBreakdown,
    FeedbackResponse,
)
from app.config import settings, LLMProvider
from app.utils.helpers import setup_logger

log = setup_logger(__name__)


# ── Learning resource templates ───────────────────────────────

RESOURCE_MAP = {
    "python": {"platform": "Coursera", "course": "Python for Everybody", "url": "https://coursera.org/specializations/python"},
    "java": {"platform": "Udemy", "course": "Java Masterclass", "url": "https://udemy.com"},
    "javascript": {"platform": "freeCodeCamp", "course": "JavaScript Algorithms", "url": "https://freecodecamp.org"},
    "react": {"platform": "Udemy", "course": "React - The Complete Guide", "url": "https://udemy.com"},
    "machine learning": {"platform": "Coursera", "course": "Machine Learning Specialization", "url": "https://coursera.org/specializations/machine-learning-introduction"},
    "deep learning": {"platform": "Coursera", "course": "Deep Learning Specialization", "url": "https://coursera.org/specializations/deep-learning"},
    "sql": {"platform": "Khan Academy", "course": "Intro to SQL", "url": "https://khanacademy.org"},
    "docker": {"platform": "Docker Docs", "course": "Get Started Guide", "url": "https://docs.docker.com/get-started/"},
    "kubernetes": {"platform": "Udemy", "course": "Kubernetes for Beginners", "url": "https://udemy.com"},
    "aws": {"platform": "AWS", "course": "AWS Cloud Practitioner", "url": "https://aws.amazon.com/training/"},
    "tensorflow": {"platform": "TensorFlow", "course": "Official Tutorials", "url": "https://tensorflow.org/tutorials"},
    "pytorch": {"platform": "PyTorch", "course": "Official Tutorials", "url": "https://pytorch.org/tutorials/"},
    "data visualization": {"platform": "Coursera", "course": "Data Visualization with Python", "url": "https://coursera.org"},
    "agile": {"platform": "Scrum.org", "course": "PSM I Certification", "url": "https://scrum.org"},
    "git": {"platform": "Atlassian", "course": "Git Tutorials", "url": "https://atlassian.com/git/tutorials"},
    "node.js": {"platform": "Udemy", "course": "Node.js Developer Course", "url": "https://udemy.com"},
    "typescript": {"platform": "TypeScript Docs", "course": "Official Handbook", "url": "https://typescriptlang.org/docs/"},
    "ci/cd": {"platform": "GitLab", "course": "CI/CD Fundamentals", "url": "https://docs.gitlab.com/ee/ci/"},
}

DEFAULT_RESOURCE = {"platform": "Udemy / Coursera", "course": "Search for relevant courses", "url": "https://udemy.com"}


class FeedbackGenerator:
    """Generates structured career improvement feedback."""

    def generate(
        self,
        screening: ScreeningResult,
        focus_areas: Optional[list[str]] = None,
    ) -> FeedbackResponse:
        scores = screening.scores
        skills = screening.skill_analysis

        overall_feedback = self._build_overall_feedback(scores, skills)
        skill_recs = self._build_skill_recommendations(skills, focus_areas)
        career_tips = self._build_career_tips(scores, screening.predicted_role)
        resources = self._build_learning_resources(skills.missing)
        priorities = self._build_priority_actions(scores, skills)

        return FeedbackResponse(
            overall_feedback=overall_feedback,
            skill_recommendations=skill_recs,
            career_tips=career_tips,
            learning_resources=resources,
            priority_actions=priorities,
        )

    async def generate_with_llm(
        self,
        screening: ScreeningResult,
        focus_areas: Optional[list[str]] = None,
    ) -> FeedbackResponse:
        """Enhanced version that uses LLM for richer suggestions."""
        base = self.generate(screening, focus_areas)

        try:
            from app.services.rag_engine import rag_engine

            prompt = self._build_llm_prompt(screening, focus_areas)
            response, _, _ = rag_engine.generate(prompt)

            if response and "offline mode" not in response:
                base.overall_feedback = response
        except Exception as e:
            log.debug(f"LLM enhancement skipped: {e}")

        return base

    # ── Builders ──────────────────────────────────────────────

    def _build_overall_feedback(
        self, scores: ScoreBreakdown, skills: SkillAnalysis
    ) -> str:
        parts = []
        overall = scores.overall

        if overall >= 80:
            parts.append(
                "Your profile is highly competitive for this position. "
                "You demonstrate strong alignment across most evaluation criteria."
            )
        elif overall >= 60:
            parts.append(
                "You have a solid foundation for this role. With targeted "
                "improvements in a few areas, you could become a top candidate."
            )
        elif overall >= 40:
            parts.append(
                "Your profile shows potential but has notable gaps compared "
                "to the requirements. Focused upskilling would significantly "
                "strengthen your candidacy."
            )
        else:
            parts.append(
                "This position may require significant additional preparation. "
                "Consider building foundational skills before applying, or "
                "look for junior-level positions in this field."
            )

        # Highlight strengths
        strongest = max(
            [("semantic alignment", scores.semantic),
             ("skill match", scores.skill),
             ("experience", scores.experience),
             ("education", scores.education)],
            key=lambda x: x[1],
        )
        if strongest[1] >= 60:
            parts.append(
                f"Your strongest area is {strongest[0]} ({strongest[1]:.0f}/100)."
            )

        return " ".join(parts)

    def _build_skill_recommendations(
        self,
        skills: SkillAnalysis,
        focus_areas: Optional[list[str]] = None,
    ) -> list[dict]:
        recs = []
        missing = skills.missing
        if focus_areas:
            # Prioritize skills the user wants to focus on
            missing = sorted(
                missing,
                key=lambda s: 0 if s.lower() in [f.lower() for f in focus_areas] else 1,
            )

        for skill in missing[:8]:
            importance = self._assess_importance(skill, skills)
            rec = {
                "skill": skill,
                "importance": importance,
                "suggestion": self._skill_suggestion(skill, importance),
                "estimated_time": self._estimate_learning_time(skill),
            }
            recs.append(rec)

        return recs

    def _build_career_tips(
        self, scores: ScoreBreakdown, predicted_role: Optional[str]
    ) -> list[str]:
        tips = []

        if scores.semantic < 60:
            tips.append(
                "Customize your resume for each application by mirroring "
                "the language and keywords from the job description."
            )

        if scores.skill < 60:
            tips.append(
                "Prioritize learning the most commonly required skills in "
                "your target role. Focus on depth over breadth."
            )

        if scores.experience < 50:
            tips.append(
                "Contribute to open-source projects or take on freelance work "
                "to build practical experience in relevant technologies."
            )

        if predicted_role:
            tips.append(
                f"Your profile aligns with a {predicted_role} role. Consider "
                f"tailoring your job search to positions in this area."
            )

        tips.append(
            "Include quantifiable achievements in your resume (e.g., "
            "'Reduced API latency by 40%') to stand out."
        )

        tips.append(
            "Maintain an active GitHub/portfolio showcasing your best "
            "projects, especially those relevant to target positions."
        )

        return tips

    def _build_learning_resources(self, missing_skills: list[str]) -> list[dict]:
        resources = []
        for skill in missing_skills[:6]:
            key = skill.lower()
            resource = RESOURCE_MAP.get(key, DEFAULT_RESOURCE).copy()
            resource["skill"] = skill
            resources.append(resource)
        return resources

    def _build_priority_actions(
        self, scores: ScoreBreakdown, skills: SkillAnalysis
    ) -> list[str]:
        actions = []

        # Most impactful action first
        weakest = min(
            [("semantic alignment", scores.semantic),
             ("skill match", scores.skill),
             ("experience", scores.experience),
             ("education", scores.education)],
            key=lambda x: x[1],
        )

        action_map = {
            "semantic alignment": "Rewrite your resume to better match the job description terminology.",
            "skill match": f"Learn these critical missing skills: {', '.join(skills.missing[:3])}.",
            "experience": "Seek hands-on project experience through internships, freelancing, or open source.",
            "education": "Consider certifications or courses that strengthen your educational profile.",
        }

        if weakest[1] < 60:
            actions.append(f"[HIGHEST PRIORITY] {action_map[weakest[0]]}")

        for skill in skills.missing[:3]:
            actions.append(f"Learn {skill} — required for this position.")

        if skills.bonus:
            actions.append(
                f"Emphasize your bonus skills ({', '.join(skills.bonus[:3])}) "
                f"in your resume — they set you apart."
            )

        return actions

    # ── Helpers ───────────────────────────────────────────────

    def _assess_importance(self, skill: str, skills: SkillAnalysis) -> str:
        total_required = len(skills.matched) + len(skills.missing)
        if total_required == 0:
            return "medium"
        missing_ratio = len(skills.missing) / total_required
        if missing_ratio > 0.5:
            return "high"
        elif missing_ratio > 0.25:
            return "medium"
        return "low"

    def _skill_suggestion(self, skill: str, importance: str) -> str:
        if importance == "high":
            return f"Start learning {skill} immediately — it's critical for this role."
        elif importance == "medium":
            return f"Add {skill} to your learning roadmap within the next few months."
        return f"Consider picking up {skill} to round out your profile."

    def _estimate_learning_time(self, skill: str) -> str:
        complex_skills = {
            "machine learning", "deep learning", "kubernetes", "system design",
            "cloud architecture", "data engineering", "microservices",
        }
        if skill.lower() in complex_skills:
            return "3-6 months"
        return "2-4 weeks"

    def _build_llm_prompt(
        self, screening: ScreeningResult, focus_areas: Optional[list[str]]
    ) -> str:
        parts = [
            "Based on this resume screening result, provide personalized "
            "career improvement advice in 2-3 paragraphs:\n",
            f"Overall Score: {screening.scores.overall:.0f}/100",
            f"Matched Skills: {', '.join(screening.skill_analysis.matched[:10])}",
            f"Missing Skills: {', '.join(screening.skill_analysis.missing[:10])}",
            f"Predicted Role: {screening.predicted_role or 'N/A'}",
        ]
        if focus_areas:
            parts.append(f"Focus Areas: {', '.join(focus_areas)}")
        parts.append(
            "\nGive specific, actionable advice. Be encouraging but honest."
        )
        return "\n".join(parts)


# Singleton
feedback_generator = FeedbackGenerator()
