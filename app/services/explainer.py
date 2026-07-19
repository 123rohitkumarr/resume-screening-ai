"""
Explainable AI engine — produces human-readable score attributions.

Uses a combination of deterministic rule-based attribution (for the
composite score dimensions) and optional SHAP values (when the ML
classifier is available) to explain *why* a candidate received a
particular score.
"""

import math
from typing import Optional
from app.models.schemas import ScoreBreakdown, SkillAnalysis, ExplanationItem
from app.config import settings
from app.utils.helpers import setup_logger, clamp

log = setup_logger(__name__)


# ── Education ordinal scale ───────────────────────────────────

EDUCATION_RANK = {
    "high school": 1,
    "associate": 2,
    "bachelors": 3,
    "bachelor": 3,
    "masters": 4,
    "master": 4,
    "phd": 5,
    "doctorate": 5,
}


def _edu_label(level: str) -> str:
    return level.strip().title() if level else "Not specified"


def _impact_label(score: float) -> str:
    if score >= 75:
        return "positive"
    elif score >= 45:
        return "neutral"
    return "negative"


# ── Main explainer ────────────────────────────────────────────

class ScoreExplainer:
    """Generates structured explanations for each scoring dimension."""

    def explain(
        self,
        scores: ScoreBreakdown,
        skill_analysis: SkillAnalysis,
        candidate_exp: Optional[float] = None,
        required_exp: Optional[float] = None,
        candidate_edu: Optional[str] = None,
        required_edu: Optional[str] = None,
    ) -> list[ExplanationItem]:
        items = []

        items.append(self._explain_semantic(scores.semantic))
        items.append(self._explain_skills(scores.skill, skill_analysis))
        items.append(self._explain_experience(
            scores.experience, candidate_exp, required_exp
        ))
        items.append(self._explain_education(
            scores.education, candidate_edu, required_edu
        ))

        return items

    def generate_summary(
        self,
        scores: ScoreBreakdown,
        skill_analysis: SkillAnalysis,
        predicted_role: Optional[str] = None,
    ) -> str:
        """One-paragraph natural-language summary of the screening."""
        parts = []

        # Overall verdict
        overall = scores.overall
        if overall >= 80:
            parts.append(f"This candidate is an excellent match with a score of {overall:.0f}/100.")
        elif overall >= 60:
            parts.append(f"This candidate is a good match with a score of {overall:.0f}/100.")
        elif overall >= 40:
            parts.append(f"This candidate is a partial match with a score of {overall:.0f}/100.")
        else:
            parts.append(f"This candidate has limited compatibility with a score of {overall:.0f}/100.")

        if predicted_role:
            parts.append(f"Their profile aligns with a {predicted_role} role.")

        # Skills detail
        n_matched = len(skill_analysis.matched)
        n_missing = len(skill_analysis.missing)
        if n_matched and not n_missing:
            parts.append("They possess all required skills.")
        elif n_matched:
            parts.append(
                f"They match {n_matched} required skill(s) but are missing "
                f"{n_missing}: {', '.join(skill_analysis.missing[:5])}."
            )
        elif n_missing:
            parts.append(
                f"They are missing {n_missing} required skill(s) including "
                f"{', '.join(skill_analysis.missing[:3])}."
            )

        if skill_analysis.bonus:
            parts.append(
                f"They also bring bonus skills: {', '.join(skill_analysis.bonus[:4])}."
            )

        return " ".join(parts)

    def generate_suggestions(
        self,
        scores: ScoreBreakdown,
        skill_analysis: SkillAnalysis,
        candidate_exp: Optional[float] = None,
        required_exp: Optional[float] = None,
    ) -> list[str]:
        """Actionable improvement recommendations."""
        suggestions = []

        # Skill gaps
        for skill in skill_analysis.missing[:5]:
            suggestions.append(
                f"Learn {skill} — this is a required skill for the position."
            )

        # Experience gap
        if candidate_exp is not None and required_exp is not None:
            gap = required_exp - candidate_exp
            if gap > 0:
                suggestions.append(
                    f"Gain approximately {gap:.0f} more year(s) of relevant experience, "
                    f"or highlight transferable experience from adjacent domains."
                )

        # Semantic alignment
        if scores.semantic < 50:
            suggestions.append(
                "Tailor your resume language to mirror the job description's "
                "terminology and focus areas."
            )

        # Education
        if scores.education < 50:
            suggestions.append(
                "Consider pursuing additional certifications or education "
                "relevant to this role."
            )

        # General
        if scores.overall < 60:
            suggestions.append(
                "Build portfolio projects demonstrating the key skills "
                "listed in the job description."
            )

        return suggestions if suggestions else [
            "Your profile is well-aligned. Keep your skills current and "
            "consider specializing further in your strongest areas."
        ]

    # ── Private helpers ───────────────────────────────────────

    def _explain_semantic(self, score: float) -> ExplanationItem:
        if score >= 75:
            detail = (
                "Your resume content closely matches the job description's "
                "language and focus areas."
            )
        elif score >= 50:
            detail = (
                "There is moderate alignment between your resume and the "
                "job description. Consider using more relevant terminology."
            )
        else:
            detail = (
                "Your resume language differs significantly from the job "
                "description. Tailoring your resume could improve this score."
            )

        return ExplanationItem(
            factor="Semantic Similarity",
            score=score,
            impact=_impact_label(score),
            detail=detail,
        )

    def _explain_skills(
        self, score: float, analysis: SkillAnalysis
    ) -> ExplanationItem:
        matched = len(analysis.matched)
        missing = len(analysis.missing)
        total = matched + missing

        if total == 0:
            detail = "No specific skills could be compared."
        elif missing == 0:
            detail = f"You possess all {matched} required skills."
        else:
            detail = (
                f"You match {matched}/{total} required skills. "
                f"Missing: {', '.join(analysis.missing[:5])}."
            )

        return ExplanationItem(
            factor="Skill Match",
            score=score,
            impact=_impact_label(score),
            detail=detail,
        )

    def _explain_experience(
        self,
        score: float,
        candidate: Optional[float],
        required: Optional[float],
    ) -> ExplanationItem:
        if candidate is None or required is None:
            detail = "Experience information was not fully available for comparison."
        elif candidate >= required:
            ovr = candidate - required
            if ovr <= 3:
                detail = (
                    f"You meet the experience requirement ({candidate:.0f} "
                    f"vs {required:.0f} years needed)."
                )
            else:
                detail = (
                    f"You exceed the requirement by {ovr:.0f} years — you "
                    f"may be overqualified for this position."
                )
        else:
            gap = required - candidate
            detail = (
                f"You're {gap:.0f} year(s) short of the required {required:.0f} "
                f"years. Highlight transferable experience to compensate."
            )

        return ExplanationItem(
            factor="Experience Fit",
            score=score,
            impact=_impact_label(score),
            detail=detail,
        )

    def _explain_education(
        self,
        score: float,
        candidate_edu: Optional[str],
        required_edu: Optional[str],
    ) -> ExplanationItem:
        c_label = _edu_label(candidate_edu)
        r_label = _edu_label(required_edu)

        if not candidate_edu or not required_edu:
            detail = "Education comparison could not be fully evaluated."
        elif score >= 90:
            detail = f"Your education ({c_label}) meets or exceeds the requirement ({r_label})."
        elif score >= 60:
            detail = (
                f"Your education ({c_label}) is close to but below the "
                f"preferred level ({r_label})."
            )
        else:
            detail = (
                f"Your education ({c_label}) falls short of the "
                f"requirement ({r_label}). Relevant certifications may help."
            )

        return ExplanationItem(
            factor="Education Alignment",
            score=score,
            impact=_impact_label(score),
            detail=detail,
        )


# Singleton
explainer = ScoreExplainer()
