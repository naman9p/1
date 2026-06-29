"""
Behavioral Signal Parser

Parses and normalizes behavioral signals from candidate data.
"""

from typing import Optional, Dict, Any, List
import math
from datetime import datetime, timedelta

from loguru import logger

from ..core.config import settings, Settings
from ..core.logging_config import get_logger
from ..schemas.candidate import CandidateProfile, BehavioralSignals


class BehavioralSignalParser:
    """
    Service for parsing and computing behavioral signals.

    Features:
    - Signal normalization (0-1 scale)
    - Composite score calculation
    - Recency scoring
    - Trust score computation
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the behavioral signal parser.

        Args:
            settings: Application settings
        """
        self.settings = settings or settings
        self.logger = get_logger(__name__)

    def parse(self, profile: CandidateProfile) -> BehavioralSignals:
        """
        Parse behavioral signals from a candidate profile.

        Args:
            profile: Candidate profile

        Returns:
            BehavioralSignals with normalized scores
        """
        self.logger.debug(f"Parsing behavioral signals for: {profile.candidate_id}")

        # Extract or compute individual signals
        signals = BehavioralSignals()

        # If signals already exist in profile, use them
        if profile.behavioral_signals:
            existing = profile.behavioral_signals
            signals.recruiter_response_rate = self._normalize(existing.recruiter_response_rate)
            signals.profile_completeness = self._normalize(existing.profile_completeness)
            signals.activity_score = self._normalize(existing.activity_score)
            signals.hiring_interaction = self._normalize(existing.hiring_interaction)
            signals.message_acceptance = self._normalize(existing.message_acceptance)
            signals.application_quality = self._normalize(existing.application_quality)
            signals.recency_score = self._normalize(existing.recency_score)
            signals.consistency_score = self._normalize(existing.consistency_score)
            signals.trust_score = self._normalize(existing.trust_score)
        else:
            # Compute signals from profile data
            signals = self._compute_signals(profile)

        # Compute composite behavior score
        signals.behavior_score = self._compute_behavior_score(signals)

        return signals

    def _normalize(self, value: float) -> float:
        """Normalize a value to 0-1 range."""
        return max(0.0, min(1.0, value))

    def _compute_signals(self, profile: CandidateProfile) -> BehavioralSignals:
        """
        Compute behavioral signals from profile data.

        Args:
            profile: Candidate profile

        Returns:
            BehavioralSignals with computed values
        """
        signals = BehavioralSignals()

        # Profile completeness score
        signals.profile_completeness = self._compute_profile_completeness(profile)

        # Recency score based on last update
        signals.recency_score = self._compute_recency_score(profile.last_updated)

        # Activity score based on profile activity indicators
        signals.activity_score = self._compute_activity_score(profile)

        # Consistency score based on profile consistency
        signals.consistency_score = self._compute_consistency_score(profile)

        # Trust score as composite of other signals
        signals.trust_score = self._compute_trust_score(signals)

        # Default values for signals we can't compute from profile
        signals.recruiter_response_rate = 0.5  # Default neutral
        signals.hiring_interaction = 0.5
        signals.message_acceptance = 0.5
        signals.application_quality = signals.profile_completeness  # Use completeness as proxy

        return signals

    def _compute_profile_completeness(self, profile: CandidateProfile) -> float:
        """
        Compute profile completeness score.

        Args:
            profile: Candidate profile

        Returns:
            Completeness score (0-1)
        """
        score = 0.0
        max_score = 10.0

        # Check various profile fields
        if profile.name:
            score += 1.0
        if profile.headline:
            score += 1.0
        if profile.summary and len(profile.summary) > 50:
            score += 1.5
        if profile.location:
            score += 0.5
        if profile.work_experience:
            score += min(2.0, len(profile.work_experience) * 0.5)
        if profile.education:
            score += min(1.5, len(profile.education) * 0.5)
        if profile.skills and len(profile.skills) > 5:
            score += 1.5
        elif profile.skills:
            score += 0.5
        if profile.certifications:
            score += min(1.0, len(profile.certifications) * 0.3)
        if profile.projects:
            score += min(1.0, len(profile.projects) * 0.3)

        return self._normalize(score / max_score)

    def _compute_recency_score(self, last_updated: Optional[datetime]) -> float:
        """
        Compute recency score based on last update time.

        Args:
            last_updated: Last update timestamp

        Returns:
            Recency score (0-1, higher = more recent)
        """
        if not last_updated:
            return 0.3  # Default for unknown

        now = datetime.utcnow()
        days_since_update = (now - last_updated).days

        # Exponential decay: recent updates score higher
        if days_since_update <= 7:
            return 1.0
        elif days_since_update <= 30:
            return 0.9
        elif days_since_update <= 90:
            return 0.7
        elif days_since_update <= 180:
            return 0.5
        elif days_since_update <= 365:
            return 0.3
        else:
            return 0.1

    def _compute_activity_score(self, profile: CandidateProfile) -> float:
        """
        Compute activity score from profile indicators.

        Args:
            profile: Candidate profile

        Returns:
            Activity score (0-1)
        """
        score = 0.0
        max_score = 5.0

        # Recent work experience indicates activity
        if profile.work_experience:
            current = any(exp.is_current for exp in profile.work_experience)
            if current:
                score += 1.5

        # Projects indicate activity
        if profile.projects:
            score += min(1.5, len(profile.projects) * 0.3)

        # Skills endorsements indicate engagement
        if profile.endorsements:
            total_endorsements = sum(profile.endorsements.values())
            if total_endorsements > 50:
                score += 1.5
            elif total_endorsements > 20:
                score += 1.0
            elif total_endorsements > 5:
                score += 0.5

        # Publications/patents indicate activity
        if profile.publications:
            score += min(0.5, len(profile.publications) * 0.1)
        if profile.patents:
            score += min(0.5, len(profile.patents) * 0.1)

        return self._normalize(score / max_score)

    def _compute_consistency_score(self, profile: CandidateProfile) -> float:
        """
        Compute consistency score from profile data.

        Args:
            profile: Candidate profile

        Returns:
            Consistency score (0-1)
        """
        score = 1.0

        # Check for employment gaps
        if profile.work_experience:
            gaps = self._detect_employment_gaps(profile.work_experience)
            if gaps > 2:
                score -= 0.3
            elif gaps > 0:
                score -= 0.1

        # Check for job hopping (many short tenures)
        short_tenures = self._count_short_tenures(profile.work_experience)
        if short_tenures > 3:
            score -= 0.2
        elif short_tenures > 1:
            score -= 0.1

        # Check skill consistency with experience
        if profile.skills and profile.work_experience:
            # Simple heuristic: more skills than expected for experience level
            expected_skills = max(5, profile.total_years_experience or 1 * 2)
            if len(profile.skills) < expected_skills * 0.5:
                score -= 0.1

        return self._normalize(score)

    def _detect_employment_gaps(
        self, experiences: List[Any]
    ) -> int:
        """
        Detect employment gaps in work history.

        Args:
            experiences: List of work experiences

        Returns:
            Number of significant gaps (>3 months)
        """
        if len(experiences) < 2:
            return 0

        gaps = 0
        sorted_exp = sorted(
            [e for e in experiences if e.start_date],
            key=lambda x: x.start_date or "",
            reverse=True,
        )

        for i in range(len(sorted_exp) - 1):
            current = sorted_exp[i]
            next_exp = sorted_exp[i + 1]

            if current.end_date and next_exp.start_date:
                try:
                    # Simple date parsing
                    end = datetime.strptime(current.end_date, "%Y-%m")
                    start = datetime.strptime(next_exp.start_date, "%Y-%m")
                    gap_months = (end.year - start.year) * 12 + (end.month - start.month)

                    if gap_months > 3:
                        gaps += 1
                except (ValueError, AttributeError):
                    pass

        return gaps

    def _count_short_tenures(self, experiences: List[Any]) -> int:
        """
        Count short tenures (<1 year) in work history.

        Args:
            experiences: List of work experiences

        Returns:
            Number of short tenures
        """
        count = 0
        for exp in experiences:
            if exp.duration_months and exp.duration_months < 12:
                count += 1
        return count

    def _compute_trust_score(self, signals: BehavioralSignals) -> float:
        """
        Compute overall trust score from individual signals.

        Args:
            signals: Behavioral signals

        Returns:
            Trust score (0-1)
        """
        weights = {
            "profile_completeness": 0.20,
            "recency_score": 0.15,
            "activity_score": 0.15,
            "consistency_score": 0.20,
            "recruiter_response_rate": 0.10,
            "message_acceptance": 0.10,
            "application_quality": 0.10,
        }

        score = 0.0
        for signal_name, weight in weights.items():
            value = getattr(signals, signal_name, 0.5)
            score += value * weight

        return self._normalize(score)

    def _compute_behavior_score(self, signals: BehavioralSignals) -> float:
        """
        Compute composite behavior score.

        Args:
            signals: Behavioral signals

        Returns:
            Composite behavior score (0-1)
        """
        # Weighted average of key signals
        weights = {
            "profile_completeness": 0.15,
            "activity_score": 0.15,
            "recruiter_response_rate": 0.15,
            "message_acceptance": 0.15,
            "application_quality": 0.15,
            "recency_score": 0.10,
            "consistency_score": 0.10,
            "trust_score": 0.05,
        }

        score = 0.0
        for signal_name, weight in weights.items():
            value = getattr(signals, signal_name, 0.5)
            score += value * weight

        return self._normalize(score)


# Global instance
_behavioral_parser: Optional[BehavioralSignalParser] = None


def get_behavioral_parser(settings: Optional[Settings] = None) -> BehavioralSignalParser:
    """
    Get or create the behavioral signal parser singleton.

    Args:
        settings: Optional settings override

    Returns:
        BehavioralSignalParser instance
    """
    global _behavioral_parser

    if _behavioral_parser is None:
        _behavioral_parser = BehavioralSignalParser(settings)

    return _behavioral_parser
