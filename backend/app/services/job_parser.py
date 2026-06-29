"""
Job Parser Service

Parses job descriptions to extract structured information.
Uses rule-based extraction for efficiency and reliability.
"""

from typing import List, Optional, Dict, Any, Tuple
import re
from datetime import datetime

from loguru import logger

from ..core.config import settings, Settings
from ..core.logging_config import get_logger
from ..schemas.job import JobDescriptionInput, ParsedJob


class JobParserService:
    """
    Service for parsing job descriptions into structured format.

    Features:
    - Skill extraction (required and preferred)
    - Responsibility extraction
    - Seniority detection
    - Experience requirement parsing
    - Industry classification
    - Technology identification
    """

    # Common skill patterns
    SKILL_PATTERNS = {
        "languages": [
            r"\bPython\b",
            r"\bJava\b",
            r"\bJavaScript\b",
            r"\bTypeScript\b",
            r"\bGo\b",
            r"\bRust\b",
            r"\bC\+\+\b",
            r"\bScala\b",
            r"\bR\b",
            r"\bSQL\b",
        ],
        "ml_frameworks": [
            r"\bPyTorch\b",
            r"\bTensorFlow\b",
            r"\bKeras\b",
            r"\bScikit-learn\b",
            r"\bXGBoost\b",
            r"\bLightGBM\b",
            r"\bHugging Face\b",
            r"\bTransformers\b",
            r"\bLangChain\b",
        ],
        "backend": [
            r"\bFastAPI\b",
            r"\bDjango\b",
            r"\bFlask\b",
            r"\bSpring\b",
            r"\bNode\.js\b",
            r"\bExpress\b",
            r"\bGraphQL\b",
            r"\bREST API\b",
        ],
        "data": [
            r"\bPandas\b",
            r"\bNumPy\b",
            r"\bSpark\b",
            r"\bKafka\b",
            r"\bAirflow\b",
            r"\bETL\b",
            r"\bData Pipeline\b",
        ],
        "cloud": [
            r"\bAWS\b",
            r"\bAzure\b",
            r"\bGCP\b",
            r"\bGoogle Cloud\b",
            r"\bKubernetes\b",
            r"\bDocker\b",
            r"\bTerraform\b",
        ],
        "vector_db": [
            r"\bChromaDB\b",
            r"\bPinecone\b",
            r"\bWeaviate\b",
            r"\bMilvus\b",
            r"\bQdrant\b",
            r"\bVector Database\b",
            r"\bVector Search\b",
            r"\bEmbedding\b",
        ],
        "llm": [
            r"\bLLM\b",
            r"\bLarge Language Model\b",
            r"\bGPT\b",
            r"\bBERT\b",
            r"\bRAG\b",
            r"\bRetrieval\b",
            r"\bGenerative AI\b",
            r"\bNLP\b",
        ],
    }

    # Seniority keywords
    SENIORITY_PATTERNS = {
        "entry": [r"\bentry[- ]?level\b", r"\bjunior\b", r"\bassociate\b", r"\b0[- ]?2\b"],
        "mid": [r"\bmid[- ]?level\b", r"\bmiddle\b", r"\b2[- ]?5\b", r"\b3[- ]?5\b"],
        "senior": [r"\bsenior\b", r"\blead\b", r"\b5[- ]?8\b", r"\b5\+\b"],
        "staff": [r"\bstaff\b", r"\bprincipal\b", r"\b8[- ]?10\b"],
        "executive": [r"\bhead\b", r"\bdirector\b", r"\bvp\b", r"\bchief\b", r"\bcto\b", r"\bceo\b"],
    }

    # Industry keywords
    INDUSTRY_PATTERNS = {
        "technology": [r"\btech\b", r"\bsoftware\b", r"\bsaas\b", r"\bplatform\b"],
        "finance": [r"\bfintech\b", r"\bfinance\b", r"\bbanking\b", r"\btrading\b"],
        "healthcare": [r"\bhealth\b", r"\bmedical\b", r"\bhealthcare\b", r"\bpharma\b"],
        "ecommerce": [r"\becommerce\b", r"\bretail\b", r"\bmarketplace\b"],
        "automotive": [r"\bautomotive\b", r"\bauto\b", r"\bvehicle\b"],
        "ai_ml": [r"\bai\b", r"\bmachine learning\b", r"\bml\b", r"\bdeep learning\b"],
        "data": [r"\bdata\b", r"\banalytics\b", r"\bbig data\b"],
    }

    # Education patterns
    EDUCATION_PATTERNS = [
        (r"\bBachelor['']?s?\b", "Bachelor's Degree"),
        (r"\bMaster['']?s?\b", "Master's Degree"),
        (r"\bPh\.?D\.?\b", "PhD"),
        (r"\bMBA\b", "MBA"),
        (r"\bMS\b", "Master's Degree"),
        (r"\bBS\b", "Bachelor's Degree"),
    ]

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the job parser service.

        Args:
            settings: Application settings
        """
        self.settings = settings or settings
        self.logger = get_logger(__name__)

    def parse(self, job_input: JobDescriptionInput) -> ParsedJob:
        """
        Parse a job description into structured format.

        Args:
            job_input: Raw job description input

        Returns:
            ParsedJob with extracted information
        """
        self.logger.info(f"Parsing job: {job_input.job_id} - {job_input.title}")
        start_time = datetime.now()

        description = job_input.description
        title = job_input.title
        combined_text = f"{title} {description}".lower()

        # Extract skills
        required_skills, preferred_skills = self._extract_skills(description)

        # Extract responsibilities
        responsibilities = self._extract_responsibilities(description)

        # Detect seniority
        seniority = self._detect_seniority(combined_text)

        # Extract years of experience
        years_experience = self._extract_years_experience(combined_text)

        # Detect industry
        industry = self._detect_industry(combined_text)

        # Extract education requirements
        education = self._extract_education(description)

        # Extract technologies
        technologies = self._extract_technologies(description)

        # Extract keywords
        keywords = self._extract_keywords(description)

        # Extract behavioral expectations
        behavioral_expectations = self._extract_behavioral_expectations(description)

        # Create processed document for embedding
        processed_document = self._create_processed_document(
            title=title,
            description=description,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            responsibilities=responsibilities,
            seniority=seniority,
            industry=industry,
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Job parsed in {elapsed:.3f}s")

        return ParsedJob(
            job_id=job_input.job_id,
            title=job_input.title,
            company=job_input.company,
            location=job_input.location,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            responsibilities=responsibilities,
            industry=industry,
            seniority=seniority,
            years_experience=years_experience,
            education=education,
            keywords=keywords,
            technologies=technologies,
            behavioral_expectations=behavioral_expectations,
            raw_description=description,
            processed_document=processed_document,
        )

    def _extract_skills(self, description: str) -> Tuple[List[str], List[str]]:
        """
        Extract required and preferred skills from description.

        Args:
            description: Job description text

        Returns:
            Tuple of (required_skills, preferred_skills)
        """
        required_skills = []
        preferred_skills = []

        description_lower = description.lower()

        # Check for required/preferred sections
        required_section = re.search(
            r"(?:required|must have|essential|necessary)[:\s]+([^.]+)",
            description,
            re.IGNORECASE,
        )
        preferred_section = re.search(
            r"(?:preferred|nice to have|plus|bonus)[:\s]+([^.]+)",
            description,
            re.IGNORECASE,
        )

        # Extract all skills from the description
        all_skills = []
        for category, patterns in self.SKILL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    # Get the skill name from pattern
                    skill_name = pattern.replace(r"\b", "").replace(r"\.", ".")
                    all_skills.append(skill_name)

        # Remove duplicates
        all_skills = list(dict.fromkeys(all_skills))

        # If we found sections, prioritize those
        if required_section:
            required_text = required_section.group(1).lower()
            required_skills = [s for s in all_skills if s.lower() in required_text]

        if preferred_section:
            preferred_text = preferred_section.group(1).lower()
            preferred_skills = [s for s in all_skills if s.lower() in preferred_text]

        # If no sections found, use all skills as required
        if not required_skills and not preferred_skills:
            required_skills = all_skills[:10]  # Top 10 skills
            preferred_skills = all_skills[10:]

        return required_skills, preferred_skills

    def _extract_responsibilities(self, description: str) -> List[str]:
        """
        Extract responsibilities from description.

        Args:
            description: Job description text

        Returns:
            List of responsibilities
        """
        responsibilities = []

        # Look for bullet points or numbered lists
        bullet_pattern = r"^[•\-\*]\s+(.+)$"
        number_pattern = r"^\d+[\.\\)]\s+(.+)$"

        for line in description.split("\n"):
            line = line.strip()
            match = re.match(bullet_pattern, line) or re.match(number_pattern, line)
            if match:
                resp = match.group(1).strip()
                if len(resp) > 10 and len(resp) < 200:
                    responsibilities.append(resp)

        # If no bullets found, look for responsibility keywords
        if not responsibilities:
            resp_keywords = [
                "responsible for",
                "you will",
                "your role",
                "key responsibilities",
                "duties include",
            ]
            for keyword in resp_keywords:
                pattern = rf"{keyword}[:\s]+([^.]+)"
                matches = re.findall(pattern, description, re.IGNORECASE)
                responsibilities.extend(matches)

        return responsibilities[:10]  # Limit to 10

    def _detect_seniority(self, text: str) -> Optional[str]:
        """
        Detect seniority level from text.

        Args:
            text: Combined job title and description

        Returns:
            Seniority level or None
        """
        for level, patterns in self.SENIORITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        return None

    def _extract_years_experience(self, text: str) -> Optional[int]:
        """
        Extract years of experience requirement.

        Args:
            text: Combined job title and description

        Returns:
            Years of experience or None
        """
        # Look for patterns like "5+ years", "3-5 years", etc.
        patterns = [
            r"(\d+)\+?\s*years?",
            r"(\d+)[- ](\d+)\s*years?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    return int(match.group(1))
                else:
                    # Return average of range
                    return (int(match.group(1)) + int(match.group(2))) // 2

        return None

    def _detect_industry(self, text: str) -> Optional[str]:
        """
        Detect industry from text.

        Args:
            text: Combined job title and description

        Returns:
            Industry or None
        """
        for industry, patterns in self.INDUSTRY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return industry
        return None

    def _extract_education(self, description: str) -> Optional[str]:
        """
        Extract education requirements.

        Args:
            description: Job description text

        Returns:
            Education requirement or None
        """
        for pattern, education in self.EDUCATION_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                return education
        return None

    def _extract_technologies(self, description: str) -> List[str]:
        """
        Extract technologies mentioned in description.

        Args:
            description: Job description text

        Returns:
            List of technologies
        """
        technologies = []

        tech_patterns = [
            r"\bPython\b",
            r"\bJava\b",
            r"\bJavaScript\b",
            r"\bTypeScript\b",
            r"\bPyTorch\b",
            r"\bTensorFlow\b",
            r"\bFastAPI\b",
            r"\bDjango\b",
            r"\bFlask\b",
            r"\bAWS\b",
            r"\bAzure\b",
            r"\bGCP\b",
            r"\bDocker\b",
            r"\bKubernetes\b",
            r"\bPostgreSQL\b",
            r"\bMongoDB\b",
            r"\bRedis\b",
            r"\bKafka\b",
            r"\bSpark\b",
            r"\bChromaDB\b",
            r"\bPinecone\b",
        ]

        for pattern in tech_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                tech_name = pattern.replace(r"\b", "")
                technologies.append(tech_name)

        return list(dict.fromkeys(technologies))

    def _extract_keywords(self, description: str) -> List[str]:
        """
        Extract important keywords from description.

        Args:
            description: Job description text

        Returns:
            List of keywords
        """
        # Common important keywords in job descriptions
        keywords = [
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "data science",
            "software engineering",
            "backend",
            "api",
            "microservices",
            "cloud",
            "scalable",
            "production",
            "mlops",
            "deployment",
            "agile",
            "collaboration",
            "leadership",
            "mentoring",
        ]

        found_keywords = []
        description_lower = description.lower()

        for keyword in keywords:
            if keyword in description_lower:
                found_keywords.append(keyword)

        return found_keywords

    def _extract_behavioral_expectations(self, description: str) -> List[str]:
        """
        Extract behavioral expectations from description.

        Args:
            description: Job description text

        Returns:
            List of behavioral expectations
        """
        behavioral_keywords = [
            "communication",
            "teamwork",
            "collaboration",
            "leadership",
            "mentoring",
            "problem-solving",
            "analytical",
            "detail-oriented",
            "self-motivated",
            "proactive",
            "adaptability",
            "innovation",
        ]

        found = []
        description_lower = description.lower()

        for keyword in behavioral_keywords:
            if keyword in description_lower:
                found.append(keyword)

        return found

    def _create_processed_document(
        self,
        title: str,
        description: str,
        required_skills: List[str],
        preferred_skills: List[str],
        responsibilities: List[str],
        seniority: Optional[str],
        industry: Optional[str],
    ) -> str:
        """
        Create a processed document optimized for embedding.

        Args:
            title: Job title
            description: Raw description
            required_skills: Required skills
            preferred_skills: Preferred skills
            responsibilities: Responsibilities
            seniority: Seniority level
            industry: Industry

        Returns:
            Processed document string
        """
        parts = [
            f"Job Title: {title}",
            f"Seniority: {seniority or 'Not specified'}",
            f"Industry: {industry or 'Not specified'}",
            f"Required Skills: {', '.join(required_skills)}",
            f"Preferred Skills: {', '.join(preferred_skills)}",
            f"Responsibilities: {'; '.join(responsibilities)}",
            f"Description: {description}",
        ]

        return " | ".join(parts)


# Global instance
_job_parser: Optional[JobParserService] = None


def get_job_parser(settings: Optional[Settings] = None) -> JobParserService:
    """
    Get or create the job parser service singleton.

    Args:
        settings: Optional settings override

    Returns:
        JobParserService instance
    """
    global _job_parser

    if _job_parser is None:
        _job_parser = JobParserService(settings)

    return _job_parser
