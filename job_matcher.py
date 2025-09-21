# job_matcher.py - COMPLETE FIXED VERSION
import json
import re
import hashlib
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JDModel(BaseModel):
    required_skills: List[str]
    preferred_skills: List[str]
    experience_required: str
    education_required: str
    job_title: str
    company: str

class JobMatcher:
    def __init__(self):
        # Configure Generative AI with retry logic
        self.model = None
        self._init_gemini()
        
        # Text vectorizers
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Lazy-load embedding model
        self.embed_model: Optional[SentenceTransformer] = None
        
        # Cache parsed JDs
        self._jd_cache: Dict[str, Dict] = {}
        
        # Enhanced skill patterns for fallback parsing
        self.comprehensive_skills = [
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'go',
            'rust', 'scala', 'kotlin', 'sql', 'r', 'swift', 'dart', 'php', 'ruby',
            # Web Technologies  
            'react', 'angular', 'vue', 'node.js', 'nodejs', 'express', 'django',
            'flask', 'fastapi', 'spring', 'next.js', 'html', 'css', 'bootstrap',
            # Data Science & ML
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            'xgboost', 'lightgbm', 'mlflow', 'jupyter', 'tableau', 'power bi',
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
            'jenkins', 'github actions', 'gitlab ci', 'helm', 'prometheus',
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'snowflake', 'bigquery',
            'elasticsearch', 'cassandra', 'dynamodb',
            # Specialized
            'restful api', 'graphql', 'grpc', 'microservices', 'machine learning',
            'deep learning', 'nlp', 'computer vision', 'blockchain', 'agile', 'scrum'
        ]
    
    def _init_gemini(self):
        """Initialize Gemini with error handling"""
        try:
            if Config.GEMINI_API_KEY:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini AI initialized successfully")
        except Exception as e:
            logger.warning(f"Gemini initialization failed: {e}")
            self.model = None
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    
    def parse_job_description(self, jd_text: str) -> Dict:
        """Enhanced JD parsing with better fallback"""
        key = self._hash_text(jd_text)
        if key in self._jd_cache:
            return self._jd_cache[key]
        
        # Try Gemini first
        jd_data = self._parse_with_gemini(jd_text)
        
        # Use enhanced fallback if Gemini fails
        if not jd_data:
            jd_data = self._enhanced_fallback_parsing(jd_text)
        
        # Normalize and validate
        jd_data = self._normalize_jd_data(jd_data)
        
        self._jd_cache[key] = jd_data
        return jd_data
    
    def _parse_with_gemini(self, jd_text: str) -> Optional[Dict]:
        """Parse JD using Gemini AI"""
        if not self.model:
            return None
        
        prompt = f"""
        Analyze this job description and extract information as JSON:
        
        {{
            "required_skills": ["skill1", "skill2", ...],
            "preferred_skills": ["skill1", "skill2", ...],
            "experience_required": "X years" or "Not specified",
            "education_required": "degree level" or "Not specified", 
            "job_title": "extracted title",
            "company": "company name" or "Not specified"
        }}
        
        Job Description:
        {jd_text[:3000]}
        
        Return ONLY the JSON object.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000
                )
            )
            
            content = response.text.strip()
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                
                # Validate required keys
                required_keys = ['required_skills', 'preferred_skills', 'experience_required', 
                               'education_required', 'job_title', 'company']
                if all(key in data for key in required_keys):
                    return data
                    
        except Exception as e:
            logger.warning(f"Gemini parsing failed: {e}")
        
        return None
    
    def _enhanced_fallback_parsing(self, jd_text: str) -> Dict:
        """Enhanced regex-based fallback parsing"""
        text_lower = jd_text.lower()
        
        # Extract skills with better matching
        found_skills = []
        for skill in self.comprehensive_skills:
            if len(skill.split()) > 1:
                # Multi-word skills
                if skill in text_lower:
                    found_skills.append(skill)
            else:
                # Single word skills with word boundaries
                if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                    found_skills.append(skill)
        
        # Split skills into required vs preferred
        # Skills in first half of JD are more likely required
        midpoint = len(jd_text) // 2
        first_half = jd_text[:midpoint].lower()
        
        required_skills = []
        preferred_skills = []
        
        for skill in found_skills:
            if skill in first_half:
                required_skills.append(skill)
            else:
                preferred_skills.append(skill)
        
        # If no clear split, use first 2/3 as required
        if not required_skills and found_skills:
            split_point = len(found_skills) * 2 // 3
            required_skills = found_skills[:split_point]
            preferred_skills = found_skills[split_point:]
        
        # Extract experience
        experience = self._extract_experience_requirement(jd_text)
        
        # Extract education
        education = self._extract_education_requirement(jd_text)
        
        # Extract job title
        job_title = self._extract_job_title(jd_text)
        
        # Extract company
        company = self._extract_company_name(jd_text)
        
        return {
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "experience_required": experience,
            "education_required": education,
            "job_title": job_title,
            "company": company
        }
    
    def _extract_experience_requirement(self, text: str) -> str:
        """Extract experience requirements"""
        patterns = [
            r'(\d+(?:-\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'minimum\s*(?:of\s*)?(\d+)\s*(?:years?|yrs?)',
            r'at least\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+\s*(?:years?|yrs?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return f"{match.group(1)} years"
        
        return "Not specified"
    
    def _extract_education_requirement(self, text: str) -> str:
        """Extract education requirements"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['phd', 'ph.d', 'doctorate', 'doctoral']):
            return "PhD"
        elif any(term in text_lower for term in ['master', 'm.tech', 'mtech', 'm.sc', 'mba']):
            return "Master's degree"
        elif any(term in text_lower for term in ['bachelor', 'b.tech', 'btech', 'b.e', 'degree']):
            return "Bachelor's degree"
        
        return "Not specified"
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title from JD"""
        lines = text.split('\n')[:5]  # Check first few lines
        
        title_indicators = ['job title:', 'position:', 'role:', 'hiring for']
        
        for line in lines:
            line_lower = line.lower().strip()
            for indicator in title_indicators:
                if indicator in line_lower:
                    title = line.split(':', 1)[-1].strip()
                    if title:
                        return title
        
        # If no explicit title found, use first meaningful line
        for line in lines:
            if len(line.strip()) > 5 and not line.startswith(('http', 'www', 'email')):
                return line.strip()
        
        return "Not specified"
    
    def _extract_company_name(self, text: str) -> str:
        """Extract company name"""
        lines = text.split('\n')[:10]
        
        company_indicators = ['company:', 'organization:', 'employer:', 'at ']
        
        for line in lines:
            line_lower = line.lower().strip()
            for indicator in company_indicators:
                if indicator in line_lower:
                    company = line.split(':', 1)[-1].strip()
                    if company:
                        return company
        
        return "Not specified"
    
    def _normalize_jd_data(self, jd_data: Dict) -> Dict:
        """Normalize and clean JD data"""
        # Ensure all required keys exist
        default_jd = {
            "required_skills": [],
            "preferred_skills": [],
            "experience_required": "Not specified",
            "education_required": "Not specified",
            "job_title": "Not specified",
            "company": "Not specified"
        }
        
        for key, default_value in default_jd.items():
            if key not in jd_data:
                jd_data[key] = default_value
        
        # Normalize skills
        jd_data["required_skills"] = sorted(list(set(
            s.strip().lower() for s in jd_data["required_skills"] if s and s.strip()
        )))
        jd_data["preferred_skills"] = sorted(list(set(
            s.strip().lower() for s in jd_data["preferred_skills"] if s and s.strip()
        )))
        
        return jd_data
    
    def _bm25_score(self, resume_text: str, jd_terms: List[str]) -> float:
        """Calculate BM25 score"""
        if not jd_terms or not resume_text.strip():
            return 0.0
        
        try:
            tokens = re.findall(r'\b\w+\b', resume_text.lower())
            if not tokens:
                return 0.0
            
            corpus = [tokens]
            bm25 = BM25Okapi(corpus)
            score = bm25.get_scores(jd_terms)[0]
            
            # Normalize to 0-100 scale
            return max(0.0, min(100.0, score * 15.0))
        except Exception:
            return 0.0
    
    def calculate_hard_match_score(self, resume_data: Dict, jd_data: Dict) -> Dict:
        """Enhanced hard matching with multiple techniques"""
        resume_skills = set(s.lower().strip() for s in resume_data.get('skills', []))
        required_skills = jd_data.get('required_skills', [])
        preferred_skills = jd_data.get('preferred_skills', [])
        
        # Exact matches
        exact_required = resume_skills.intersection(set(required_skills))
        exact_preferred = resume_skills.intersection(set(preferred_skills))
        
        # Fuzzy matching for variations
        def fuzzy_matches(resume_skills: set, target_skills: List[str], threshold: int = 80) -> set:
            matches = set()
            for target in target_skills:
                for resume_skill in resume_skills:
                    if fuzz.token_set_ratio(resume_skill, target) >= threshold:
                        matches.add(target)
                        break
            return matches
        
        fuzzy_required = fuzzy_matches(resume_skills, required_skills)
        fuzzy_preferred = fuzzy_matches(resume_skills, preferred_skills)
        
        # BM25 scoring
        resume_text = resume_data.get('raw_text', '')
        bm25_required = self._bm25_score(resume_text, required_skills)
        bm25_preferred = self._bm25_score(resume_text, preferred_skills)
        
        # Calculate coverage scores
        total_required = len(required_skills) if required_skills else 1
        total_preferred = len(preferred_skills) if preferred_skills else 1
        
        req_matches = exact_required.union(fuzzy_required)
        pref_matches = exact_preferred.union(fuzzy_preferred)
        
        req_coverage = (len(req_matches) / total_required) * 100.0
        pref_coverage = (len(pref_matches) / total_preferred) * 100.0
        
        # Weighted hard score
        hard_score = (
            0.5 * req_coverage +        # 50% - required skills coverage
            0.2 * pref_coverage +       # 20% - preferred skills coverage  
            0.2 * bm25_required +       # 20% - BM25 required
            0.1 * bm25_preferred        # 10% - BM25 preferred
        )
        
        return {
            "hard_score": round(hard_score, 2),
            "required_matches": sorted(req_matches),
            "preferred_matches": sorted(pref_matches),
            "missing_required": sorted(set(required_skills) - req_matches),
            "missing_preferred": sorted(set(preferred_skills) - pref_matches),
            "req_coverage": round(req_coverage, 2),
            "pref_coverage": round(pref_coverage, 2)
        }
    
    def _get_embed_model(self) -> Optional[SentenceTransformer]:
        """Lazy load embedding model"""
        if self.embed_model is not None:
            return self.embed_model
        
        try:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embed_model = None
        
        return self.embed_model
    
    def calculate_semantic_score(self, resume_data: Dict, jd_text: str) -> float:
        """Enhanced semantic matching"""
        resume_text = resume_data.get('raw_text', '').strip()
        jd_text = jd_text.strip()
        
        if not resume_text or not jd_text:
            return 0.0
        
        # Try embedding-based similarity first
        try:
            model = self._get_embed_model()
            if model is not None:
                embeddings = model.encode([resume_text, jd_text], normalize_embeddings=True)
                similarity = float(np.dot(embeddings[0], embeddings[1]))
                # Convert from [-1, 1] to [0, 100]
                return round((similarity + 1.0) * 50.0, 2)
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}")
        
        # Fallback to TF-IDF
        try:
            documents = [resume_text, jd_text]
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return round(similarity * 100.0, 2)
        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}")
            return 0.0
    
    def _calculate_verdict(self, score: float) -> str:
        """Determine verdict based on score"""
        if score >= Config.HIGH_THRESHOLD:
            return "High"
        elif score >= Config.MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "Low"
    
    def _generate_feedback(self, resume_data: Dict, jd_data: Dict, hard_match: Dict) -> str:
        """Generate actionable feedback"""
        if not self.model:
            return self._generate_fallback_feedback(resume_data, jd_data, hard_match)
        
        missing_required = hard_match.get("missing_required", [])[:5]
        resume_skills = resume_data.get('skills', [])
        
        prompt = f"""
        Generate 3-5 specific, actionable recommendations to improve resume relevance:
        
        Current skills: {resume_skills}
        Required but missing: {missing_required}
        Job requirements: {jd_data.get('required_skills', [])}
        
        Focus on:
        1. Specific skills to learn/certify
        2. Project suggestions
        3. Experience improvements
        4. Resume presentation tips
        
        Keep recommendations concise and actionable. Use bullet points.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )
            feedback = response.text.strip()
            if feedback:
                return feedback
        except Exception as e:
            logger.warning(f"Feedback generation failed: {e}")
        
        return self._generate_fallback_feedback(resume_data, jd_data, hard_match)
    
    def _generate_fallback_feedback(self, resume_data: Dict, jd_data: Dict, hard_match: Dict) -> str:
        """Generate basic feedback when AI is unavailable"""
        missing_required = hard_match.get("missing_required", [])
        
        if not missing_required:
            return "Strong skill alignment! Consider adding quantifiable achievements and relevant project examples to strengthen your application."
        
        feedback_parts = []
        
        if len(missing_required) <= 3:
            feedback_parts.append(f"• Develop skills in: {', '.join(missing_required)}")
            feedback_parts.append(f"• Create a project demonstrating {missing_required[0]} proficiency")
        else:
            top_missing = missing_required[:3]
            feedback_parts.append(f"• Priority skills to develop: {', '.join(top_missing)}")
            feedback_parts.append(f"• Consider certification in {top_missing[0]}")
        
        feedback_parts.append("• Add quantifiable achievements to your experience section")
        feedback_parts.append("• Include relevant project links and detailed descriptions")
        
        return "\n".join(feedback_parts)
    
    def evaluate_resume(self, resume_data: Dict, jd_text: str) -> Dict:
        """Main evaluation method with comprehensive scoring"""
        try:
            # Parse job description
            jd_data = self.parse_job_description(jd_text)
            
            # Calculate component scores
            hard_match = self.calculate_hard_match_score(resume_data, jd_data)
            semantic_score = self.calculate_semantic_score(resume_data, jd_text)
            
            # Calculate overall score
            overall_score = (
                hard_match['hard_score'] * Config.HARD_MATCH_WEIGHT + 
                semantic_score * Config.SEMANTIC_MATCH_WEIGHT
            )
            
            # Generate feedback
            feedback = self._generate_feedback(resume_data, jd_data, hard_match)
            
            # Compile results
            evaluation = {
                "job_details": jd_data,
                "hard_match_score": hard_match['hard_score'],
                "semantic_score": semantic_score,
                "overall_score": round(overall_score, 2),
                "verdict": self._calculate_verdict(overall_score),
                "required_matches": hard_match["required_matches"],
                "preferred_matches": hard_match["preferred_matches"],
                "missing_required": hard_match["missing_required"],
                "missing_preferred": hard_match["missing_preferred"],
                "feedback": feedback,
                "metrics": {
                    "req_coverage": hard_match.get("req_coverage", 0),
                    "pref_coverage": hard_match.get("pref_coverage", 0),
                    "total_skills_found": len(resume_data.get('skills', [])),
                    "skill_match_ratio": len(hard_match["required_matches"]) / max(len(jd_data.get("required_skills", [])), 1)
                }
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "job_details": {},
                "hard_match_score": 0.0,
                "semantic_score": 0.0,
                "overall_score": 0.0,
                "verdict": "Error",
                "required_matches": [],
                "preferred_matches": [],
                "missing_required": [],
                "missing_preferred": [],
                "feedback": f"Evaluation failed: {str(e)}",
                "metrics": {}
            }
