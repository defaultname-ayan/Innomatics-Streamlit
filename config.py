# config.py - ENHANCED VERSION
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        print("⚠️  WARNING: GEMINI_API_KEY not found in environment variables")
        print("   Create a .env file with: GEMINI_API_KEY=your_api_key_here")
        print("   AI-powered features will use fallback methods")
    
    # Directory paths
    UPLOAD_FOLDER = "uploads"
    DATA_FOLDER = "data"
    
    # Ensure directories exist
    Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
    Path(DATA_FOLDER).mkdir(exist_ok=True)
    
    # Scoring weights (must sum to 1.0)
    HARD_MATCH_WEIGHT = 0.4
    SEMANTIC_MATCH_WEIGHT = 0.6
    
    # Performance thresholds
    HIGH_THRESHOLD = 75
    MEDIUM_THRESHOLD = 50
    LOW_THRESHOLD = 0
    
    # Model configurations
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_FILE_SIZE_MB = 10
    
    # UI Settings
    MAX_RESUMES_PER_BATCH = 20
    PAGINATION_SIZE = 10
    
    # Cache settings
    CACHE_TTL_SECONDS = 3600  # 1 hour
    MAX_CACHE_SIZE = 100
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        if cls.HARD_MATCH_WEIGHT + cls.SEMANTIC_MATCH_WEIGHT != 1.0:
            errors.append("Scoring weights must sum to 1.0")
        
        if not (0 <= cls.HIGH_THRESHOLD <= 100):
            errors.append("HIGH_THRESHOLD must be between 0 and 100")
            
        if not (0 <= cls.MEDIUM_THRESHOLD <= 100):
            errors.append("MEDIUM_THRESHOLD must be between 0 and 100")
            
        if cls.HIGH_THRESHOLD <= cls.MEDIUM_THRESHOLD:
            errors.append("HIGH_THRESHOLD must be greater than MEDIUM_THRESHOLD")
        
        return errors

# Validate configuration on import
config_errors = Config.validate_config()
if config_errors:
    print("❌ Configuration Errors:")
    for error in config_errors:
        print(f"   - {error}")
    raise ValueError("Invalid configuration")
