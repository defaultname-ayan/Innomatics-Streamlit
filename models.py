from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from db import Base

class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, index=True)
    job_title = Column(String(256))
    filename = Column(String(512))
    overall_score = Column(Float)
    verdict = Column(String(16))
    hard_score = Column(Float)
    semantic_score = Column(Float)
    required_matches = Column(Text)   # JSON string
    preferred_matches = Column(Text)  # JSON string
    missing_required = Column(Text)   # JSON string
    missing_preferred = Column(Text)  # JSON string
    feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
