Resume Relevance Check System
An AI-powered automated resume relevance evaluation system built with Streamlit that analyzes resumes against job descriptions and provides detailed scoring, feedback, and recommendations.

Features
Core Capabilities
AI-Powered Analysis: Uses Google Gemini AI for intelligent job description parsing and feedback generation

Multi-Format Support: Handles PDF, DOCX, and text file formats for both resumes and job descriptions

Comprehensive Scoring: Combines hard skill matching (40%) and semantic similarity (60%) for accurate relevance scores

Interactive Dashboard: Multi-tab Streamlit interface with results visualization, heatmaps, and evidence explorer

Advanced Features
Evidence Explorer: Sentence-level matching between job descriptions and resumes using semantic similarity

Skill Database: Comprehensive skill extraction covering programming languages, web technologies, data science, cloud, and specialized skills

Fallback Systems: Robust regex-based parsing when AI services are unavailable

Performance Calibration: Built-in system for threshold tuning and model validation

Architecture
Core Components
Component	Description	Key Features
Resume Parser	Extracts structured data from resumes	Multi-format support, skill detection, contact extraction 
Job Matcher	AI-powered matching engine	Gemini integration, BM25 scoring, semantic similarity 
Streamlit App	Interactive web interface	Multi-tab dashboard, real-time evaluation, data persistence 
Database Layer	SQLite-based storage	Evaluation history, performance tracking 
Scoring Algorithm
The system uses a hybrid scoring approach:

Hard Match Score (40%): Exact and fuzzy skill matching using TF-IDF and BM25

Semantic Score (60%): Context-aware similarity using sentence transformers

Verdict Classification: High (≥75), Medium (≥50), Low (<50)

Installation
Prerequisites
Python 3.8 or higher

Google Gemini API key (optional, fallback methods available)

Setup Steps
Clone and navigate to the project directory:

bash
git clone <repository-url>
cd resume-relevance-system
Install dependencies:

bash
pip install -r requirements.txt
Configure environment variables (create .env file):

text
GEMINI_API_KEY=your_gemini_api_key_here
Run the application:

bash
streamlit run app.py
Usage
Basic Workflow
Upload Job Description:

Upload JD file (TXT, PDF, DOCX) or paste text directly

System automatically extracts required skills, experience, and education requirements

Upload Resumes:

Select multiple resume files (PDF, DOCX supported)

Batch processing for up to 20 resumes simultaneously

Run Evaluation:

Click "Run Evaluation" to process all resumes

View results across multiple interactive tabs

Dashboard Tabs
Tab	Purpose	Features
Results	Overview and rankings	Sortable grid, quick charts, candidate selection 
Detailed	In-depth analysis	Score breakdowns, skill gaps, actionable feedback 
Heatmap	Visual comparison	Interactive JD×Resume score visualization 
Evidence	Sentence-level matching	Semantic similarity between JD and resume sections 
Calibration	Performance tuning	Threshold adjustment, confusion matrix analysis 
Configuration
Key Settings
Scoring Weights: Hard match (40%) vs Semantic (60%) - adjustable in config.py

Performance Thresholds: High (75+), Medium (50-74), Low (<50)

File Limits: Maximum 10MB per file, 20 resumes per batch

API Configuration
The system works with or without API keys:

With Gemini API: Enhanced JD parsing and intelligent feedback generation

Without API: Regex-based fallback with comprehensive skill detection

Dependencies
Core Requirements
text
streamlit>=1.28.0
PyMuPDF>=1.23.0
python-docx>=0.8.11
google-generativeai>=0.6.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.2
rank-bm25>=0.2.2
rapidfuzz>=3.6.1
SQLAlchemy>=2.0.29
UI Components
text
streamlit-aggrid>=1.0.5
streamlit-echarts>=0.4.0
plotly>=5.17.0
matplotlib>=3.7.0
Database Schema
The system uses SQLite with the following evaluation model :

Evaluation ID: Primary key

Job Details: Title, filename

Scores: Overall, hard match, semantic scores

Skill Analysis: Required matches, missing skills (JSON)

Feedback: AI-generated recommendations

Timestamps: Creation tracking

Technical Highlights
AI Integration
Google Gemini 1.5 Flash: For intelligent job description parsing and feedback generation

Sentence Transformers: Using all-MiniLM-L6-v2 for semantic similarity

Fallback Systems: Comprehensive regex patterns ensure functionality without AI

Performance Optimizations
Caching: LRU cache for parsed job descriptions

Lazy Loading: Models loaded on-demand to reduce startup time

Batch Processing: Efficient handling of multiple resumes

Error Handling
Graceful Degradation: System works even when AI services fail

File Format Support: Robust parsing with error recovery

Database Resilience: Proper session management and error handling

