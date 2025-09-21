# 📄 Resume Relevance Check System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff69b4.svg)](https://streamlit.io/)

An **AI-powered automated resume evaluation system** that analyzes candidate resumes against job descriptions using advanced machine learning techniques.  
Built with **Streamlit** for the web interface and powered by **Google Gemini AI** for intelligent analysis.

---

## 🚀 Features

- **AI-Powered Analysis** – Gemini integration for intelligent JD parsing & feedback  
- **Multi-Format Support** – PDF, DOCX, and TXT for both resumes & JDs  
- **Hybrid Scoring** – Combines hard skill matching (40%) + semantic similarity (60%)  
- **Interactive Dashboard** – Five specialized tabs for detailed evaluation  
- **Evidence Explorer** – Sentence-level JD ↔ Resume matching  
- **Performance Calibration** – Threshold tuning & validation tools  
- **Database Persistence** – SQLite storage for history & analytics  

---

## 📋 Requirements

### System
- Python **3.8+**
- RAM: **4GB+** (8GB recommended)
- Storage: **1GB free space**
- Internet: Required for Gemini API & model downloads  

### Dependencies
```text
streamlit>=1.28.0
PyMuPDF>=1.23.0
python-docx>=0.8.11
google-generativeai>=0.6.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.2.2
🛠️ Installation
1. Clone Repository
bash
Copy code
git clone https://github.com/your-username/resume-relevance-system.git
cd resume-relevance-system
2. Create Virtual Environment
bash
Copy code
python -m venv venv

# Activate
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Configure Environment
Create a .env file in the project root:

env
Copy code
GEMINI_API_KEY=your_gemini_api_key_here
MAX_FILE_SIZE_MB=10
CACHE_TTL_SECONDS=3600
Works without API keys (fallback methods), but Gemini enhances accuracy.

5. Directory Setup
System auto-creates:

kotlin
Copy code
uploads/   → Temporary file storage
data/      → SQLite DB & processed data
🚀 Quick Start
Run the app:

bash
Copy code
streamlit run app.py
Open browser → http://localhost:8501

Workflow
Upload Job Description (TXT, PDF, DOCX or paste text)

Upload Resumes (up to 20, PDF/DOCX)

Run evaluation → View results in interactive tabs

📊 Dashboard
Tab	Purpose	Features
📋 Results	Candidate rankings & summary	Interactive grid, sorting, quick insights
📈 Detailed	Individual analysis	Score breakdowns, gaps, AI feedback
🗺️ Heatmap	Visual score comparison	JD×Resume heat visualization
🔎 Evidence	Sentence-level matching	Semantic similarity explorer
📐 Calibration	Performance tuning	Thresholds, confusion matrix

⚖️ Scoring
Hard Match (40%) → TF-IDF + BM25 exact skills

Semantic Match (60%) → Sentence Transformers

Final Verdict

High: ≥ 75

Medium: 50–74

Low: < 50

python
Copy code
HARD_MATCH_WEIGHT = 0.4
SEMANTIC_MATCH_WEIGHT = 0.6
HIGH_THRESHOLD = 75
MEDIUM_THRESHOLD = 50
MAX_RESUMES_PER_BATCH = 20
MAX_FILE_SIZE_MB = 10
🏗️ Project Structure
bash
Copy code
resume-relevance-system/
├── app.py             # Main Streamlit app
├── resume_parser.py   # Resume parsing
├── job_matcher.py     # Matching engine
├── config.py          # Config settings
├── db.py              # Database connection
├── models.py          # SQLAlchemy schemas
├── requirements.txt   # Dependencies
├── .env               # Environment variables
├── uploads/           # Temporary files
├── data/              # SQLite DB
└── README.md
🔧 Advanced Features
Skill Database – Programming, web, data science, cloud, DBs

Optimization – Caching, lazy loading, batch processing

Error Handling – Works offline, robust parsing, clear messages

🎯 Use Cases
HR: Automated screening

Agencies: Bulk evaluation

Job Portals: JD–Resume matching

Career Services: Resume feedback

Research: HR analytics & bias detection

🔍 Troubleshooting
ModuleNotFoundError →

bash
Copy code
pip install -r requirements.txt
Slow performance → Keep files <10MB, smaller batches

DB errors →

bash
Copy code
chmod 755 data/
Tips

Use PDFs <5MB

Process 5–10 resumes per batch

Add Gemini API key for best accuracy

🤝 Contributing
Fork repo

Create branch → git checkout -b feature/XYZ

Commit → git commit -m 'Add XYZ'

Push → git push origin feature/XYZ

Open PR

📄 License
Licensed under MIT. See LICENSE.

🙏 Acknowledgments
Innomatics Research Labs – Research & development

Google Gemini – AI-powered analysis

Streamlit Community – Web framework

Hugging Face – Sentence transformers

scikit-learn – ML algorithms

📞 Support
Open an issue on GitHub

Contact: [Your Info]

Docs: [Link to detailed docs]
