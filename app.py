# app.py

import os
import json
import re
from datetime import datetime

import streamlit as st
import pandas as pd

from resume_parser import ResumeParser
from job_matcher import JobMatcher
from config import Config

from db import engine, SessionLocal
from models import Base, Evaluation

# UI components
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_echarts import st_echarts

# -----------------------------
# Setup & cached resources
# -----------------------------

@st.cache_resource
def load_explorer_model():
    """Sentence-level evidence model; lazy and optional."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

@st.cache_resource
def init_components():
    return ResumeParser(), JobMatcher()

# Folders & DB
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.DATA_FOLDER, exist_ok=True)
Base.metadata.create_all(bind=engine)

# Page config
st.set_page_config(
    page_title="Resume Relevance Check System",
    page_icon="📋",
    layout="wide"
)

parser, matcher = init_components()

# -----------------------------
# Title
# -----------------------------
st.title("🎯 Automated Resume Relevance Check System")
st.markdown("**Innomatics Research Labs - AI-Powered Resume Evaluation**")

# Keep evaluation results in session so tabs persist after button click
if "results" not in st.session_state:
    st.session_state["results"] = []
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "jd_name" not in st.session_state:
    st.session_state["jd_name"] = "JD_1"

# -----------------------------
# Sidebar: Uploads
# -----------------------------
with st.sidebar:
    st.header("📁 Upload Files")

    st.subheader("Job Description")
    jd_input_method = st.radio("Choose input method:", ["Upload JD File", "Paste JD Text"])
    jd_text = ""

    if jd_input_method == "Upload JD File":
        jd_file = st.file_uploader("Upload Job Description", type=['txt', 'pdf', 'docx'], key="jd_uploader")
        if jd_file:
            if jd_file.type == "text/plain":
                jd_text = jd_file.read().decode("utf-8", errors="ignore")
            else:
                temp_jd = os.path.join(Config.UPLOAD_FOLDER, f"jd_{jd_file.name}")
                with open(temp_jd, "wb") as f:
                    f.write(jd_file.getbuffer())
                jd_parsed = parser.parse_resume(temp_jd)
                jd_text = jd_parsed.get("raw_text", "")
                os.remove(temp_jd)
            st.session_state["jd_text"] = jd_text
            st.session_state["jd_name"] = os.path.splitext(jd_file.name)[0]
    else:
        jd_text_input = st.text_area("Paste Job Description", height=220, key="jd_text_area")
        if jd_text_input:
            st.session_state["jd_text"] = jd_text_input
            st.session_state["jd_name"] = "JD_1"

    st.subheader("Resumes")
    resumes = st.file_uploader("Upload Resumes", type=['pdf', 'docx'], accept_multiple_files=True, key="res_uploader")

    run_eval = st.button("Run Evaluation")

# -----------------------------
# Evaluation
# -----------------------------
if run_eval:
    jd_text = (st.session_state.get("jd_text") or "").strip()
    if not jd_text:
        st.error("Please provide a Job Description (upload or paste).")
    elif not resumes:
        st.error("Please upload at least one resume.")
    else:
        eval_results = []
        with st.spinner("Evaluating resumes..."):
            for uploaded in resumes:
                # Save temp
                temp_file = os.path.join(Config.UPLOAD_FOLDER, f"{datetime.utcnow().timestamp()}_{uploaded.name}")
                with open(temp_file, "wb") as f:
                    f.write(uploaded.getbuffer())

                # Parse resume (returns raw_text and structured fields)
                res_parsed = parser.parse_resume(temp_file)
                resume_text = res_parsed.get("raw_text", "")

                # Evaluate
                evaluation = matcher.evaluate_resume(res_parsed, jd_text)

                # Persist to DB
                try:
                    with SessionLocal() as db:
                        ev = Evaluation(
                            job_title=evaluation["job_details"].get("job_title",""),
                            filename=uploaded.name,
                            overall_score=evaluation["overall_score"],
                            verdict=evaluation["verdict"],
                            hard_score=evaluation["hard_match_score"],
                            semantic_score=evaluation["semantic_score"],
                            required_matches=json.dumps(evaluation["required_matches"]),
                            preferred_matches=json.dumps(evaluation["preferred_matches"]),
                            missing_required=json.dumps(evaluation["missing_required"]),
                            missing_preferred=json.dumps(evaluation["missing_preferred"]),
                            feedback=evaluation["feedback"],
                        )
                        db.add(ev)
                        db.commit()
                except Exception as e:
                    st.warning(f"DB write failed: {e}")

                # Cleanup temp
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

                # Store for UI/tabs
                eval_results.append({
                    "filename": uploaded.name,
                    "score": evaluation["overall_score"],
                    "verdict": evaluation["verdict"],
                    "hard_match": evaluation["hard_match_score"],
                    "semantic_match": evaluation["semantic_score"],
                    "required_matches": evaluation["required_matches"],
                    "missing_required": evaluation["missing_required"],
                    "feedback": evaluation["feedback"],
                    "full_evaluation": evaluation,
                    "resume_text": resume_text,
                    "jd_text": jd_text,
                })

        if eval_results:
            st.session_state["results"] = eval_results
            st.success("Evaluation complete.")

results = st.session_state["results"]
jd_text = st.session_state["jd_text"]
jd_name = st.session_state["jd_name"]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Results", "📈 Detailed", "🗺️ Heatmap", "🔎 Evidence", "📐 Calibration"])

# -----------------------------
# Tab 1: Results (AgGrid + quick charts)
# -----------------------------
with tab1:
    st.subheader("📋 Candidate Rankings")
    if results:
        df_results = pd.DataFrame([
            {
                "Filename": r["filename"],
                "Overall Score": r["score"],
                "Verdict": r["verdict"],
                "Hard": r["hard_match"],
                "Semantic": r["semantic_match"],
                "Required Matches": ", ".join(r["required_matches"]),
                "Missing Required (top)": ", ".join(r["missing_required"][:8]),
            } for r in sorted(results, key=lambda x: x["score"], reverse=True)
        ])

        gb = GridOptionsBuilder.from_dataframe(df_results)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_selection('single')
        grid_options = gb.build()

        grid = AgGrid(df_results, gridOptions=grid_options, enable_enterprise_modules=True, theme="alpine")
        sel = grid.get("selected_rows", [])
        if sel:
            st.info(f"Selected: {sel[0].get('Filename')}")

        st.subheader("📊 Quick Insights")
        st.bar_chart(df_results["Overall Score"])
    else:
        st.info("Upload a JD and resumes, then click Run Evaluation.")

# -----------------------------
# Tab 2: Detailed (expanders)
# -----------------------------
with tab2:
    st.subheader("📈 Detailed Analysis")
    if results:
        min_score = st.slider("Min Score", 0, 100, 0, key="min_score_detailed")
        bucket = st.selectbox("Verdict Filter", ["All", "High", "Medium", "Low"], key="bucket_detailed")
        filtered = [r for r in results if r["score"] >= min_score and (bucket == "All" or r["verdict"] == bucket)]

        top_n = st.number_input("Show top N resumes", min_value=1, max_value=min(10, len(filtered)) if filtered else 1, value=min(5, len(filtered)) if filtered else 1)
        for r in sorted(filtered, key=lambda x: x["score"], reverse=True)[:int(top_n)]:
            with st.expander(f"{r['filename']} • {r['score']} • {r['verdict']}"):
                st.markdown(f"- Hard: {r['hard_match']}")
                st.markdown(f"- Semantic: {r['semantic_match']}")
                st.markdown(f"- Required Matches: {', '.join(r['required_matches'])}")
                st.markdown(f"- Missing Required: {', '.join(r['missing_required'][:8])}")
                st.markdown("**Feedback**")
                st.write(r["feedback"])
    else:
        st.info("No results to display.")

# -----------------------------
# Tab 3: Heatmap (JD × Resume)
# -----------------------------
with tab3:
    st.subheader("🗺️ JD × Resume Heatmap")
    if results:
        jd_names = [jd_name]  # single JD for now; extend to multi-JD if needed
        resume_names = [r["filename"] for r in results]
        scores = [[r["score"]] for r in results]  # rows=resumes, cols=JDs

        options = {
            "tooltip": {"position": "top"},
            "grid": {"height": "60%", "top": "10%"},
            "xAxis": {"type": "category", "data": jd_names},
            "yAxis": {"type": "category", "data": resume_names},
            "visualMap": {"min": 0, "max": 100, "calculable": True, "orient": "horizontal", "left": "center"},
            "series": [{
                "name": "Relevance",
                "type": "heatmap",
                "data": [[j, i, scores[i][j]] for i in range(len(resume_names)) for j in range(len(jd_names))],
                "label": {"show": True},
                "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}}
            }]
        }
        clicked = st_echarts(options=options, events={"click": "function(params){ return params; }"}, height="520px")
        if clicked:
            row_idx, col_idx = clicked["data"][1], clicked["data"][0]
            st.session_state["selected_pair"] = (resume_names[row_idx], jd_names[col_idx])
            st.success(f"Selected pair → Resume: {resume_names[row_idx]} | JD: {jd_names[col_idx]}")
    else:
        st.info("No results to visualize.")

# -----------------------------
# Tab 4: Evidence Explorer
# -----------------------------
def evidence_explorer(jd_text_local: str, resume_text_local: str, topk: int = 3):
    model = load_explorer_model()
    jd_sents = [s.strip() for s in re.split(r"[.\n]", jd_text_local) if len(s.strip()) > 0]
    res_sents = [s.strip() for s in re.split(r"[.\n]", resume_text_local) if len(s.strip()) > 0]
    if not jd_sents or not res_sents:
        return []

    if model is not None:
        try:
            from sentence_transformers import util
            emb_jd = model.encode(jd_sents, convert_to_tensor=True, normalize_embeddings=True)
            emb_res = model.encode(res_sents, convert_to_tensor=True, normalize_embeddings=True)
            pairs = []
            for i, jd_s in enumerate(jd_sents):
                hits = util.semantic_search(emb_jd[i:i+1], emb_res, top_k=topk)[0]
                for h in hits:
                    pairs.append({"jd": jd_s, "resume": res_sents[h["corpus_id"]], "score": float(h["score"])})
            pairs.sort(key=lambda x: -x["score"])
            return pairs[: min(len(pairs), 10)]
        except Exception:
            pass

    # TF-IDF fallback
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as _Tvec
        from sklearn.metrics.pairwise import cosine_similarity as _cos
        pairs = []
        for jd_s in jd_sents:
            X = _Tvec(stop_words="english").fit_transform([jd_s] + res_sents)
            sims = _cos(X[0:1], X[1:]).flatten()
            top_idx = sims.argsort()[-topk:][::-1]
            for idx in top_idx:
                pairs.append({"jd": jd_s, "resume": res_sents[idx], "score": float(sims[idx])})
        pairs.sort(key=lambda x: -x["score"])
        return pairs[: min(len(pairs), 10)]
    except Exception:
        return []

with tab4:
    st.subheader("🔎 Evidence Explorer")
    if results and jd_text:
        filenames = [r["filename"] for r in results]
        default_file = st.session_state.get("selected_pair", (filenames[0] if filenames else None, jd_name))[0] if filenames else None
        candidate = st.selectbox("Select candidate", filenames, index=filenames.index(default_file) if default_file in filenames else 0)
        if candidate:
            sel = next(r for r in results if r["filename"] == candidate)
            pairs = evidence_explorer(sel["jd_text"], sel["resume_text"])
            if not pairs:
                st.warning("No evidence available. Ensure resume text is captured during parsing.")
            else:
                for p in pairs:
                    st.markdown(f"- JD: {p['jd']}\n  \n  Match: {p['resume']}\n  \n  Score: {round(p['score'], 3)}")
    else:
        st.info("No evaluated results for evidence.")

# -----------------------------
# Tab 5: Calibration
# -----------------------------
with tab5:
    st.subheader("📐 Calibration & Thresholds")
    st.markdown("Upload a CSV with columns: overall_score (0-100), true_bucket (High/Medium/Low).")
    file_cal = st.file_uploader("Upload labeled CSV", type=["csv"], key="cal_csv")
    if file_cal:
        df_lab = pd.read_csv(file_cal)
        if {"overall_score", "true_bucket"}.issubset(df_lab.columns):
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            pred = pd.cut(df_lab["overall_score"], bins=[-0.1, 50, 75, 100], labels=["Low","Medium","High"]).astype(str)
            cm = confusion_matrix(df_lab["true_bucket"], pred, labels=["High","Medium","Low"])
            fig, ax = plt.subplots(figsize=(4,4))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["High","Medium","Low"]).plot(ax=ax, colorbar=False)
            st.pyplot(fig)
            st.write("Counts:", cm.tolist())
        else:
            st.error("CSV must contain overall_score and true_bucket columns.")
