import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import re
from groq import Groq
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load .env file
load_dotenv()

# ── HuggingFace config
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO  = os.getenv("HF_REPO", "your-hf-username/resume-shortlister")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Resume Shortlister",
    page_icon  = "🎯",
    layout     = "wide"
)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
MODELS_DIR = "../models"
DATA_DIR   = "../data"

SKILLS = [
    "python","java","c++","c","javascript","typescript","go","rust",
    "scala","r","matlab","kotlin","swift","php","bash","powershell",
    "dart","perl","machine learning","deep learning","data science",
    "predictive modeling","statistical modeling","time series","nlp",
    "computer vision","reinforcement learning","feature engineering",
    "model deployment","model evaluation","data mining",
    "bayesian statistics","a/b testing","scikit-learn","tensorflow",
    "pytorch","keras","xgboost","lightgbm","catboost","huggingface",
    "transformers","spacy","nltk","gensim","opencv","fastai","mlflow",
    "dvc","shap","lime","optuna","smote","mlops","pandas","numpy",
    "scipy","statsmodels","matplotlib","seaborn","plotly","bokeh",
    "tableau","power bi","excel","looker","data visualization",
    "hadoop","spark","pyspark","hive","kafka","flink","airflow","dbt",
    "databricks","delta lake","sql","mysql","postgresql","sqlite",
    "mongodb","cassandra","redis","dynamodb","elasticsearch","snowflake",
    "bigquery","redshift","etl","elt","data pipeline","data warehouse",
    "data lake","docker","kubernetes","jenkins","terraform","ansible",
    "aws","gcp","azure","aws s3","aws ec2","aws lambda","html","css",
    "react","node.js","django","flask","fastapi","rest api","langchain",
    "rag","llm","openai api","faiss","pinecone","bert","gpt","llama",
    "streamlit","gradio","linux","git","github","selenium",
    "beautifulsoup","prompt engineering","vector database",
    "generative ai","fine tuning","embeddings"
]

FEATURE_LABELS = {
    'reason_resume_sim'    : 'semantic match with job requirements',
    'embedding_cosine_sim' : 'overall semantic similarity',
    'tfidf_cosine_sim'     : 'keyword match score',
    'skill_overlap_count'  : 'number of matching skills',
    'skill_overlap_ratio'  : 'percentage of required skills matched',
    'resume_skill_count'   : 'total skills in resume',
    'jd_skill_count'       : 'total skills required by JD',
    'skill_gap'            : 'number of missing required skills',
    'resume_length'        : 'resume detail level',
    'jd_length'            : 'job description length',
    'length_ratio'         : 'resume to JD length ratio',
    'role_in_resume'       : 'job role mentioned in resume',
    'seniority_match'      : 'seniority level match',
    'skill_diversity'      : 'breadth of skill categories'
}

ROLE_SKILLS_MAP = {
    'Machine Learning Engineer': [
        'python','machine learning','deep learning','scikit-learn',
        'tensorflow','pytorch','xgboost','feature engineering',
        'mlflow','shap','pandas','numpy','optuna','model deployment','mlops'
    ],
    'Data Scientist': [
        'python','machine learning','statistics','sql','pandas','numpy',
        'tensorflow','scikit-learn','data visualization','tableau',
        'deep learning','a/b testing','statistical modeling'
    ],
    'Data Engineer': [
        'python','sql','spark','kafka','airflow','etl','data pipeline',
        'bigquery','snowflake','postgresql','aws','docker','dbt',
        'pyspark','data warehouse','data lake'
    ],
    'Data Analyst': [
        'sql','python','tableau','power bi','excel','pandas','statistics',
        'data visualization','a/b testing','matplotlib','seaborn','looker'
    ],
    'AI Engineer': [
        'python','langchain','rag','llm','openai api','faiss',
        'prompt engineering','vector database','generative ai',
        'bert','fine tuning','embeddings','streamlit','fastapi'
    ],
    'Software Engineer': [
        'python','java','git','docker','rest api','sql','linux','aws',
        'kubernetes','microservices','system design','agile'
    ],
    'DevOps Engineer': [
        'docker','kubernetes','aws','jenkins','terraform','linux',
        'ansible','prometheus','grafana','github actions','bash','ci/cd'
    ],
    'Full Stack Developer': [
        'javascript','typescript','react','node.js','python','sql',
        'docker','git','rest api','html','css','mongodb'
    ],
}

senior_keywords = ['senior','lead','principal','head',
                   'architect','manager','director']
junior_keywords = ['junior','entry','fresher','intern',
                   'trainee','associate','graduate']

# ─────────────────────────────────────────
# LOAD MODELS — cached
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    """
    Load models from HuggingFace Hub if HF_TOKEN is set,
    otherwise fall back to local ../models/ directory.
    """
    use_hf = HF_TOKEN and HF_REPO

    def load_pkl(filename):
        if use_hf:
            path = hf_hub_download(
                repo_id  = HF_REPO,
                filename = f"models/{filename}",
                token    = HF_TOKEN
            )
        else:
            path = f"{MODELS_DIR}/{filename}"
        return joblib.load(path)

    def load_json(filename):
        if use_hf:
            path = hf_hub_download(
                repo_id  = HF_REPO,
                filename = f"models/{filename}",
                token    = HF_TOKEN
            )
        else:
            path = f"{MODELS_DIR}/{filename}"
        with open(path) as f:
            return json.load(f)

    model           = load_pkl("final_model.pkl")
    tfidf           = load_pkl("tfidf_vectorizer.pkl")
    embedding_model = load_pkl("embedding_model.pkl")
    explainer       = load_pkl("shap_explainer.pkl")
    feature_cols    = load_json("feature_cols.json")

    source = "HuggingFace Hub" if use_hf else "local ../models/"
    print(f"✅ Models loaded from: {source}")

    return model, tfidf, embedding_model, explainer, feature_cols

# ── GitHub raw URL for JDs dataset
GITHUB_RAW = "https://raw.githubusercontent.com/anant1123/resume_shortlister/master/data/processed/cleaned_resumes.csv"

@st.cache_data
def load_naukri_jds():
    try:
        # Load from GitHub raw URL (works both local and deployed)
        return pd.read_csv(GITHUB_RAW)
    except:
        try:
            # Fall back to local
            return pd.read_csv(f"{DATA_DIR}/processed/cleaned_resumes.csv")
        except:
            return pd.DataFrame()

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    doc  = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\-\+\#]', ' ', text)
    return text.strip()

def extract_skills(text):
    text_lower = text.lower()
    return list(set([s for s in SKILLS if s in text_lower]))

def get_seniority(text):
    text_lower = text.lower()
    if any(k in text_lower for k in senior_keywords):
        return 2
    elif any(k in text_lower for k in junior_keywords):
        return 0
    return 1

def skill_diversity(skills):
    categories = {
        'ml'    : ['machine learning','deep learning','xgboost','scikit-learn'],
        'cloud' : ['aws','azure','gcp','docker','kubernetes'],
        'data'  : ['sql','pandas','spark','airflow','etl'],
        'ai'    : ['langchain','llm','rag','openai api','bert'],
        'devops': ['git','ci/cd','jenkins','terraform','linux']
    }
    return sum(1 for cat_skills in categories.values()
               if any(s in skills for s in cat_skills))

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_features(resume_text, jd_text, role,
                   tfidf, embedding_model, feature_cols):

    jd_combined   = role + " " + jd_text
    resume_skills = extract_skills(resume_text)
    jd_skills     = list(set(
        extract_skills(jd_text) + ROLE_SKILLS_MAP.get(role, [])
    ))

    # TF-IDF
    tfidf_resume = tfidf.transform([resume_text])
    tfidf_jd     = tfidf.transform([jd_combined])
    tfidf_sim    = cosine_similarity(tfidf_resume, tfidf_jd)[0][0]

    # Embeddings
    emb_resume = embedding_model.encode([resume_text])
    emb_jd     = embedding_model.encode([jd_combined])
    emb_sim    = cosine_similarity(emb_resume, emb_jd)[0][0]

    # Reason sim — use JD embedding as proxy
    reason_sim = float(emb_sim) * 0.9

    # Skill features
    overlap       = set(resume_skills) & set(jd_skills)
    overlap_count = len(overlap)
    overlap_ratio = overlap_count / max(len(jd_skills), 1)
    skill_gap     = len(set(jd_skills) - set(resume_skills))

    # Length features
    resume_len = len(resume_text.split())
    jd_len     = len(jd_combined.split())
    length_rat = resume_len / (jd_len + 1)

    # Other features
    role_in_res   = 1 if role.lower() in resume_text.lower() else 0
    sen_resume    = get_seniority(resume_text)
    sen_jd        = get_seniority(jd_combined)
    sen_match     = 1 if sen_resume == sen_jd else 0
    skill_div     = skill_diversity(resume_skills)

    features = {
        'tfidf_cosine_sim'     : round(float(tfidf_sim), 6),
        'embedding_cosine_sim' : round(float(emb_sim), 6),
        'reason_resume_sim'    : round(reason_sim, 6),
        'skill_overlap_count'  : overlap_count,
        'skill_overlap_ratio'  : round(overlap_ratio, 6),
        'resume_skill_count'   : len(resume_skills),
        'jd_skill_count'       : len(jd_skills),
        'skill_gap'            : skill_gap,
        'resume_length'        : resume_len,
        'jd_length'            : jd_len,
        'length_ratio'         : round(length_rat, 6),
        'role_in_resume'       : role_in_res,
        'seniority_match'      : sen_match,
        'skill_diversity'      : skill_div
    }

    return pd.DataFrame([features])[feature_cols], resume_skills, jd_skills

# ─────────────────────────────────────────
# SHAP + LANGCHAIN
# ─────────────────────────────────────────
def get_shap_values(model, explainer, X_row):
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list):
        return shap_vals[1][0]
    return shap_vals[0]

def build_prompt(rank, total, score_pct, prediction,
                 shap_row, feature_cols, feature_labels):
    features  = list(zip(feature_cols, shap_row))
    positives = sorted([(f, s) for f, s in features if s > 0],
                       key=lambda x: x[1], reverse=True)[:4]
    negatives = sorted([(f, s) for f, s in features if s < 0],
                       key=lambda x: x[1])[:4]

    strengths  = "\n".join([f"- {feature_labels.get(f,f)} (+{s:.2f})"
                            for f, s in positives]) or "None"
    weaknesses = "\n".join([f"- {feature_labels.get(f,f)} ({s:.2f})"
                            for f, s in negatives]) or "None"

    return f"""You are an expert HR recruiter assistant.
A resume screening ML model evaluated a candidate.
Write a concise 2-3 sentence recruiter summary.

CANDIDATE:
- Rank        : #{rank} out of {total}
- Match Score : {score_pct}%
- Decision    : {prediction}

STRENGTHS:
{strengths}

WEAKNESSES:
{weaknesses}

INSTRUCTIONS:
- Exactly 2-3 sentences
- Professional recruiter language
- Do NOT mention SHAP or ML
- No bullet points
- Start with "Candidate ranked #{rank}"
"""

def generate_summary(prompt, client):
    try:
        resp = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [
                {"role": "system", "content": "You are a professional HR recruiter assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature = 0.3,
            max_tokens  = 150
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {str(e)}"

# ─────────────────────────────────────────
# SHAP BAR CHART
# ─────────────────────────────────────────
def plot_shap_bar(shap_row, feature_cols, feature_labels):
    labels = [feature_labels.get(f, f) for f in feature_cols]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in shap_row]

    sorted_pairs = sorted(zip(labels, shap_row),
                          key=lambda x: abs(x[1]), reverse=True)[:8]
    sorted_labels = [p[0] for p in sorted_pairs]
    sorted_vals   = [p[1] for p in sorted_pairs]
    sorted_colors = ['#2ecc71' if v > 0 else '#e74c3c'
                     for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(sorted_labels[::-1], sorted_vals[::-1],
                   color=sorted_colors[::-1])
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on score)")
    ax.set_title("Feature Impact", fontweight='bold', fontsize=11)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
def main():
    # Header
    st.title("🎯 Resume Shortlisting Engine")
    st.markdown("Upload resumes and a job description to rank candidates using ML + AI explanations.")
    st.divider()

    # Load models
    with st.spinner("Loading models..."):
        model, tfidf, embedding_model, explainer, feature_cols = load_models()
    naukri_df = load_naukri_jds()

    # ── Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.divider()
        st.markdown("**How it works:**")
        st.markdown("1. Enter Job Description")
        st.markdown("2. Upload PDF resumes")
        st.markdown("3. Click Rank Candidates")
        st.markdown("4. See ranked results + AI explanations")
        st.divider()
        st.markdown("**Model:** XGBoost (AUC 0.806)")
        st.markdown("**Features:** 14 engineered features")
        st.markdown("**Explainability:** SHAP values")
        st.markdown("**AI Layer:** Groq LLaMA 3.3")

    # ── Step 1: Job Description
    st.subheader("📋 Step 1 — Job Description")

    jd_tab1, jd_tab2 = st.tabs(["📂 Select from Naukri JDs", "✏️ Type / Paste Manually"])

    jd_text  = ""
    selected_role = "Machine Learning Engineer"

    with jd_tab1:
        if not naukri_df.empty:
            role_options = naukri_df['Role'].unique().tolist() \
                if 'Role' in naukri_df.columns else []
            selected_role_naukri = st.selectbox(
                "Filter by Role", role_options
            )
            filtered = naukri_df[naukri_df['Role'] == selected_role_naukri]
            jd_options = filtered['Job_Description'].tolist() \
                if 'Job_Description' in filtered.columns else []

            if jd_options:
                selected_jd = st.selectbox(
                    "Select Job Description",
                    jd_options,
                    format_func=lambda x: x[:80] + "..."
                )
                if st.button("Use This JD"):
                    st.session_state['jd_text']  = selected_jd
                    st.session_state['jd_role']  = selected_role_naukri
                    st.success("✅ JD loaded!")
        else:
            st.info("Naukri JDs not found. Please use manual entry tab.")

    with jd_tab2:
        manual_role = st.selectbox(
            "Select Role",
            list(ROLE_SKILLS_MAP.keys())
        )
        manual_jd = st.text_area(
            "Paste Job Description here",
            height      = 200,
            placeholder = "We are looking for a Machine Learning Engineer..."
        )
        if st.button("Use This JD ✏️"):
            st.session_state['jd_text'] = manual_jd
            st.session_state['jd_role'] = manual_role
            if manual_jd.strip():
                st.success("✅ JD loaded!")
            else:
                st.error("Please enter a job description first.")

    if 'jd_text' in st.session_state and st.session_state['jd_text']:
        jd_text       = st.session_state['jd_text']
        selected_role = st.session_state['jd_role']
        with st.expander("📄 Active Job Description", expanded=False):
            st.write(f"**Role:** {selected_role}")
            st.text_area("Full Job Description", value=jd_text, height=300, disabled=True)

    st.divider()

    # ── Step 2: Upload Resumes
    st.subheader("📁 Step 2 — Upload Resumes (PDF)")

    uploaded_files = st.file_uploader(
        "Upload multiple PDF resumes",
        type            = ["pdf"],
        accept_multiple_files = True
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) uploaded")

    st.divider()

    # ── Step 3: Rank
    st.subheader("🚀 Step 3 — Rank Candidates")

    rank_btn = st.button(
        "🎯 Rank Candidates",
        type = "primary",
        use_container_width = True
    )

    if rank_btn:
        # Validations
        if not jd_text.strip():
            st.error("❌ Please provide a Job Description first.")
            return
        if not uploaded_files:
            st.error("❌ Please upload at least one resume.")
            return
        # Load Groq key from .env
        groq_key = os.getenv("GROQ_API_KEY", "")
        client   = Groq(api_key=groq_key) if groq_key else None
        if not groq_key:
            st.warning("⚠️ No Groq API key in .env — summaries will be skipped.")

        # Process all resumes
        results = []
        progress = st.progress(0, text="Processing resumes...")

        for idx, pdf_file in enumerate(uploaded_files):
            progress.progress(
                (idx + 1) / len(uploaded_files),
                text=f"Processing {pdf_file.name}..."
            )

            # Extract text
            resume_text = extract_text_from_pdf(pdf_file)
            resume_text = clean_text(resume_text)

            if len(resume_text.split()) < 50:
                st.warning(f"⚠️ {pdf_file.name} — too short, skipping.")
                continue

            # Build features
            X_row, resume_skills, jd_skills = build_features(
                resume_text, jd_text, selected_role,
                tfidf, embedding_model, feature_cols
            )

            # Predict
            score    = model.predict_proba(X_row)[0][1]
            decision = "✅ SELECT" if score >= 0.5 else "❌ REJECT"

            # SHAP
            shap_row = get_shap_values(model, explainer, X_row)

            results.append({
                'name'          : pdf_file.name.replace('.pdf', ''),
                'score'         : round(float(score), 4),
                'score_pct'     : round(float(score) * 100, 1),
                'decision'      : decision,
                'resume_skills' : resume_skills,
                'jd_skills'     : jd_skills,
                'shap_row'      : shap_row,
                'X_row'         : X_row,
                'resume_text'   : resume_text[:300]
            })

        progress.empty()

        if not results:
            st.error("No valid resumes processed.")
            return

        # Sort by score
        results = sorted(results,
                         key=lambda x: x['score'], reverse=True)

        # ── Results
        st.divider()
        st.subheader(f"🏆 Results — {len(results)} Candidates Ranked")

        # Summary table
        summary_data = {
            'Rank'     : list(range(1, len(results) + 1)),
            'Candidate': [r['name'] for r in results],
            'Score'    : [f"{r['score_pct']}%" for r in results],
            'Decision' : [r['decision'] for r in results]
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width = True,
            hide_index          = True
        )

        st.divider()

        # ── Detailed cards
        for rank, result in enumerate(results, 1):
            with st.expander(
                f"#{rank} — {result['name']} | "
                f"{result['score_pct']}% | {result['decision']}",
                expanded = rank <= 3
            ):
                col1, col2 = st.columns([1, 1])

                with col1:
                    # Score gauge
                    score_color = "#2ecc71" if result['score'] >= 0.5 \
                                  else "#e74c3c"
                    st.markdown(
                        f"<h2 style='color:{score_color};'>"
                        f"{result['score_pct']}% Match</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Decision:** {result['decision']}")
                    st.progress(result['score'])

                    # Skills
                    st.markdown("**✅ Matched Skills:**")
                    matched = list(
                        set(result['resume_skills']) &
                        set(result['jd_skills'])
                    )[:10]
                    if matched:
                        st.markdown(" ".join(
                            [f"`{s}`" for s in matched]
                        ))
                    else:
                        st.markdown("_No matched skills found_")

                    st.markdown("**❌ Missing Skills:**")
                    missing = list(
                        set(result['jd_skills']) -
                        set(result['resume_skills'])
                    )[:8]
                    if missing:
                        st.markdown(" ".join(
                            [f"`{s}`" for s in missing]
                        ))
                    else:
                        st.markdown("_No missing skills_")

                with col2:
                    # SHAP chart
                    fig = plot_shap_bar(
                        result['shap_row'],
                        feature_cols,
                        FEATURE_LABELS
                    )
                    st.pyplot(fig)
                    plt.close()

                # AI Summary
                st.markdown("---")
                st.markdown("**🤖 AI Recruiter Summary:**")

                if client:
                    with st.spinner("Generating summary..."):
                        prompt = build_prompt(
                            rank, len(results),
                            result['score_pct'],
                            result['decision'],
                            result['shap_row'],
                            feature_cols,
                            FEATURE_LABELS
                        )
                        summary = generate_summary(prompt, client)
                    st.info(summary)
                else:
                    st.warning("Add Groq API key in sidebar for AI summaries.")

if __name__ == "__main__":
    main()