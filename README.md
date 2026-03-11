# 🎯 Resume Shortlisting Engine

> An end-to-end ML + AI system that ranks candidates against a Job Description using TF-IDF, Sentence Embeddings, XGBoost, SHAP explainability, and LangChain-powered recruiter summaries.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![AUC](https://img.shields.io/badge/AUC--ROC-0.806-brightgreen)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/AI-LangChain%20%2B%20Groq-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Problem Statement

Recruiters spend hours manually screening resumes. This system automates candidate ranking by combining classical ML with semantic AI — giving recruiters not just a score, but a plain-English explanation for every decision.

---

## 🚀 Live Demo

> 📹 **Demo Video:** _Coming soon — will be added after Streamlit Cloud deployment_

> 🌐 **Live App:** _Coming soon_

---

## 🏗️ System Architecture

```
Job Description + Resumes (PDF)
         ↓
  Feature Engineering (14 features)
  ├── TF-IDF Cosine Similarity
  ├── MiniLM Embedding Similarity
  ├── Reason-Resume Semantic Sim
  ├── Skill Overlap Count/Ratio
  ├── Seniority Match
  └── 8 more engineered features
         ↓
  XGBoost Classifier (AUC 0.806)
         ↓
  SHAP Explainability
         ↓
  LangChain + Groq → Plain English Summary
         ↓
  Streamlit App → Ranked Candidates
```


---

## ✨ Features

- 📄 **PDF Resume Parsing** — Upload multiple resumes at once using PyMuPDF
- 🔍 **Dual NLP Matching** — TF-IDF (keyword) + MiniLM embeddings (semantic)
- 🧠 **XGBoost Classifier** — Trained on 5,323 labeled resume-JD pairs
- 📊 **SHAP Explainability** — Per-candidate feature impact visualization
- 🤖 **AI Summaries** — LangChain + Groq LLaMA 3.3 generates recruiter summaries
- 🗂️ **Naukri JD Dropdown** — 352 real scraped Indian job descriptions
- ✏️ **Manual JD Entry** — Paste any custom job description
- 🏆 **Ranked Results** — Candidates ranked by match score with full breakdown

---

## 📊 Model Performance

| Model               | AUC-ROC | Avg Precision |
|---------------------|---------|---------------|
| Logistic Regression | 0.697   | 0.638         |
| Random Forest       | 0.798   | 0.781         |
| **XGBoost (Tuned)** | **0.806** | **0.795**   |

> Tuned using **Optuna** (50 trials, 5-fold Stratified CV)

---

## 🔬 Feature Engineering (14 Features)

| Feature | Description |
|---------|-------------|
| `tfidf_cosine_sim` | Keyword overlap between resume and JD |
| `embedding_cosine_sim` | Semantic similarity via MiniLM-L6-v2 |
| `reason_resume_sim` | Semantic match with hiring reason ⭐ most important |
| `skill_overlap_count` | Count of matched skills |
| `skill_overlap_ratio` | % of required JD skills found in resume |
| `skill_gap` | Number of missing required skills |
| `resume_skill_count` | Total skills detected in resume |
| `jd_skill_count` | Total skills required by JD |
| `resume_length` | Word count of resume |
| `jd_length` | Word count of JD |
| `length_ratio` | Resume to JD length ratio |
| `role_in_resume` | Whether job role is mentioned in resume |
| `seniority_match` | Junior/Mid/Senior level alignment |
| `skill_diversity` | Breadth of skill categories covered |


---

## 🗂️ Project Structure

```
Resume Shortlister/
├── app/
│   ├── app.py                    ← Streamlit application
│   └── langchain_utils.py        ← LangChain + Groq functions
├── data/
│   ├── raw/                      ← Scraped Naukri + Internshala JDs
│   └── processed/
│       ├── cleaned_resumes.csv   ← Cleaned training data
│       ├── featured_df.csv       ← Feature engineered dataset
│       ├── X_features.csv        ← Feature matrix (14 features)
│       └── y_labels.csv          ← Labels
├── models/
│   ├── final_model.pkl           ← XGBoost (best model)
│   ├── tfidf_vectorizer.pkl      ← Fitted TF-IDF
│   ├── embedding_model.pkl       ← MiniLM sentence transformer
│   ├── shap_explainer.pkl        ← SHAP TreeExplainer
│   └── feature_cols.json         ← Feature column names
├── notebooks/
│   ├── resume-cleaning.ipynb     ← Data cleaning
│   ├── resume-eda.ipynb          ← Exploratory analysis
│   ├── feature_engineering.ipynb ← Feature engineering
│   ├── model_training.ipynb      ← Model training + Optuna
│   ├── shap_explainability.ipynb ← SHAP analysis
│   ├── langchain_layer.ipynb     ← LangChain + Groq testing
│   ├── naukri_scrapping.ipynb    ← Naukri JD scraper
│   └── intershala-data-cleaning.ipynb ← Internshala scraper
├── reports/
│   ├── shap_importance.png
│   ├── shap_beeswarm.png
│   ├── model_comparison.png
│   └── roc_pr_curves.png
├── scrapper/
├── .env.example                  ← API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| Data Collection | BeautifulSoup, Selenium |
| Data Processing | Pandas, NumPy, Regex |
| NLP | TF-IDF, SentenceTransformers (MiniLM), spaCy |
| ML | XGBoost, scikit-learn, imbalanced-learn (SMOTE) |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| AI Layer | LangChain, Groq API (LLaMA 3.3) |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |


---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/anant1123/resume-shortlister.git
cd resume-shortlister
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get free key at: https://console.groq.com
```

### 5. Run the app
```bash
cd app
streamlit run app.py
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your **free** Groq API key at [console.groq.com](https://console.groq.com)

---

## 📓 Notebook Pipeline

Run notebooks in this order to reproduce the full pipeline:

```
1. notebooks/resume-cleaning.ipynb       → Clean raw dataset
2. notebooks/resume-eda.ipynb            → Exploratory analysis
3. notebooks/feature_engineering.ipynb  → Build 14 features
4. notebooks/model_training.ipynb        → Train + tune models
5. notebooks/shap_explainability.ipynb  → SHAP analysis
6. notebooks/langchain_layer.ipynb       → Test LangChain layer
```

---

## 📈 Key Findings from EDA

- Dataset: **5,323** labeled resume-JD pairs across **14 tech roles**
- Nearly balanced labels: 51% reject / 49% select
- Selected and rejected resumes share similar vocabulary — justifying **semantic embeddings** over keyword matching
- `reason_resume_sim` was the **strongest predictor** (SHAP value: 0.747)
- ML Engineer role had the **lowest acceptance rate** (44.2%)


---

## 🤖 How The App Works

1. **Select Job Description** — Choose from 352 scraped Naukri JDs or paste your own
2. **Upload Resumes** — Drag and drop multiple PDF resumes
3. **Click Rank Candidates** — Model processes all resumes in seconds
4. **View Results** — Each candidate gets:
   - Match score (0-100%)
   - SELECT / REJECT decision
   - Matched and missing skills
   - SHAP feature impact chart
   - AI-generated recruiter summary

---


## 🙋 Author

- 🔗 [LinkedIn](https://www.linkedin.com/in/anantkhandelwal3)
- 💻 [GitHub](https://github.com/anant1123)
- 📧 anantkhandelwal3@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star This Repo

If you found this project useful, please consider giving it a ⭐ — it helps others find it!

