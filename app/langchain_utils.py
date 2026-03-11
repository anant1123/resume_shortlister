
import os
import json
import joblib
import numpy as np
import pandas as pd
from groq import Groq

FEATURE_LABELS = {
    "reason_resume_sim"    : "semantic match with job requirements",
    "embedding_cosine_sim" : "overall semantic similarity",
    "tfidf_cosine_sim"     : "keyword match score",
    "skill_overlap_count"  : "number of matching skills",
    "skill_overlap_ratio"  : "percentage of required skills matched",
    "resume_skill_count"   : "total skills in resume",
    "jd_skill_count"       : "total skills required by JD",
    "skill_gap"            : "number of missing required skills",
    "resume_length"        : "resume detail level",
    "jd_length"            : "job description length",
    "length_ratio"         : "resume to JD length ratio",
    "role_in_resume"       : "job role mentioned in resume",
    "seniority_match"      : "seniority level match",
    "skill_diversity"      : "breadth of skill categories"
}

def get_shap_summary(candidate_idx, X_test, shap_vals,
                      feature_cols, model, top_n=5):
    prob     = model.predict_proba(X_test)[candidate_idx][1]
    pred     = "SELECT" if prob >= 0.5 else "REJECT"
    shap_row = shap_vals[candidate_idx]
    features = list(zip(feature_cols,
                        X_test.iloc[candidate_idx].values,
                        shap_row))
    positives = sorted(
        [(f, v, s) for f, v, s in features if s > 0],
        key=lambda x: x[2], reverse=True
    )[:top_n]
    negatives = sorted(
        [(f, v, s) for f, v, s in features if s < 0],
        key=lambda x: x[2]
    )[:top_n]
    return {
        "probability" : round(float(prob), 4),
        "prediction"  : pred,
        "score_pct"   : round(float(prob) * 100, 1),
        "positives"   : positives,
        "negatives"   : negatives
    }

def build_prompt(candidate_idx, rank, total_candidates,
                 X_test, shap_vals, feature_cols,
                 model, feature_labels):
    summary = get_shap_summary(
        candidate_idx, X_test, shap_vals,
        feature_cols, model
    )
    strengths  = [f"- {feature_labels.get(f, f)} (impact: +{s:.2f})"
                  for f, v, s in summary["positives"]]
    weaknesses = [f"- {feature_labels.get(f, f)} (impact: {s:.2f})"
                  for f, v, s in summary["negatives"]]
    prompt = f"""You are an expert HR recruiter assistant.
A resume screening ML model has evaluated a candidate.
Generate a concise 2-3 sentence recruiter summary.

CANDIDATE DETAILS:
- Rank        : #{rank} out of {total_candidates} candidates
- Match Score : {summary["score_pct"]}%
- Decision    : {summary["prediction"]}

TOP STRENGTHS:
{chr(10).join(strengths) if strengths else "None"}

TOP WEAKNESSES:
{chr(10).join(weaknesses) if weaknesses else "None"}

INSTRUCTIONS:
- Write exactly 2-3 sentences
- Be specific about strengths and weaknesses  
- Use professional recruiter language
- Do NOT mention SHAP or ML model
- Do NOT use bullet points
- Start with "Candidate ranked #{{rank}}"
"""
    return prompt, summary

def generate_summary(prompt, client, model="llama3-8b-8192"):
    try:
        response = client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": "You are a professional HR recruiter assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature = 0.3,
            max_tokens  = 150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {str(e)}"
