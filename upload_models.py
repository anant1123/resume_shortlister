"""
Upload models to HuggingFace Hub.
Run this once after training is complete.

Usage:
    python upload_models.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN  = os.getenv("HF_TOKEN")
HF_REPO   = os.getenv("HF_REPO", "your-hf-username/resume-shortlister")

# ── Absolute path to models folder
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ── Files to upload
MODEL_FILES = [
    MODELS_DIR / "final_model.pkl",
    MODELS_DIR / "tfidf_vectorizer.pkl",
    MODELS_DIR / "embedding_model.pkl",
    MODELS_DIR / "shap_explainer.pkl",
    MODELS_DIR / "feature_cols.json",
    MODELS_DIR / "best_params.json",
]

DATA_FILES = [
    BASE_DIR / "data" / "processed" / "cleaned_resumes.csv",
]

def upload_models():
    print(f"Logging in to HuggingFace...")
    login(token=HF_TOKEN)

    api = HfApi()

    # Create repo if not exists
    api.create_repo(
        repo_id   = HF_REPO,
        exist_ok  = True,
        repo_type = "model"
    )
    print(f"✅ Repo ready: {HF_REPO}")

    # Upload model files
    for filepath in MODEL_FILES:
        if not filepath.exists():
            print(f"⚠️  Skipping {filepath.name} — file not found")
            continue
        print(f"Uploading {filepath.name}...")
        api.upload_file(
            path_or_fileobj = str(filepath),
            path_in_repo    = f"models/{filepath.name}",
            repo_id         = HF_REPO,
            token           = HF_TOKEN
        )
        print(f"✅ Uploaded: {filepath.name}")

    # Upload data files
    for filepath in DATA_FILES:
        if not filepath.exists():
            print(f"⚠️  Skipping {filepath.name} — file not found")
            continue
        print(f"Uploading {filepath.name}...")
        api.upload_file(
            path_or_fileobj = str(filepath),
            path_in_repo    = f"data/{filepath.name}",
            repo_id         = HF_REPO,
            token           = HF_TOKEN
        )
        print(f"✅ Uploaded: {filepath.name}")

    print(f"\n🎉 All models uploaded!")
    print(f"   View at: https://huggingface.co/{HF_REPO}")

if __name__ == "__main__":
    upload_models()
