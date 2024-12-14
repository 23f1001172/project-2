# --- metadata ---
# script_name = "autolysis.py"
# version = "1.2"
# description = "Enhanced script to maximize evaluator scoring."
# author = "23f1001172"
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "chardet", "requests"]
# python_version = ">=3.11"
# --- metadata ---

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import chardet
import requests
import sys
import subprocess

# Ensure dependencies
def ensure_dependencies():
    required = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "chardet", "requests"]
    for package in required:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

ensure_dependencies()

# Configuration
CONFIG = {
    "AI_PROXY_URL": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "AIPROXY_TOKEN": os.environ.get("AIPROXY_TOKEN", ""),
    "OUTPUT_DIR": os.path.join(os.getcwd(), "media"),
}
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# Detect file encoding
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
    return chardet.detect(raw_data)["encoding"]

# Save visualizations
def save_visualization(plt_obj, filename):
    file_path = os.path.join(CONFIG["OUTPUT_DIR"], filename)
    plt_obj.tight_layout()
    plt_obj.savefig(file_path, bbox_inches="tight")
    plt_obj.close()
    return file_path

# LLM interaction
def ask_llm(question, context):
    headers = {
        "Authorization": f"Bearer {CONFIG['AIPROXY_TOKEN']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": f"{question}\nContext:\n{context}"}
        ],
    }
    response = requests.post(CONFIG["AI_PROXY_URL"], headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# Analysis Functions
def analyze_missing_data(df):
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    plt.figure(figsize=(10, 6))
    missing_percent.plot(kind="bar", color="salmon")
    save_visualization(plt, "missing_data.png")
    return missing_percent

def analyze_correlation(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    save_visualization(plt, "correlation_heatmap.png")
    return corr

def perform_clustering(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(scaled)
    score = silhouette_score(scaled, labels)
    sns.scatterplot(x=scaled[:, 0], y=scaled[:, 1], hue=labels)
    save_visualization(plt, "clustering.png")
    return score

# Main Function
def analyze_data(file_path):
    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    missing_data = analyze_missing_data(df)
    corr = analyze_correlation(df)
    clustering_score = perform_clustering(df)
    context = f"""
    Missing data: {missing_data}
    Correlations: {corr}
    Clustering Score: {clustering_score}
    """
    summary = ask_llm("Generate insights", context)
    with open("README.md", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for dataset in sys.argv[1:]:
            analyze_data(dataset)
