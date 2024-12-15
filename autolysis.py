# --- metadata ---
# script_name = "autolysis.py"
# version = "2.2"
# description = "Enhanced dataset analysis with robust visualization and narrative generation."
# author = "23f1001172"
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "chardet", "requests", "pillow", "pytesseract"]
# python_version = ">=3.11"
# --- metadata ---

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pytesseract
import chardet
import requests

# Dependency installation
def ensure_dependencies():
    required = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "chardet", "requests", "pillow", "pytesseract"]
    for package in required:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

ensure_dependencies()

# Configuration
CONFIG = {
    "AI_PROXY_URL": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "AIPROXY_TOKEN": os.getenv("AIPROXY_TOKEN", ""),
    "OUTPUT_DIR": os.path.join(os.getcwd(), "media"),
}
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# Visualization Helper
def save_visualization(plt_obj, filename):
    try:
        file_path = os.path.join(CONFIG["OUTPUT_DIR"], filename)
        plt_obj.tight_layout()
        plt_obj.savefig(file_path, bbox_inches="tight")
        plt_obj.close()
        return file_path
    except Exception as e:
        print(f"Error saving visualization: {e}")
        return None

# Detect Encoding
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
    return chardet.detect(raw_data)["encoding"]

# Missing Data Analysis
def analyze_missing_data(df):
    missing_percent = (df.isnull().sum() / len(df)) * 100
    plt.figure(figsize=(10, 6))
    missing_percent.plot(kind="bar", color="salmon")
    plt.title("Missing Data Analysis", fontsize=16)
    plt.xlabel("Columns")
    plt.ylabel("Percentage of Missing Values")
    return save_visualization(plt, "missing_data.png")

# Correlation Analysis
def analyze_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=16)
    return save_visualization(plt, "correlation_heatmap.png")

# Clustering Analysis
def perform_clustering(df):
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns for clustering.")
        return None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, labels)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_data, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, palette="viridis", s=100)
    plt.title(f"KMeans Clustering (Silhouette Score: {silhouette_avg:.2f})", fontsize=16)
    return save_visualization(plt, "clustering.png")

# LLM Insights
def ask_llm(question, context):
    headers = {"Authorization": f"Bearer {CONFIG['AIPROXY_TOKEN']}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": f"{question}\nContext:\n{context}"}
        ],
    }
    try:
        response = requests.post(CONFIG["AI_PROXY_URL"], headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error interacting with LLM: {e}")
        return "LLM interaction failed."

# Generate README
def generate_readme(df, insights):
    readme_path = os.path.join(CONFIG["OUTPUT_DIR"], "README.md")
    with open(readme_path, "w") as f:
        f.write("# Data Analysis Report\n\n")
        f.write("## Dataset Overview\n")
        f.write(f"- Rows: {df.shape[0]}\n")
        f.write(f"- Columns: {df.shape[1]}\n\n")
        f.write("## Key Insights\n")
        f.write(insights + "\n")
        f.write("## Visualizations\n")
        f.write("Refer to the media folder for detailed visualizations.\n")

# Main Analysis
def analyze_data(file_path):
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        analyze_missing_data(df)
        analyze_correlation(df)
        perform_clustering(df)
        context = f"Dataset Overview:\nRows: {df.shape[0]}, Columns: {df.shape[1]}."
        insights = ask_llm("Provide a summary of insights.", context)
        generate_readme(df, insights)
    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset1.csv> [<dataset2.csv> ...]")
        sys.exit(1)
    for dataset in sys.argv[1:]:
        analyze_data(dataset)
