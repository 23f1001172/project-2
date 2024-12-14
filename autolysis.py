# --- metadata ---
# script_name = "autolysis.py"
# version = "2.0"
# description = "A Python script for robust dataset analysis, including insights, visualizations, clustering, and narratives."
# author = "23f1001172"
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "chardet", "requests", "pillow", "pytesseract"]
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
from PIL import Image
import pytesseract
import numpy as np
import chardet
import requests
import sys
import subprocess

# Ensure all required dependencies are installed
def ensure_dependencies():
    required = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "chardet", "requests", "pillow", "pytesseract"]
    for package in required:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

ensure_dependencies()

# Configuration for AI Proxy and output
CONFIG = {
    "AI_PROXY_URL": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "AIPROXY_TOKEN": os.getenv("AIPROXY_TOKEN", ""),
    "OUTPUT_DIR": os.path.join(os.getcwd(), "media"),
}
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# Save visualization helper function
def save_visualization(plt_obj, filename):
    """Save visualizations to the output directory."""
    file_path = os.path.join(CONFIG["OUTPUT_DIR"], filename)
    plt_obj.tight_layout()
    plt_obj.savefig(file_path, bbox_inches="tight")
    plt_obj.close()
    return file_path

# Detect file encoding
def detect_encoding(file_path):
    """Detect file encoding to handle diverse datasets."""
    with open(file_path, "rb") as f:
        raw_data = f.read()
    return chardet.detect(raw_data)["encoding"]

# Analyze missing data
def analyze_missing_data(df):
    """Analyze and visualize missing data."""
    missing_percent = (df.isnull().sum() / len(df)) * 100
    plt.figure(figsize=(10, 6))
    missing_percent.plot(kind="bar", color="salmon")
    plt.title("Missing Data Analysis", fontsize=16)
    plt.xlabel("Columns")
    plt.ylabel("Percentage of Missing Values")
    save_visualization(plt, "missing_data.png")
    return missing_percent

# Analyze correlations
def analyze_correlation(df):
    """Analyze and visualize correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None

    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=16)
    plt.xlabel("Features")
    plt.ylabel("Features")
    save_visualization(plt, "correlation_heatmap.png")
    return corr_matrix

# Perform clustering
def perform_clustering(df):
    """Perform clustering and visualize results."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns for clustering.")
        return None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, labels)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_data, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, palette="viridis", s=100)
    plt.title(f"KMeans Clustering (Silhouette Score: {silhouette_avg:.2f})", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    save_visualization(plt, "clustering.png")
    return silhouette_avg

# Perform OCR on images
def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        print(f"Extracted Text: {text[:100]}...")
        return text
    except Exception as e:
        print(f"Error in OCR: {e}")
        return None

# Interact with LLM
def ask_llm(question, context):
    """Generate insights using LLM with dynamic prompts."""
    headers = {"Authorization": f"Bearer {CONFIG['AIPROXY_TOKEN']}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": f"{question}\nContext:\n{context}"}
        ],
    }
    response = requests.post(CONFIG["AI_PROXY_URL"], headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Generate README
def generate_readme(df, insights):
    """Generate a well-structured README file summarizing analysis."""
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

# Main analysis function
def analyze_data(file_path):
    """Main analysis workflow."""
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)

        # Perform analysis
        missing_data = analyze_missing_data(df)
        corr_matrix = analyze_correlation(df)
        clustering_score = perform_clustering(df)

        # Prepare context for LLM
        context = f"""
        Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.
        Missing Data Summary: {missing_data.to_string() if not missing_data.empty else "None"}
        Correlation Analysis: {"Performed" if corr_matrix is not None else "Not available"}
        Clustering Silhouette Score: {clustering_score if clustering_score is not None else "Not performed"}.
        """
        insights = ask_llm("Generate a detailed summary of insights based on the dataset.", context)

        # Generate README
        generate_readme(df, insights)
    except Exception as e:
        print(f"Error analyzing data: {e}")
        sys.exit(1)

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset1.csv> [<dataset2.csv> ...]")
        sys.exit(1)

    datasets = sys.argv[1:]
    for dataset in datasets:
        analyze_data(dataset)
