# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
#   "chardet",
#   "requests",
# ]
# usage = "python autolysis.py <dataset1.csv> [<dataset2.csv> ...]"
# description = "A script to analyze datasets dynamically and generate visualizations, summaries, and insights."
# ///

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import chardet
import requests
import sys

# Configuration for API Proxy
CONFIG = {
    "AI_PROXY_URL": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "AIPROXY_TOKEN": os.environ.get("AIPROXY_TOKEN", ""),
    "OUTPUT_DIR": os.path.join(os.getcwd(), "media"),
}

# Ensure output directory exists
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# LLM Interaction Function
def ask_llm(question, context):
    """Send prompts dynamically to the LLM."""
    try:
        headers = {
            "Authorization": f"Bearer {CONFIG['AIPROXY_TOKEN']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": f"{question}\nContext:\n{context}"}
            ],
        }
        response = requests.post(CONFIG["AI_PROXY_URL"], headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error interacting with LLM: {e}")
        return "LLM interaction failed."

# Save Visualization Helper
def save_visualization(plt_obj, filename):
    """Save visualizations to the output directory."""
    try:
        plt_obj.tight_layout()
        file_path = os.path.join(CONFIG["OUTPUT_DIR"], filename)
        plt_obj.savefig(file_path, bbox_inches="tight")
        plt_obj.close()
        return file_path
    except Exception as e:
        print(f"Error saving visualization: {e}")
        return None

# Detect File Encoding
def detect_encoding(file_path):
    """Detect file encoding to handle diverse datasets."""
    with open(file_path, "rb") as f:
        raw_data = f.read()
    return chardet.detect(raw_data)["encoding"]

# Missing Data Analysis
def analyze_missing_data(df):
    """Analyze and visualize missing data."""
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_summary = missing_percent[missing_percent > 0].sort_values(ascending=False)

    if not missing_summary.empty:
        plt.figure(figsize=(10, 6))
        missing_summary.plot(kind="bar", color="salmon")
        plt.title("Missing Data by Column")
        plt.ylabel("Percentage")
        save_visualization(plt, "missing_data.png")

    return missing_summary

# Correlation Heatmap
def analyze_correlation(df):
    """Analyze and visualize correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        save_visualization(plt, "correlation_heatmap.png")
        return corr_matrix
    else:
        print("No numeric columns for correlation analysis.")
    return None

# Clustering with Visualization
def perform_clustering(df):
    """Perform clustering and visualize results."""
    numerical_cols = df.select_dtypes(include=[np.number]).dropna()
    if len(numerical_cols.columns) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_cols)
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
        plt.title(f"KMeans Clustering (Silhouette Score: {silhouette_avg:.2f})")
        save_visualization(plt, "clustering.png")
        return silhouette_avg
    else:
        print("Not enough numeric columns for clustering.")
    return None

# Generate README
def generate_readme(df, missing_summary, corr_matrix, clustering_score):
    """Generate a README file summarizing analysis."""
    context = f"""
    Dataset has {df.shape[0]} rows and {df.shape[1]} columns.
    Missing Data: {missing_summary.to_string() if not missing_summary.empty else "None"}
    Correlation Analysis: {'Performed' if corr_matrix is not None else 'Not available (No numeric columns)'}
    Clustering Silhouette Score: {clustering_score if clustering_score is not None else "Not performed"}
    """
    story = ask_llm("Generate a summary and narrative based on the dataset insights.", context)
    readme_path = os.path.join(CONFIG["OUTPUT_DIR"], "README.md")

    with open(readme_path, "w") as readme_file:
        readme_file.write("# Data Analysis Report\n\n")
        readme_file.write("## Summary\n\n")
        readme_file.write(context + "\n\n")
        readme_file.write("## Narrative Insights\n\n")
        readme_file.write(story)

# Main Function
def analyze_data(file_path):
    """Main analysis workflow."""
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        missing_summary = analyze_missing_data(df)
        corr_matrix = analyze_correlation(df)
        clustering_score = perform_clustering(df)
        generate_readme(df, missing_summary, corr_matrix, clustering_score)
    except Exception as e:
        print(f"Error analyzing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset1.csv> [<dataset2.csv> ...]")
        sys.exit(1)

    datasets = sys.argv[1:]

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        analyze_data(dataset)
