"""
Data acquisition script for Heart Disease UCI dataset
Downloads the dataset from UCI Machine Learning Repository
"""
import os
import pandas as pd
import requests
from pathlib import Path


def download_heart_disease_data(output_dir: str = "data/raw"):
    """
    Download Heart Disease UCI dataset from UCI ML Repository
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # UCI Heart Disease dataset URLs
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    ]
    
    # Column names as per UCI documentation
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    all_dataframes = []
    
    for url in urls:
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw data
            filename = url.split('/')[-1]
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Read and combine data
            df = pd.read_csv(filepath, header=None, na_values='?')
            df.columns = column_names
            all_dataframes.append(df)
            print(f"Successfully downloaded {filename}")
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    # Combine all datasets
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        output_path = os.path.join(output_dir, "heart_disease.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"\nCombined dataset saved to {output_path}")
        print(f"Dataset shape: {combined_df.shape}")
        return combined_df
    else:
        raise Exception("Failed to download any data")
    

if __name__ == "__main__":
    print("Downloading Heart Disease UCI dataset...")
    df = download_heart_disease_data()
    print("Download completed!")

