"""
Data preprocessing pipeline for Heart Disease dataset
Handles missing values, encoding, and feature scaling
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib


class HeartDiseasePreprocessor:
    """Preprocessing pipeline for heart disease data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """
        Fit preprocessing transformers on training data
        
        Args:
            df: Training dataframe
        """
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        # Fit imputer
        self.imputer.fit(df[numeric_cols])
        
        # Fit scaler (after imputation)
        X_filled = self.imputer.transform(df[numeric_cols])
        self.scaler.fit(X_filled)
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        # Impute missing values
        df[numeric_cols] = self.imputer.transform(df[numeric_cols])
        
        # Scale features
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # Binarize target (0 = no disease, 1 = disease)
        if 'target' in df.columns:
            df['target'] = (df['target'] > 0).astype(int)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    def save(self, filepath: str):
        """Save preprocessor to disk"""
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor from disk"""
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.imputer = data['imputer']
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor


def load_and_preprocess_data(data_path: str = "data/raw/heart_disease.csv"):
    """
    Load and preprocess heart disease dataset
    
    Args:
        data_path: Path to raw data CSV file
        
    Returns:
        Preprocessed dataframe
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor()
    
    # Fit and transform
    df_processed = preprocessor.fit_transform(df)
    
    return df_processed, preprocessor


if __name__ == "__main__":
    # Example usage
    df, preprocessor = load_and_preprocess_data()
    print("Preprocessed data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nTarget distribution:")
    print(df['target'].value_counts())

