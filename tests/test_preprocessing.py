"""
Unit tests for data preprocessing
"""
import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import HeartDiseasePreprocessor


@pytest.fixture
def sample_data():
    """Create sample dataframe for testing"""
    data = {
        'age': [63, 37, 41, 56, 57],
        'sex': [1, 1, 0, 1, 0],
        'cp': [3, 2, 1, 1, 0],
        'trestbps': [145, 130, 130, 120, 120],
        'chol': [233, 250, 204, 236, 354],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 1, 0, 1, 1],
        'thalach': [150, 187, 172, 178, 163],
        'exang': [0, 0, 0, 0, 1],
        'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
        'slope': [0, 0, 2, 2, 2],
        'ca': [0, 0, 0, 0, 0],
        'thal': [1, 2, 2, 2, 2],
        'target': [1, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing():
    """Create sample dataframe with missing values"""
    data = {
        'age': [63, 37, np.nan, 56, 57],
        'sex': [1, 1, 0, 1, 0],
        'cp': [3, 2, 1, 1, 0],
        'trestbps': [145, np.nan, 130, 120, 120],
        'chol': [233, 250, 204, 236, 354],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 1, 0, 1, 1],
        'thalach': [150, 187, 172, 178, 163],
        'exang': [0, 0, 0, 0, 1],
        'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
        'slope': [0, 0, 2, 2, 2],
        'ca': [0, 0, 0, 0, 0],
        'thal': [1, 2, 2, 2, 2],
        'target': [1, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)


def test_preprocessor_fit(sample_data):
    """Test preprocessor fitting"""
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(sample_data)
    assert preprocessor.is_fitted == True


def test_preprocessor_transform(sample_data):
    """Test preprocessor transformation"""
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)
    
    # Check that target is binarized
    assert transformed['target'].isin([0, 1]).all()
    
    # Check that numeric columns are scaled
    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('target')
    assert len(transformed.columns) == len(sample_data.columns)


def test_preprocessor_fit_transform(sample_data):
    """Test fit_transform method"""
    preprocessor = HeartDiseasePreprocessor()
    transformed = preprocessor.fit_transform(sample_data)
    
    assert preprocessor.is_fitted == True
    assert transformed.shape[0] == sample_data.shape[0]
    assert 'target' in transformed.columns


def test_preprocessor_missing_values(sample_data_with_missing):
    """Test handling of missing values"""
    preprocessor = HeartDiseasePreprocessor()
    transformed = preprocessor.fit_transform(sample_data_with_missing)
    
    # Check no missing values remain
    numeric_cols = transformed.select_dtypes(include=[np.number]).columns
    assert transformed[numeric_cols].isna().sum().sum() == 0


def test_preprocessor_save_load(sample_data, tmp_path):
    """Test saving and loading preprocessor"""
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(sample_data)
    
    # Save
    filepath = tmp_path / "preprocessor.pkl"
    preprocessor.save(str(filepath))
    
    # Load
    loaded_preprocessor = HeartDiseasePreprocessor.load(str(filepath))
    
    # Test loaded preprocessor
    original_transformed = preprocessor.transform(sample_data)
    loaded_transformed = loaded_preprocessor.transform(sample_data)
    
    pd.testing.assert_frame_equal(original_transformed, loaded_transformed)


def test_preprocessor_raises_if_not_fitted(sample_data):
    """Test that transform raises error if not fitted"""
    preprocessor = HeartDiseasePreprocessor()
    with pytest.raises(ValueError, match="must be fitted"):
        preprocessor.transform(sample_data)

