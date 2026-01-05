#!/usr/bin/env python3
"""
Sample prediction script for testing the API
"""
import requests
import json
import sys


def make_prediction(api_url="http://localhost:8000", sample_data=None):
    """
    Make a prediction using the API
    
    Args:
        api_url: Base URL of the API
        sample_data: Optional custom data, otherwise uses default
    """
    if sample_data is None:
        # Default sample data
        sample_data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
    
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        print("Prediction Result:")
        print(json.dumps(result, indent=2))
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error making prediction: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make a prediction using the Heart Disease API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--age", type=float, help="Age")
    parser.add_argument("--sex", type=float, help="Sex (1=male, 0=female)")
    parser.add_argument("--cp", type=float, help="Chest pain type")
    parser.add_argument("--trestbps", type=float, help="Resting blood pressure")
    parser.add_argument("--chol", type=float, help="Cholesterol")
    parser.add_argument("--fbs", type=float, help="Fasting blood sugar")
    parser.add_argument("--restecg", type=float, help="Resting ECG")
    parser.add_argument("--thalach", type=float, help="Max heart rate")
    parser.add_argument("--exang", type=float, help="Exercise induced angina")
    parser.add_argument("--oldpeak", type=float, help="ST depression")
    parser.add_argument("--slope", type=float, help="Slope")
    parser.add_argument("--ca", type=float, help="Number of vessels")
    parser.add_argument("--thal", type=float, help="Thalassemia")
    
    args = parser.parse_args()
    
    # Use provided arguments or default
    if any([args.age, args.sex, args.cp]):
        sample_data = {
            "age": args.age or 63,
            "sex": args.sex or 1,
            "cp": args.cp or 3,
            "trestbps": args.trestbps or 145,
            "chol": args.chol or 233,
            "fbs": args.fbs or 1,
            "restecg": args.restecg or 0,
            "thalach": args.thalach or 150,
            "exang": args.exang or 0,
            "oldpeak": args.oldpeak or 2.3,
            "slope": args.slope or 0,
            "ca": args.ca or 0,
            "thal": args.thal or 1
        }
        make_prediction(args.url, sample_data)
    else:
        make_prediction(args.url)

