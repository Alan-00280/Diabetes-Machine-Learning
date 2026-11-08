"""
Utility module for California Housing Price Prediction application.
Handles model loading, input validation, and preprocessing pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURE_COLUMNS = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'rooms_per_household',
    'bedrooms_per_room',
    'population_per_household',
    'ocean_proximity_enc'
]

OCEAN_PROXIMITY_OPTIONS = [
    '<1H OCEAN',
    'INLAND',
    'NEAR OCEAN',
    'NEAR BAY',
    'ISLAND'
]


@st.cache_resource
def load_artifacts(model_dir: str):
    """
    Loads model, scaler, and categorical encoder from the model directory.
    Cached via Streamlit to ensure optimal performance.
    """
    model_path = os.path.join(model_dir, 'trained_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    encoder_path = os.path.join(model_dir, 'ocean_encoder.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found at: {encoder_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    return model, scaler, encoder


def validate_inputs(inputs: dict) -> list[str]:
    """
    Validates user input values against logical domain constraints.
    Returns a list of error messages, or an empty list if valid.
    """
    errors = []

    if inputs['total_rooms'] <= 0:
        errors.append("Total rooms must be greater than zero.")
    if inputs['total_bedrooms'] <= 0:
        errors.append("Total bedrooms must be greater than zero.")
    if inputs['population'] <= 0:
        errors.append("Population must be greater than zero.")
    if inputs['households'] <= 0:
        errors.append("Households must be greater than zero.")
    if inputs['median_income'] <= 0:
        errors.append("Median income must be greater than zero.")
    if inputs['total_bedrooms'] > inputs['total_rooms']:
        errors.append("Total bedrooms cannot exceed total rooms in a block group.")
    if inputs['households'] > inputs['population']:
        errors.append("Households cannot exceed total population.")

    return errors


def preprocess_input(inputs: dict, encoder, scaler) -> tuple[np.ndarray, dict]:
    """
    Transforms raw inputs into model-ready scaled feature vectors.
    Computes derived features and ordinal encoding matching the training pipeline.
    """
    # 1. Feature Engineering
    rooms_per_household = inputs['total_rooms'] / inputs['households']
    bedrooms_per_room = inputs['total_bedrooms'] / inputs['total_rooms']
    population_per_household = inputs['population'] / inputs['households']

    # 2. Categorical Encoding (Ocean Proximity)
    ocean_df = pd.DataFrame({'ocean_proximity': [inputs['ocean_proximity']]})
    ocean_enc = encoder.transform(ocean_df)[0][0]

    # 3. Assemble full feature dictionary in exact training order
    raw_feature_dict = {
        'longitude': inputs['longitude'],
        'latitude': inputs['latitude'],
        'housing_median_age': inputs['housing_median_age'],
        'total_rooms': inputs['total_rooms'],
        'total_bedrooms': inputs['total_bedrooms'],
        'population': inputs['population'],
        'households': inputs['households'],
        'median_income': inputs['median_income'],
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household,
        'ocean_proximity_enc': ocean_enc
    }

    feature_df = pd.DataFrame([raw_feature_dict])[FEATURE_COLUMNS]

    # 4. Scaling (values only to match train fit array without warnings)
    scaled_array = scaler.transform(feature_df.values)

    derived_summary = {
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household,
        'ocean_proximity_enc': int(ocean_enc)
    }

    return scaled_array, derived_summary


def predict_price(model, preprocessed_features: np.ndarray) -> float:
    """
    Executes regression model prediction and ensures non-negative pricing.
    """
    prediction = model.predict(preprocessed_features)
    predicted_val = float(prediction[0])
    return max(0.0, predicted_val)

