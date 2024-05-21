# pytest tests/test_utilities.py
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Now import modules from project root directory
from api_utilities.lightfm_utilities_single_datafile import RecommenderSystem, Request

# Define test data paths
DATA_PATH = "exported_data/combined_data_small.csv"
MODEL_FOLDER_PATH = "saved_models/"
MODEL_ID = "lightfm_model_combined_data_small"

@pytest.fixture
def recommender_system():
    # Initialize RecommenderSystem
    recommender_system = RecommenderSystem(DATA_PATH, MODEL_FOLDER_PATH)
    return recommender_system

def test_load_data(recommender_system):
    recommender_system.load_data()
    assert recommender_system.dataset is not None

def test_load_model(recommender_system):
    recommender_system.load_model(MODEL_ID)
    assert recommender_system.model is not None


def test_get_validation_AUC_score(recommender_system):
    recommender_system.load_data()
    recommender_system.load_model(MODEL_ID)
    auc_score = recommender_system.get_validation_AUC_score()
    assert auc_score is not None
    assert isinstance(auc_score, np.floating)

# Test retrain method
def test_retrain(recommender_system):
    recommender_system.load_data()
    recommender_system.load_model(MODEL_ID)
    auc_score_before_retrain = recommender_system.get_validation_AUC_score()

    epochs = 1  # Or any number of epochs you want to test with
    recommender_system.retrain(epochs)
    recommender_system.load_model("lightfm_model_retrained")
    auc_score_after_retrain = recommender_system.get_validation_AUC_score()

    # Ensure model is retrained
    # This is a questionable test, as the AUC score could in theory be the same
    # And it does not necessarily mean the model was properly retrained
    assert recommender_system.model is not None
    assert auc_score_after_retrain != auc_score_before_retrain

# Test make_predictions_for_user method
def test_make_predictions_for_user(recommender_system):
    recommender_system.load_data()
    recommender_system.load_model(MODEL_ID)
    news_data = pd.read_parquet("./ebnerd_data/ebnerd_small/articles.parquet")
    test_request = Request(user_id="136336", start_index=1, no_recommendations=10)
    response = recommender_system.make_predictions_for_user(test_request)
    
    # Check if the dictionary is not empty
    assert response
    # Check if response resembles a Supabase table
    assert isinstance(response, dict)