# pytest tests/test_utilities.py
import pytest
import pandas as pd
import numpy as np
from LightFM_Utilities_Single_Datafile import RecommenderSystem, Request

# Define test data paths
DATA_PATH = "exported_data/combined_data_small.csv"
MODEL_FOLDER_PATH = "Saved_Model/"
MODEL_ID = "lightfm_model_multi_file"

@pytest.fixture
def test_request():
    return Request(user_id="136336", start_index=1, no_recommendations=10)

@pytest.fixture
def recommender_system():
    # Initialize RecommenderSystem and load the model
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

# Test make_predictions_for_user method