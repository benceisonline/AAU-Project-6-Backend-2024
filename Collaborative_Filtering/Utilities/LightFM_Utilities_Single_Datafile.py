# USE THIS UTILITY FILE WHEN WORKING WITH A SINGLE DATAFILE FOR BOTH TRAINING AND TESTING DATA - NOT PRE-SPLIT

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation
from lightfm.evaluation import auc_score
import joblib
import os
from supabase_utils import fetch_data_from_supabase, fetch_articles

# This version of the RecommenderSystem works with a single data file
class RecommenderSystem:
    def __init__(self, data_path, models_folder_path):
        self.data_path = data_path
        self.models_folder_path = models_folder_path
        self.model = None
        self.dataset = None

    # Load data from CSV files and build interactions
    # Paths are given at recommender initialization
    def load_data(self):
        TEST_PERCENTAGE = 0.25 # percentage of data used for testing
        SEED = 42 # seed for pseudonumber generations

        # Load data from Supabase (Supabase is not being nice right now)
        # data = fetch_data_from_supabase('LightFM')
        
        data = pd.read_csv(self.data_path)

        self.dataset = Dataset()
        self.dataset.fit(users=data['userID'], items=data['itemID'])

        (interactions, weights) = self.dataset.build_interactions(data.iloc[:, 0:3].values)
        self.train_interactions, self.test_interactions = cross_validation.random_train_test_split(
        interactions, test_percentage=TEST_PERCENTAGE, random_state=np.random.RandomState(SEED))

        print("Train interactions shape:", self.train_interactions.shape)
        print("Test interactions shape:", self.test_interactions.shape)

    # Load model from joblib file with given model_id
    def load_model(self, model_id):
        model_file = os.path.join(self.models_folder_path, f"{model_id}.joblib")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found for model ID: {model_id}")

        # Load the model
        self.model = joblib.load(model_file)
        print(f"Loaded model {model_id}")

    # Make predictions for a given user and return number of news items based on request
    def make_predictions_for_user(self, request):
        user_id = int(request.user_id) 
        start_index = request.start_index
        num_of_recs = request.no_recommendations

        if user_id not in self.dataset.mapping()[0]:
            raise ValueError("User ID not found in the dataset")

        internal_user_id = self.dataset.mapping()[0][user_id]
        all_item_ids = np.array(list(self.dataset.mapping()[2].values()))

        predictions = self.model.predict(internal_user_id, all_item_ids)

        actual_item_ids = np.array(list(self.dataset.mapping()[2].keys()))

        sorted_indices = np.argsort(predictions)[::-1]
        sorted_item_ids = actual_item_ids[sorted_indices]

        top_item_ids = sorted_item_ids[start_index:start_index+num_of_recs]
    
        articles = fetch_articles(top_item_ids)

        return {"news": articles}

    # As a cautionary note to new users, when using model.fit_partial 
    # the users/items/features in the supplied matrices 
    # must have been present during the initial training. 
    # Meaning, if you want to add new users/items/features, 
    # you must retrain the model. This is why we use retrain 
    # not partial fit.
    def retrain(self, epochs):
        self.load_data()

        # model learning rate
        LEARNING_RATE = 0.25
        # no of latent factors
        NO_COMPONENTS = 20
        # seed for pseudonumber generations
        SEED = 42

        self.model = LightFM(loss='warp', no_components=NO_COMPONENTS,
                 learning_rate=LEARNING_RATE,
                 random_state=np.random.RandomState(SEED))
        self.model.fit(interactions=self.train_interactions, epochs=epochs);
        joblib.dump(self.model, 'saved_models/lightfm_model_retrained.joblib')

    # Get AUC score for the model
    def get_validation_AUC_score(self, num_threads=1):
        test_interactions_excl_train = self.test_interactions - self.train_interactions.multiply(self.test_interactions)

        auc_scores = auc_score(
            self.model,
            test_interactions=test_interactions_excl_train,
            num_threads=num_threads,
        )

        average_auc = np.mean(auc_scores)
        print(f"Average AUC Score: {average_auc}")

        return average_auc

class Request:
    def __init__(self, user_id, start_index, no_recommendations):
        self.user_id = user_id
        self.start_index = start_index
        self.no_recommendations = no_recommendations
