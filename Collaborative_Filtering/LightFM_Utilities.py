import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
import joblib

class RecommenderSystem:
    def __init__(self, train_data_path, test_data_path, model_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_path = model_path
        self.model = None
        self.dataset = None

    def load_data(self):
        train_data = pd.read_csv(self.train_data_path)
        test_data = pd.read_csv(self.test_data_path)

        self.dataset = Dataset()
        self.dataset.fit(users=pd.concat([train_data['userID'], test_data['userID']]),
                         items=pd.concat([train_data['itemID'], test_data['itemID']]))

        self.train_interactions, _ = self.dataset.build_interactions(train_data.iloc[:, 0:3].values)
        self.test_interactions, _ = self.dataset.build_interactions(test_data.iloc[:, 0:3].values)

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def make_predictions_for_user(self, request, news_data): #replace user_id with request
        user_id = int(request.user_id)  # Convert user_id to int
        num_of_recs = request.no_recommendations

        if user_id not in self.dataset.mapping()[0]:
            raise ValueError("User ID not found in the dataset")

        internal_user_id = self.dataset.mapping()[0][user_id]
        all_item_ids = np.array(list(self.dataset.mapping()[2].values()))

        predictions = self.model.predict(internal_user_id, all_item_ids)

        actual_item_ids = np.array(list(self.dataset.mapping()[2].keys()))

        sorted_indices = np.argsort(predictions)[::-1]
        sorted_predictions = predictions[sorted_indices]
        sorted_item_ids = actual_item_ids[sorted_indices]

        top_item_ids = sorted_item_ids[:num_of_recs]
        top_predictions = sorted_predictions[:num_of_recs]

        # Extract relevant columns for recommendation
        column_names = ["article_id", "title", "subtitle", "last_modified_time", "premium", "body", "published_time", 
                        "article_type", "url", "category", "category_str", "sentiment_score", "sentiment_label", "image_ids"]

        # Extract recommended items
        recommended_items = news_data[news_data["article_id"].isin(top_item_ids)][column_names].to_dict(orient='records')

        return {"news": recommended_items}
    
    def make_predictions_for_user_test(self, user_id, news_data): #replace user_id with request
        #user_id = int(request.user_id)  # Convert user_id to int
        user_id = user_id
        #num_of_recs = request.no_recommendations
        num_of_recs = 10

        if user_id not in self.dataset.mapping()[0]:
            raise ValueError("User ID not found in the dataset")

        internal_user_id = self.dataset.mapping()[0][user_id]
        all_item_ids = np.array(list(self.dataset.mapping()[2].values()))

        predictions = self.model.predict(internal_user_id, all_item_ids)

        actual_item_ids = np.array(list(self.dataset.mapping()[2].keys()))

        sorted_indices = np.argsort(predictions)[::-1]
        sorted_predictions = predictions[sorted_indices]
        sorted_item_ids = actual_item_ids[sorted_indices]

        top_item_ids = sorted_item_ids[:num_of_recs]
        top_predictions = sorted_predictions[:num_of_recs]

        # Extract relevant columns for recommendation
        column_names = ["article_id", "title", "subtitle", "last_modified_time", "premium", "body", "published_time", 
                        "article_type", "url", "category", "category_str", "sentiment_score", "sentiment_label", "image_ids"]

        # Extract recommended items
        recommended_items = news_data[news_data["article_id"].isin(top_item_ids)][column_names].to_dict(orient='records')

        return {"news": recommended_items}
    
    def load_half_trained_model(self):
        self.model = joblib.load("Saved_Model/lightfm_model_joblib_half_trained.joblib")

    def load_half_data(self):
        train_data_half = pd.read_csv("exported_data/train_data_half.csv")
        test_data = pd.read_csv(self.test_data_path)

        self.dataset = Dataset()
        self.dataset.fit(users=pd.concat([train_data_half['userID'], test_data['userID']]),
                         items=pd.concat([train_data_half['itemID'], test_data['itemID']]))

        self.train_interactions, _ = self.dataset.build_interactions(train_data_half.iloc[:, 0:3].values)
        self.test_interactions, _ = self.dataset.build_interactions(test_data.iloc[:, 0:3].values)

    # Have not tested if it works

    # As a cautionary note to new users, when using model.fit_partial 
    # the users/items/features in the supplied matrices 
    # must have been present during the initial training. 
    # Meaning, if you want to add new users/items/features, 
    # you must retrain the model.

    # What should happen is, new train data is loaded, the model is trained on it, and AUC is outputted
    def retrain(self, epochs=1, num_threads=1):
        self.load_data()
        self.model.fit(interactions=self.train_interactions, epochs=epochs);
        self.get_validation_AUC_score(num_threads=num_threads)
        joblib.dump(self.model, 'lightfm_model_joblib_retrained.joblib')

    def get_validation_AUC_score(self, num_threads=1):
        # Calculate AUC score after each training iteration
        test_interactions_excl_train = self.test_interactions - self.train_interactions.multiply(self.test_interactions)
        auc_scores = auc_score(self.model, test_interactions=test_interactions_excl_train, num_threads=num_threads)
        num_auc_scores = len(auc_scores)
        average_auc_score = np.mean(auc_scores)
        print("Number of AUC scores calculated:", num_auc_scores)
        print("Average AUC score:", average_auc_score)

if __name__ == "__main__":
    train_data_path = "exported_data/train_data.csv"
    test_data_path = "exported_data/valid_data.csv"
    model_path = "Saved_Model/lightfm_model_joblib.joblib"
    news_data = pd.read_parquet("./ebnerd_small/articles.parquet")

    recommender_system = RecommenderSystem(train_data_path, test_data_path, model_path)
    recommender_system.load_half_data() # Load half of the training data
    recommender_system.load_half_trained_model() # Load half trained model (get it from Colab)

    user_id = 1473460
    predictions_df = recommender_system.make_predictions_for_user_test(user_id, news_data)
    #print(predictions_df)

    # AUC score with half of the training data
    recommender_system.get_validation_AUC_score()

    # Partial fit with other half of the training data, should result in fully trained model
    recommender_system.retrain()

    # AUC score with full training data
    recommender_system.get_validation_AUC_score()