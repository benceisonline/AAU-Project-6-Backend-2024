import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import joblib
from supabase_utils import supabase

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
        
        response = supabase.table('Articles').select('*').in_('article_id', top_item_ids).execute()

        if response == None:
            raise ValueError("No articles found with the given IDs")
        
        return {"news": response.data}

if __name__ == "__main__":
    train_data_path = "exported_data/train_data.csv"
    test_data_path = "exported_data/valid_data.csv"
    model_path = "Saved_Model/lightfm_model_joblib.joblib"

    recommender_system = RecommenderSystem(train_data_path, test_data_path, model_path)
    recommender_system.load_data()
    recommender_system.load_model()

    user_id = 1078040
    predictions_df = recommender_system.make_predictions_for_user(user_id)
    print(predictions_df)