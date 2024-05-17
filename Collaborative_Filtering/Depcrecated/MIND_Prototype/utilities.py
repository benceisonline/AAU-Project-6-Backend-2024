from fastapi import HTTPException
import torch
import pandas as pd
from Collaborative_Filtering.Depcrecated.PyTorch_Prototype.newsmf_model import NewsMF  # Assuming 'newsmf_model' is the correct filename

def load_data():
    # Load news data
    news_data = pd.read_csv(
        "./MIND_small/news.tsv",
        sep="\t",
        names=["itemId", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    )

    # Build index of items
    ind2item = {idx + 1: itemid for idx, itemid in enumerate(news_data['itemId'].values)}
    item2ind = {itemid: idx for idx, itemid in ind2item.items()}

    # Load user behavior data
    raw_behaviour_data = pd.read_csv(
        "./MIND_small/behaviors.tsv",
        sep="\t",
        names=["impressionId", "userId", "timestamp", "click_history", "impressions"]
    )

    # Build index of users
    unique_userIds = raw_behaviour_data['userId'].unique()
    ind2user = {idx + 1: itemid for idx, itemid in enumerate(unique_userIds)}
    user2ind = {itemid: idx for idx, itemid in ind2user.items()}

    # Print the number of unique users for testing purposes
    print(f"We have {len(user2ind)} unique users in the dataset")

    raw_behaviour_data['userIdx'] = raw_behaviour_data['userId'].map(lambda x: user2ind.get(x, 0))

    return news_data, raw_behaviour_data, ind2item, item2ind, ind2user, user2ind

def initialize_model(ind2user, ind2item):
    model = NewsMF(num_users=len(ind2user) + 1, num_items=len(ind2item) + 1)
    model.load_state_dict(torch.load("Saved_Model/collaborative_filtering_model.pth"))
    model.eval()
    return model

# Function to make predictions
def make_predictions(request, model, user2ind, ind2item, news_data):
    user_id = request.user_id
    user_idx = user2ind.get(user_id, 0)
    item_ids = list(ind2item.keys())

    # If user not found, raise an HTTPException
    if user_idx == 0:
        raise HTTPException(status_code=404, detail="User not found")

    predictions = model.forward(torch.tensor([user_idx] * len(item_ids)), torch.tensor(item_ids))
    top_indices = torch.topk(predictions.flatten(), 10).indices
    top_item_ids = [ind2item[ix.item()] for ix in top_indices]
    recommended_items = news_data[news_data["itemId"].isin(top_item_ids)].to_dict(orient='records')

    return {"user_id": user_id, "recommended_items": recommended_items}