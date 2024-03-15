from fastapi import HTTPException
import torch
import pandas as pd
from newsmf_model import NewsMF

def load_data():
    # Load news data
    news_data = pd.read_parquet("./ebnerd_small/articles.parquet")

    # Load behaviour data
    train_behaviour = pd.read_parquet("./ebnerd_small/train/behaviors.parquet")
    valid_behaviour = pd.read_parquet("./ebnerd_small/validation/behaviors.parquet")
    behaviors = pd.concat([train_behaviour, valid_behaviour], ignore_index=True)

    # Load history data
    train_history = pd.read_parquet("./ebnerd_small/train/history.parquet")
    valid_history = pd.read_parquet("./ebnerd_small/validation/history.parquet")
    history = pd.concat([train_history, valid_history], ignore_index=True)

    # Join behaviour and history data
    behaviour_history_merged= pd.merge(behaviors, history, on='user_id', how='left')

    # Build index of items    
    ind2article = {idx + 1: itemid for idx, itemid in enumerate(news_data['article_id'].values)}
    article2ind = {itemid: idx for idx, itemid in ind2article.items()}

    # Build index of users
    unique_userIds = behaviour_history_merged['user_id'].unique()
    ind2user = {idx + 1: itemid for idx, itemid in enumerate(unique_userIds)}
    user2ind = {itemid: idx for idx, itemid in ind2user.items()}

    return news_data, ind2article, article2ind, ind2user, user2ind

def initialize_model(ind2user, ind2item):
    model = NewsMF(num_users=len(ind2user) + 1, num_items=len(ind2item) + 1)
    model.load_state_dict(torch.load("Saved_Model/EBNERD_collaborative_filtering_model.pth"))
    model.eval()
    return model

# Function to make predictions
def make_predictions(request, model, user2ind, ind2article, news_data):
    user_id = int(request.user_id)  # Convert user_id to int
    no_recommendations = request.no_recommendations

    # Get user index
    user_idx = user2ind.get(user_id)

    # Print unique user IDs during prediction
    print(f"User ID: {user_id}")
    print(f"User index: {user_idx}")

    # If user_id is not found, raise an HTTPException
    if user_idx is None:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
    
    if no_recommendations is None:
        raise HTTPException(status_code=404, detail=f"Please provide a valid number of recommendations.")
    
    if no_recommendations < 1:
        raise HTTPException(status_code=404, detail=f"Please provide a number of recommendations over 0.")

    item_ids = list(ind2article.keys())

    predictions = model.forward(torch.tensor([user_idx] * len(item_ids)), torch.tensor(item_ids))
    top_indices = torch.topk(predictions.flatten(), no_recommendations).indices
    top_item_ids = [ind2article[ix.item()] for ix in top_indices]
    
    # Explicitly type all column names
    all_column_names = ["article_id", "title", "subtitle", "last_modified_time", "premium", "body",
                    "published_time", "image_ids", "article_type", "url", "entity_groups", "topics",
                    "category", "subcategory", "category_str", "total_inviews", "total_pageviews",
                    "total_read_time", "sentiment_score", "sentiment_label"]
    
    # image_ids, enitity_groups, topics, subcategory, total_inviews, total_pageviews, total_read_time are freaky for some reason
    column_names = ["article_id", "title", "subtitle", "last_modified_time", "premium", "body", "published_time", 
                    "article_type", "url", "category", "category_str", "sentiment_score", "sentiment_label"]

    # Extract all columns from news_data
    recommended_items = news_data[news_data["article_id"].isin(top_item_ids)][column_names].to_dict(orient='records')

    return {"user_id": user_id, "recommended_items": recommended_items}
