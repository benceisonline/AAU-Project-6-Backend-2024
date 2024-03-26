import pandas as pd
import torch
import hashlib
from fastapi import HTTPException
from newsmf_model import NewsMF
import numpy as np  # Import NumPy

# Function to load data
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

# Function to initialize model
def initialize_model(ind2user, ind2item):
    model = NewsMF(num_users=len(ind2user) + 1, num_items=len(ind2item) + 1)
    model.load_state_dict(torch.load("Saved_Model/EBNERD_collaborative_filtering_model.pth"))
    model.eval()
    return model

# Function to generate image URLs
def generate_image_url(image_id, article_id, published_timestamp, modified_timestamp):
    hash_string = f"caravaggio-{article_id}-{published_timestamp}-{modified_timestamp}"
    md5_hash = hashlib.md5(hash_string.encode()).hexdigest()
    return f"https://img-cdn-p.ekstrabladet.dk/image/ekstrabladet/{image_id}/relationBig_910/?at={md5_hash}"

def generate_image_urls(recommended_items):
    for item in recommended_items:
        article_id = item['article_id']
        image_ids = item.get('image_ids')
        if image_ids is not None and len(image_ids) > 0:  # Check if image_ids is not None and is not empty
            image_id = image_ids[0]  # Assuming only the first image is used
            published_timestamp = item['published_time'].value / 1000000  # convert to seconds
            modified_timestamp = item['last_modified_time'].value / 1000000  # convert to seconds
            item['image_url'] = generate_image_url(image_id, article_id, published_timestamp, modified_timestamp)
        else:
            # If no image IDs are available, use default image URL
            item['image_url'] = "https://is1-ssl.mzstatic.com/image/thumb/Purple112/v4/66/53/28/66532893-b30e-3023-20be-448c4d19428f/AppIcon-0-0-1x_U007ephone-0-85-220.png/1200x630wa.png"

        # Convert NumPy array to list
        if isinstance(item['image_ids'], np.ndarray):
            item['image_ids'] = item['image_ids'].tolist()

    return recommended_items


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
    
    # Extract relevant columns for recommendation
    column_names = ["article_id", "title", "subtitle", "last_modified_time", "premium", "body", "published_time", 
                    "article_type", "url", "category", "category_str", "sentiment_score", "sentiment_label", "image_ids"]

    # Extract recommended items
    recommended_items = news_data[news_data["article_id"].isin(top_item_ids)][column_names].to_dict(orient='records')
    
    # Generate image links for recommended items
    generate_image_urls(recommended_items)

    return {"user_id": user_id, "news": recommended_items}

def get_newest_news(news_data):
    column_names = ["article_id", "title", "subtitle", "last_modified_time", "premium", "body", "published_time", 
                    "article_type", "url", "category", "category_str", "sentiment_score", "sentiment_label", "image_ids"]
        
    sorted_news = news_data[column_names].sort_values(by="published_time", ascending=False).head(10)

    news_list = sorted_news.to_dict(orient='records')

    generate_image_urls(news_list)

    return {"news": news_list}
