# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from typing import Optional
from EBNERD_utilities import load_data, initialize_model, make_predictions, get_newest_news

app = FastAPI()

url = "172.20.10.4"

# Load data
news_data, ind2article, article2ind, ind2user, user2ind = load_data()

# Initialize model
model = initialize_model(ind2user, ind2article)

class NewsItem(BaseModel):
    article_id: int
    title: str
    subtitle: str
    last_modified_time: datetime
    premium: bool
    body: str
    published_time: datetime
    image_ids: Optional[list[int]]
    article_type: str
    url: str
    entity_groups: list[str]
    topics: list[str]
    category: int
    subcategory: list[int]
    category_str: str
    total_inviews: Optional[float]
    total_pageviews: Optional[float]
    total_read_time: Optional[float]
    sentiment_score: float
    sentiment_label: str

# Pydantic model for request payload
class PredictionRequest(BaseModel):
    user_id: str
    no_recommendations: int

# Pydantic model for response payload
class PredictionResponse(BaseModel):
    recommended_items: list

# Endpoint for making predictions
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return make_predictions(request, model, user2ind, ind2article, news_data)

# Pydantic model for request payload
class FollowingRequest(BaseModel):
    user_id: int

# Pydantic model for response payload
class FollowingResponse(BaseModel):
    following_news: list

@app.get("/following", response_model=FollowingResponse)
def following(request: FollowingRequest):
    return {"following_news": ["news1", "news2", "news3"]}

# Pydantic model for response payload
class AllNewsResponse(BaseModel):
    news: list[NewsItem]

@app.get("/all", response_model=AllNewsResponse)
def all_news():
    return get_newest_news(news_data)


# Default route to handle requests to the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host=url, port=8000) # Replace with your IP address
