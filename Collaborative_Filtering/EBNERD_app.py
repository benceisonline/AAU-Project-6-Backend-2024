# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from NewsItem import NewsItem
from EBNERD_utilities import load_data, initialize_model, make_predictions, get_newest_news

app = FastAPI()

url = "172.20.10.3"

# Load data
news_data, ind2article, article2ind, ind2user, user2ind = load_data()

# Initialize model
model = initialize_model(ind2user, ind2article)

# Pydantic model for request payload
class PredictionRequest(BaseModel):
    user_id: str
    no_recommendations: int

# Pydantic model for response payload
class PredictionResponse(BaseModel):
    news: list[NewsItem]

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

# Endpoint for getting news that the user is following (placeholder)
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
