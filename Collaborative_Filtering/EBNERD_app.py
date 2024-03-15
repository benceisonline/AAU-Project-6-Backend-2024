# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from EBNERD_utilities import load_data, initialize_model, make_predictions

app = FastAPI()

url = "172.20.10.4"

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
    user_id: int
    recommended_items: list

# Endpoint for making predictions
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return make_predictions(request, model, user2ind, ind2article, news_data)

# Default route to handle requests to the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host=url, port=8000) # Replace with your IP address
