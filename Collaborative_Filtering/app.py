# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from utilities import load_data, initialize_model, make_predictions

app = FastAPI()

# Load data
news_data, raw_behaviour_data, ind2item, item2ind, ind2user, user2ind = load_data()

# Initialize model
model = initialize_model(ind2user, ind2item)

# Pydantic model for request payload
class PredictionRequest(BaseModel):
    user_id: str

# Pydantic model for response payload
class PredictionResponse(BaseModel):
    user_id: str
    recommended_items: list

# Endpoint for making predictions
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return make_predictions(request, model, user2ind, ind2item, news_data)

# Default route to handle requests to the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Default route to handle requests to favicon.ico
@app.get("/favicon.ico")
async def get_favicon():
    raise HTTPException(status_code=404, detail="Favicon not found")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
