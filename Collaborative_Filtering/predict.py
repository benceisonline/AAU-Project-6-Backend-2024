# FastAPI app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import uvicorn
from newsmf_model import NewsMF  # Assuming 'newsmf_model' is the correct filename

app = FastAPI()

# Indexize the user and item
news = pd.read_csv(
    "./MIND_small/news.tsv",
    sep="\t",
    names=["itemId", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
)

# Build index of items
ind2item = {idx + 1: itemid for idx, itemid in enumerate(news['itemId'].values)}
item2ind = {itemid: idx for idx, itemid in ind2item.items()}

# Indexize users
raw_behaviour = pd.read_csv(
    "./MIND_small/behaviors.tsv",
    sep="\t",
    names=["impressionId", "userId", "timestamp", "click_history", "impressions"]
)

unique_userIds = raw_behaviour['userId'].unique()
# Allocate a unique index for each user, but let the zeroth index be a UNK index:
ind2user = {idx + 1: itemid for idx, itemid in enumerate(unique_userIds)}
user2ind = {itemid: idx for idx, itemid in ind2user.items()}
print(f"We have {len(user2ind)} unique users in the dataset")

# Create a new column with userIdx:
raw_behaviour['userIdx'] = raw_behaviour['userId'].map(lambda x: user2ind.get(x, 0))

# Load the saved model
model = NewsMF(num_users=len(ind2user) + 1, num_items=len(ind2item) + 1)
model.load_state_dict(torch.load("Saved_Model/collaborative_filtering_model.pth"))
model.eval()

# Load news data for displaying recommendations
news_data = pd.read_csv(
    "./MIND_small/news.tsv",
    sep="\t",
    names=["itemId", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
)

# Pydantic model for request payload
class PredictionRequest(BaseModel):
    user_id: str  # Change the type to str

# Pydantic model for response payload
class PredictionResponse(BaseModel):
    user_id: str  # Change the type to str
    recommended_items: list

# Endpoint for making predictions
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    user_id = request.user_id  # Use user_id directly as a string
    user_idx = user2ind.get(user_id, 0)
    item_ids = list(ind2item.keys())

    # Ensure that the user index is valid
    if user_idx == 0:
        raise HTTPException(status_code=404, detail="User not found")

    # Forward pass to get predictions
    predictions = model.forward(torch.tensor([user_idx] * len(item_ids)), torch.tensor(item_ids))

    # Get top 10 predictions
    top_indices = torch.topk(predictions.flatten(), 10).indices
    top_item_ids = [ind2item[ix.item()] for ix in top_indices]

    # Filter for the top 10 suggested items
    recommended_items = news_data[news_data["itemId"].isin(top_item_ids)].to_dict(orient='records')

    return {"user_id": user_id, "recommended_items": recommended_items}

# Default route to handle requests to the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Default route to handle requests to favicon.ico
@app.get("/favicon.ico")
async def get_favicon():
    raise HTTPException(status_code=404, detail="Favicon not found")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
