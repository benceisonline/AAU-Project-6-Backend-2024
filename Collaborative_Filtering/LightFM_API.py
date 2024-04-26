# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from NewsItem import NewsItem
from LightFM_Utilities import RecommenderSystem
from LightFM_News_Utilities import NewsTools
import pandas as pd

app = FastAPI()

url = "172.20.10.4"

news_data = pd.read_parquet("./ebnerd_small/articles.parquet")
news_tools = NewsTools(news_data)

train_data_path = "exported_data/train_data.csv"
test_data_path = "exported_data/valid_data.csv"
model_path = "Saved_Model/lightfm_model_joblib.joblib"

recommender_system = RecommenderSystem(train_data_path, test_data_path, model_path)
recommender_system.load_data()
recommender_system.load_model()

# Pydantic model for request payload
class PredictionRequest(BaseModel):
    user_id: str
    start_index: int
    no_recommendations: int

# Pydantic model for response payload
class PredictionResponse(BaseModel):
    news: list[object]

# Endpoint for making predictions
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = recommender_system.make_predictions_for_user(request)
    news_tools.generate_image_urls(result['news'])
    return result

# Pydantic model for response payload
class AllNewsRequest(BaseModel):
    start_index: int
    no_recommendations: int

class AllNewsResponse(BaseModel):
    news: list[object]

@app.post("/all", response_model=AllNewsResponse)
def all_news(request: AllNewsRequest):
    return news_tools.get_newest_news(request)

# Default route to handle requests to the root endpoint
@app.get("/")
def read_root():
    return {"message": "You get used to it, I don't even see the code, all I see is blonde, brunette, redhead."}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host=url, port=8000) # Replace with your IP address
