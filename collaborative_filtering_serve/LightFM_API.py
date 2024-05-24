from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from api_utilities.lightfm_utilities_single_datafile import RecommenderSystem
from api_utilities.lightfm_news_utilities import NewsTools
import pandas as pd

app = FastAPI()

url = "172.30.253.127"

news_data = pd.read_parquet("./ebnerd_data/ebnerd_small/articles.parquet")
news_tools = NewsTools(news_data)

data_path = "exported_data/combined_data_small.csv"
models_folder_path = "saved_models/"
model_id = "lightfm_model_multi_file"

recommender_system = RecommenderSystem(data_path, models_folder_path)
recommender_system.load_data()
recommender_system.load_model(model_id);

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

class AllNewsRequest(BaseModel):
    start_index: int
    no_recommendations: int

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