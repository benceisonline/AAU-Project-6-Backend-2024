from fastapi import FastAPI
import uvicorn
from Model.model_prediction import test_function

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_route(user_id):
    try:
        user_id = test_function(user_id)
        return {"user_id": f"this is the user_id = {user_id}"}
    except Exception as e:
        raise e
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)