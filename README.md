# How to Run

## Step 1: Download Python, Anaconda, and Postman
Download Python from this [link](https://www.python.org/downloads) 🐍

Download Anaconda from this [link](https://docs.anaconda.com/free/anaconda/install/mac-os/) 🌳

Download Postman from this [link](https://www.postman.com/downloads/) 👷

## Step 1.5 Add Tensorboard Extension to Visual Studio Code (Optional)
You can add the extension directly inside VS Code, or by using this [link](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.tensorboard)

## Step 2: Create & Activate Conda Environment
```bash
conda create -n ml python
```
```bash
conda activate ml
```

## Step 3: Install Dependencies in the Environment
```bash
pip install -r requirements.txt
```

## Step 4: Local host the API
```bash
python app.py
```

## Step 5: Test API Endpoints 
Go to Postman GUI and make a new request for `0.0.0.0:8080`
- GET Request: `/{endpoint}`
- POST Request: `/{endpoint}?{parameter_name}={data}`
