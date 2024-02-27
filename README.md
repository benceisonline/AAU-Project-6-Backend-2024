# How to Run

## 1. step: Download Anaconda
Download Anaconda from this link (https://docs.anaconda.com/free/miniconda/)](https://docs.anaconda.com/free/anaconda/install/mac-os/)

## 2. step: Create & Activate Conda Envoirment
```bash
conda create -n ml python=3.12.2
```
```bash
conda activate ml
```

## 3. step: Install Dependencies in Envoirment
```bash
pip install -r requirements.txt
```

## 4. step: Local host the API
```bash
python app.py
```

## 5. step: Test API Endpoints 
Go to Postman GUI and make a new request for 0.0.0.0:8080
- GET Request: /{endpoint}
- POST Request: /{endpoint}?{parameter_name}={data}
