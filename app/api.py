try:
 from fastapi import FastAPI,HTTPException
except ImportError as e: raise ImportError('API requires `pip install fastapi uvicorn`.') from e
from app.schemas import PredictionRequest,PredictionResponse
from app.inference import predict_text
app=FastAPI(title='Social Content Risk Research API',version='1.0.0')
@app.get('/health')
def health(): return {'status':'ok','disclaimer':'Research prototype; not a clinical diagnostic system.'}
@app.post('/predict',response_model=PredictionResponse)
def predict(payload:PredictionRequest):
 try:return predict_text(payload.text)
 except (ValueError,FileNotFoundError) as e:raise HTTPException(status_code=400 if isinstance(e,ValueError) else 503,detail=str(e))
