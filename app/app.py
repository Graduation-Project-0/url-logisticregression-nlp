from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator, Field
from typing import List, Optional
import pickle
import numpy as np
from urllib.parse import urlparse
import uvicorn
from functools import lru_cache
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

class URLRequest(BaseModel):
    url: str = Field(..., description="URL to check for phishing", min_length=1)
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        return v.strip()

class BatchURLRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to check", min_length=1, max_length=100)
    
    @field_validator('urls')
    @classmethod
    def validate_urls(cls, v: List[str]) -> List[str]:
        if len(v) > 100:
            raise ValueError('Maximum 100 URLs per batch request')
        return [url.strip() for url in v if url.strip()]

class CleanURLResponse(BaseModel):
    success: bool
    original_url: str
    cleaned_url: str

class PredictionResponse(BaseModel):
    success: bool
    original_url: str
    cleaned_url: str
    prediction: str
    confidence: float
    probability_malicious: float
    probability_benign: float
    is_safe: bool
    risk_level: str
    recommendation: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    success: bool
    count: int
    total_processing_time_ms: float
    results: List[PredictionResponse]

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    original_url: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        print(f"Failed to start API: {e}")
        raise
    yield

app = FastAPI(
    title="Phishing Detection API",
    description="High-performance API for detecting phishing URLs using machine learning",
    version="2.0",
    docs_url="/docs",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

MODEL_PATH = r'N:\Vanguard\url-final-nlp\artifacts\phishing.pkl'
_model_cache = None

@lru_cache(maxsize=1)
def load_model():
    global _model_cache
    if _model_cache is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                _model_cache = pickle.load(f)
        except FileNotFoundError:
            print(f"Model file not found: {MODEL_PATH}")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return _model_cache

@lru_cache(maxsize=10000)
def clean_url(url: str) -> str:
    url = url.strip()
    
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    domain = parsed.netloc.split(':')[0]
    
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain

def get_risk_level(confidence: float, prediction: str) -> str:
    if prediction == 'benign':
        if confidence >= 90:
            return 'Very Low Risk'
        elif confidence >= 70:
            return 'Low Risk'
        else:
            return 'Moderate Risk'
    else:
        if confidence >= 90:
            return 'Critical Risk'
        elif confidence >= 70:
            return 'High Risk'
        else:
            return 'Moderate Risk'

def get_recommendation(prediction: str, confidence: float) -> str:
    if prediction == 'benign':
        if confidence >= 90:
            return 'This URL appears to be legitimate and safe to visit.'
        elif confidence >= 70:
            return 'This URL appears legitimate, but exercise normal caution.'
        else:
            return 'This URL may be legitimate, but verify before entering sensitive information.'
    else:
        if confidence >= 90:
            return 'DO NOT VISIT! This URL is highly likely to be a phishing attempt.'
        elif confidence >= 70:
            return 'WARNING: This URL is likely a phishing attempt. Avoid visiting.'
        else:
            return 'CAUTION: This URL may be a phishing attempt. Verify before visiting.'

def predict_url_fast(url: str, model) -> dict:
    start_time = time.perf_counter()
    
    original_url = url
    cleaned_url = clean_url(url)
    
    prediction = model.predict([cleaned_url])[0]
    probabilities = model.predict_proba([cleaned_url])[0]
    classes = list(model.classes_)
    
    # Get prediction index and confidence
    pred_idx = classes.index(prediction)
    confidence = probabilities[pred_idx] * 100
    
    # Handle different class naming conventions (bad/good, malicious/benign, 1/0, etc.)
    malicious_labels = ['malicious', 'bad', 'phishing', '1', 1]
    benign_labels = ['benign', 'good', 'legitimate', 'safe', '0', 0]
    
    malicious_idx = None
    benign_idx = None
    
    for i, cls in enumerate(classes):
        cls_lower = str(cls).lower()
        if cls in malicious_labels or cls_lower in [str(l).lower() for l in malicious_labels]:
            malicious_idx = i
        elif cls in benign_labels or cls_lower in [str(l).lower() for l in benign_labels]:
            benign_idx = i
    
    # Fallback: assume binary classification with index 0 and 1
    if malicious_idx is None or benign_idx is None:
        if len(classes) == 2:
            # Assume first class is benign/good, second is malicious/bad (common convention)
            benign_idx = 0
            malicious_idx = 1
        else:
            raise ValueError(f"Cannot determine class indices. Model classes: {classes}")
    
    # Normalize prediction label for response
    is_safe = pred_idx == benign_idx
    normalized_prediction = 'benign' if is_safe else 'malicious'
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    return {
        'success': True,
        'original_url': original_url,
        'cleaned_url': cleaned_url,
        'prediction': normalized_prediction,
        'confidence': round(confidence, 2),
        'probability_malicious': round(probabilities[malicious_idx] * 100, 2),
        'probability_benign': round(probabilities[benign_idx] * 100, 2),
        'is_safe': is_safe,
        'risk_level': get_risk_level(confidence, normalized_prediction),
        'recommendation': get_recommendation(normalized_prediction, confidence),
        'processing_time_ms': round(processing_time, 2)
    }

@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "Phishing Detection API",
        "version": "2.0",
        "performance": "High-performance with FastAPI + Uvicorn",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "POST /predict": "Predict single URL",
            "POST /predict_batch": "Predict multiple URLs (max 100)",
            "POST /clean": "Clean URL only",
            "GET /health": "Health check"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    try:
        model = load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "cache_info": clean_url.cache_info()._asdict()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: URLRequest):
    try:
        model = load_model()
        result = predict_url_fast(request.url, model)
        return result
    except FileNotFoundError as e:
        print(f"Model file error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model file not found: {MODEL_PATH}"
        )
    except ValueError as e:
        print(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Prediction error: {error_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )