"""
⚽ FOOTBALL PREDICTION - OPENAPI/REST SERVER
Antrenează și fă predicții prin HTTP API
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging

from routers import prediction, training, data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("football-api")

# ──────────────────────────────────────────────
# 🔧 CONFIGURATIONS
# ──────────────────────────────────────────────

app = FastAPI(
    title="⚽ Football Prediction ML API",
    description="API pentru antrenarea și predicția meciurilor de fotbal cu XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ──────────────────────────────────────────────
# 📦 ROUTERS
# ──────────────────────────────────────────────

app.include_router(prediction.router)
app.include_router(training.router)
app.include_router(data.router)


# ──────────────────────────────────────────────
# 📊 PYDANTIC MODELS
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    api_version: str


# ──────────────────────────────────────────────
# 🔧 HELPER FUNCTIONS
# ──────────────────────────────────────────────

def load_models():
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return rf_model, xgb_model, feature_columns
    except Exception:
        return None, None, None


# ──────────────────────────────────────────────
# 🚀 API ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifică dacă serverul și modelele sunt disponibile"""
    rf_model, xgb_model, features = load_models()
    models_loaded = rf_model is not None and xgb_model is not None
    logger.info("Health check - models_loaded=%s", models_loaded)

    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "api_version": "1.0.0"
    }


@app.get("/", tags=["Info"])
async def root():
    """Informații despre API"""
    return {
        "name": "⚽ Football Prediction ML API",
        "version": "1.0.0",
        "description": "API pentru antrenarea și predicția meciurilor de fotbal",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)",
            "train": "/train (POST)",
            "training_status": "/train/status (GET)",
            "data_import": "/data/import (POST)",
            "data_stats": "/data/stats (GET)",
            "data_matches": "/data/matches (POST)",
            "data_elo": "/data/elo (POST)",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        }
    }


# ──────────────────────────────────────────────
# 🚀 MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Football Prediction API Server")
    print("\n" + "="*60)
    print("⚽ FOOTBALL PREDICTION - API SERVER")
    print("="*60)
    print("\n🚀 Porn serverul pe http://localhost:7777")
    print("\n📚 Documentație API:")
    print("   Swagger UI:  http://localhost:7777/docs")
    print("   ReDoc:       http://localhost:7777/redoc")
    print("\n🔗 Endpoints:")
    print("   GET  /health          - Verifică status")
    print("   POST /predict         - Face o predicție")
    print("   POST /train           - Antrenează modelele")
    print("   GET  /train/status    - Status antrenament")
    print("   POST /data/import     - Importă dataset în MongoDB")
    print("   GET  /data/stats      - Statistici colecții MongoDB")
    print("   GET  /data/matches    - Interoghează meciuri din MongoDB")
    print("   GET  /data/elo        - Interoghează ratinguri Elo din MongoDB")
    print("\n" + "="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=7777)

