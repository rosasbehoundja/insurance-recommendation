import sys
from pathlib import Path
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging  

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Configuration des chemins
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

# Import des fonctions depuis ton script
from scripts.preprocessing import preprocess, prepare_for_training, PRODUCT_COLUMNS

# 2. Initialisation de l'App
app = FastAPI(title="Zimnat Insurance Recommender")

# 3. Chargement du modèle complet (dict avec model, encoder, scaler)
MODEL_PATH = BASE_DIR / "weights" / "catboost.pkl"  

try:
    loaded_data = joblib.load(MODEL_PATH)
    model = loaded_data['model']  # Le modèle MultiOutput
    encoder = loaded_data['encoder']  # OneHotEncoder fitted
    scaler = loaded_data['scaler']  # StandardScaler fitted
    logger.info(f" Modèle, encoder et scaler chargés depuis {MODEL_PATH}")
except Exception as e:
    logger.error(f" Erreur de chargement : {e}")
    model = None
    encoder = None
    scaler = None

# 4. Schéma d'entrée
class ClientFeatures(BaseModel):
    birth_year: int
    join_date: str
    sex: str  
    marital_status: str
    branch_code: str
    occupation_code: str
    occupation_category_code: str

# 5. Endpoints
@app.get("/health")
def health_check():
    return {"status": "OK", "model_loaded": model is not None}

@app.post("/predict")
def predict_endpoint(features: ClientFeatures):
    if model is None or encoder is None or scaler is None:
        raise HTTPException(status_code=500, detail="Composants modèle non disponibles.")

    try:
        # A. Création du DataFrame
        input_df = pd.DataFrame([features.dict()])
        
        # Ajoute colonnes produits à 0
        for col in PRODUCT_COLUMNS:
            input_df[col] = 0
        
        # B. Nettoyage des données
        df_processed = preprocess(input_df.copy())

        # C. Préparation features (avec encoder/scaler chargés, pas de fit)
        X_final, _, _, _ = prepare_for_training(
            df_processed, 
            fit_encoders=False,  # Utilise les loaded
            encoder=encoder,
            scaler=scaler
        )
        logger.info(f"Shape X_final: {X_final.shape}")  # Debug

        # D. Prédiction
        predictions = model.predict_proba(X_final)  # Liste d'arrays par produit

        # E. Formatage
        result = {}
        for i, prod in enumerate(PRODUCT_COLUMNS):
            prob = float(predictions[i][0][1])  # Proba classe 1
            result[prod] = round(prob, 4)

        return {"status": "success", "predictions": result}

    except Exception as e:
        logger.error(f"Erreur technique : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# 6. Lancement
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)