import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging  

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Configuration du chemin pour accéder à tes dossiers 'scripts' et 'weights'
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

# Import de tes fonctions de nettoyage (indispensable pour que le modèle comprenne les données)
from scripts.preprocessing import preprocess

# 2. Initialisation de l'App (Doit être globale pour Uvicorn)
app = FastAPI(title="Zimnat Insurance Recommender")

# 3. Chargement du modèle au démarrage
# Note : Adapte le chemin vers ton fichier .pkl ou .joblib réel
MODEL_PATH = BASE_DIR / "weights" / "catboost_model.pkl" 

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f" Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    logger.error(f" Erreur de chargement du modèle : {e}")
    model = None

# 4. Schéma d'entrée (Features du client)
class ClientFeatures(BaseModel):
    age: int
    sex: str  # 'M' ou 'F'
    marital_status: str
    birth_year: int
    branch_code: str
    occupation_code: str
    occupation_category_code: str
    join_year: int
    # Ajoute ici toute autre colonne nécessaire à ton script 'preprocess' (ex. : si tu as des features engineered)

# 5. Endpoints
@app.get("/health")
def health_check():
    return {
        "status": "OK", 
        "model_loaded": model is not None,
        "python_version": sys.version
    }

@app.post("/predict")
def predict_endpoint(features: ClientFeatures):
    if model is None:
        logger.error("Modèle non disponible")
        raise HTTPException(status_code=500, detail="Modèle non disponible sur le serveur.")

    try:
        # A. Conversion de l'entrée JSON en DataFrame
        input_df = pd.DataFrame([features.dict()])
        
        # B. Prétraitement (Calcul des colonnes calculées, encodage, etc.)
        # On utilise ta fonction officielle pour être cohérent avec l'entraînement
        df_processed = preprocess(input_df.copy())
        
        # Validation simple post-prétraitement (ex. : pas de NaN)
        if df_processed.isnull().values.any():
            raise ValueError("Données incomplètes après prétraitement")

        # C. Prédiction des probabilités
        # predictions sera une liste d'arrays (une par produit) – assume multi-output
        predictions = model.predict_proba(df_processed)

        # D. Formatage du résultat {produit: probabilité}
        products = [
            'P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 
            'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 
            'J9JW', 'GHYX', 'ECY3'
        ]
        
        # On récupère la probabilité de la classe 1 (souscription) pour chaque produit
        # Note : Si ton modèle renvoie déjà des probas formatées différemment, adapte ici
        result = {prod: float(pred[0][1]) if isinstance(pred, list) else float(pred[1]) 
                  for prod, pred in zip(products, predictions)}

        logger.info("Prédiction réussie")
        return {
            "status": "success",
            "predictions": result
        }

    except ValueError as ve:
        logger.error(f"Erreur de validation : {ve}")
        raise HTTPException(status_code=400, detail=f"Données invalides : {str(ve)}")
    except Exception as e:
        logger.error(f"Erreur d'inférence : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {str(e)}")

# 6. Lancement
if __name__ == "__main__":
    # On utilise la chaîne "app:app" pour permettre le reload automatique
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)