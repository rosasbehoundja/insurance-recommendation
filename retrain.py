import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chemins et imports
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from main import run_full_pipeline  # Assume main.py est au même niveau

def automate_retraining(train_data_path: str, model_name: str = "catboost"):
    logger.info(" Lancement du ré-entraînement automatique...")
    
    try:
        # Appel au pipeline avec params
        run_full_pipeline(
            train_data_path=train_data_path,
            model_name=model_name
        )
        logger.info(" Modèle mis à jour !")
        
        # Optionnel : Versioning – rename le modèle avec timestamp si besoin
        # (Si ton pipeline sauvegarde 'catboost_model.pkl', tu peux le renommer ici)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        old_path = BASE_DIR / "weights" / f"{model_name}_model.pkl"
        new_path = BASE_DIR / "weights" / f"{model_name}_model_{timestamp}.pkl"
        if old_path.exists():
            old_path.rename(new_path)
            logger.info(f"Modèle versionné : {new_path}")
    
    except Exception as e:
        logger.error(f" Erreur lors du retraining : {str(e)}")
        raise  # Relève pour alerter le scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retraining automatique Zimnat")
    parser.add_argument("--train_data_path", default="data/Train.csv", help="Chemin des données d'entraînement")
    parser.add_argument("--model_name", default="catboost", help="Nom du modèle à entraîner")
    
    args = parser.parse_args()
    
    automate_retraining(args.train_data_path, args.model_name)