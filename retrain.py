import subprocess
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path


def log(message: str):
    """Affiche un message avec timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def retrain_models(model: str = "all"):
    """
    Lance le réentraînement des modèles.

    :param model: 'xgboost', 'deep_learning', ou 'all'
    :param data_path: Chemin vers les données d'entraînement
    """
    log(f"Début du réentraînement - Modèle: {model}")

    scripts_dir = Path(__file__).parent / "scripts"
    training_script = scripts_dir / "training.py"

    if not training_script.exists():
        log(f"ERREUR: Script d'entraînement non trouvé: {training_script}")
        return False

    try:
        # Exécuter le script de training
        cmd = [sys.executable, str(training_script)]

        log(f"Exécution: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(scripts_dir),
            capture_output=True,
            text=True
        )

        # Afficher la sortie
        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            log(f"ERREUR: Le réentraînement a échoué (code: {result.returncode})")
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            return False

        log("Réentraînement terminé avec succès!")
        return True

    except Exception as e:
        log(f"ERREUR: {e}")
        return False


def run_scheduled(interval_seconds: int, model: str = "all", stop_event=None):
    """
    Exécute le réentraînement de façon programmée.

    :param interval_seconds: Intervalle entre chaque réentraînement (en secondes)
    :param model: Modèle à réentraîner
    :param stop_event: threading.Event() pour arrêter la boucle
    """
    log(f"Mode programmé activé - Intervalle: {interval_seconds}s")

    iteration = 1
    while True:
        if stop_event and stop_event.is_set():
            log("Arrêt de la programmation demandé")
            break

        log(f"=== Itération {iteration} ===")
        retrain_models(model=model)

        log(f"Prochaine exécution dans {interval_seconds} secondes...")

        # Attendre avec possibilité d'interruption
        if stop_event:
            if stop_event.wait(timeout=interval_seconds):
                log("Arrêt de la programmation demandé")
                break
        else:
            time.sleep(interval_seconds)
        iteration += 1


def main():
    parser = argparse.ArgumentParser(
        description="Réentraînement automatique des modèles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python retrain.py                       # Réentraîne tous les modèles une fois
  python retrain.py --schedule 3600       # Réentraîne toutes les heures
  python retrain.py --schedule 86400      # Réentraîne tous les jours
        """
    )

    parser.add_argument(
        "--model",
        default="all",
        choices=["xgboost", "deep_learning", "all"],
        help="Modèle à réentraîner (default: all)"
    )

    parser.add_argument(
        "--schedule",
        type=int,
        default=None,
        help="Intervalle de réentraînement en secondes (mode programmé)"
    )

    args = parser.parse_args()

    if args.schedule:
        # Mode programmé
        try:
            run_scheduled(args.schedule, model=args.model)
        except KeyboardInterrupt:
            log("Arrêt du réentraînement programmé (Ctrl+C)")
    else:
        # Exécution unique
        success = retrain_models(model=args.model)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
