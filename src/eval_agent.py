from ml_pipeline import train_and_evaluate

def evaluate_models_for_target(target, models=["xgboost", "randomforest"]):
    """
    Évalue plusieurs modèles pour une cible donnée et recommande le meilleur.

    Args:
        target (str): La variable cible (ex: "LoanApproved" ou "RiskClass")
        models (list): Liste des noms de modèles à évaluer

    Returns:
        dict: Contenant les scores, le meilleur modèle et une recommandation
    """
    results = {}

    for model in models:
        print(f"🔍 Évaluation de {model} pour {target}")
        try:
            acc = train_and_evaluate(target, model)
            results[model] = acc
        except Exception as e:
            print(f"⚠️ Erreur avec {model}: {e}")
            results[model] = None

    # Filtrer les modèles qui ont bien marché
    filtered = {k: v for k, v in results.items() if v is not None}

    if not filtered:
        return {
            "target": target,
            "scores": results,
            "best_model": None,
            "recommendation": "Aucun modèle n'a pu être évalué avec succès."
        }

    best_model = max(filtered, key=filtered.get)

    return {
        "target": target,
        "scores": results,
        "best_model": best_model,
        "recommendation": f"Le meilleur modèle pour {target} est {best_model} avec un score de {filtered[best_model]:.4f}."
    }

# Exemple d'appel manuel (à retirer si intégré à une app)
if __name__ == "__main__":
    print(evaluate_models_for_target("LoanApproved"))
    print(evaluate_models_for_target("RiskClass"))
