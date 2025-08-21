"""
Configuration pour l'Assistant IA Réglementaire EY
"""

import os
from pathlib import Path

# Configuration des API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

# Configuration des modèles
DEFAULT_LLM_MODEL = "deepseek/deepseek-chat"  # Format correct pour OpenRouter
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADED_DATA_DIR = DATA_DIR / "uploaded"
MODELS_DIR = PROJECT_ROOT / "models"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"
INDICES_DIR = PROJECT_ROOT / "indices"

# Créer les dossiers s'ils n'existent pas
UPLOADED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR.mkdir(parents=True, exist_ok=True)

# Configuration des paramètres ML
DEFAULT_K_BEST = 15
DEFAULT_CORR_THRESHOLD = 0.95
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Configuration RAG
DEFAULT_TOP_K = 3
RAG_SIMILARITY_THRESHOLD = 0.6

# Langues supportées
SUPPORTED_LANGUAGES = {
    "fr": "Français",
    "en": "English", 
    "es": "Español",
    "ar": "العربية",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português"
}

# Utilisateurs simples pour l'authentification (à remplacer par une base réelle si besoin)
# IMPORTANT: Pour un environnement de prod, utilisez un stockage sécurisé + mots de passe hachés.
USERS = {
    "admin": {"password": "admin123", "name": "Administrateur"},
    "ey": {"password": "ey2024", "name": "EY Utilisateur"},
}

# Messages d'erreur par défaut
DEFAULT_RAG_CONTEXT = "Contexte réglementaire : Les réglementations Bâle III et Solvabilité II s'appliquent aux institutions financières pour la gestion des risques et la conformité."

def get_llm_config():
    """Retourne la configuration LLM"""
    return {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "model": DEFAULT_LLM_MODEL,
        "temperature": 0.3,
        "max_tokens": 100  # Réduit davantage pour rester sous les quotas gratuits
    }

def is_api_key_valid():
    """Vérifie si l'API key est valide"""
    return OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your-api-key-here" 