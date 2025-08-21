#!/usr/bin/env python3
"""
Script de lancement pour l'application Streamlit EY
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Lance l'application Streamlit"""
    
    # Vérifier que nous sommes dans le bon répertoire
    if not Path("streamlit_app.py").exists():
        print("❌ Erreur : streamlit_app.py non trouvé dans le répertoire actuel")
        print("Assurez-vous d'être dans le répertoire racine du projet EY")
        sys.exit(1)
    
    # Vérifier que l'environnement virtuel existe
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Erreur : Environnement virtuel 'venv' non trouvé")
        print("Veuillez créer l'environnement virtuel avec : python -m venv venv")
        sys.exit(1)
    
    # Déterminer le chemin Python de l'environnement virtuel
    if os.name == 'nt':  # Windows
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/Mac
        python_path = venv_path / "bin" / "python"
    
    if not python_path.exists():
        print(f"❌ Erreur : Python non trouvé dans {python_path}")
        sys.exit(1)
    
    # Lancer Streamlit
    print("🚀 Lancement de l'application Streamlit EY...")
    print(f"📍 Python utilisé : {python_path}")
    print("🌐 L'application sera accessible sur : http://localhost:8501")
    print("⏹️  Appuyez sur Ctrl+C pour arrêter l'application")
    print("-" * 50)
    
    try:
        # Lancer Streamlit avec l'environnement virtuel
        subprocess.run([
            str(python_path), "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée par l'utilisateur")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du lancement : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 