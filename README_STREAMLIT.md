# 🏦 Interface Streamlit - Assistant IA Réglementaire EY

## 📋 Vue d'ensemble

Cette interface Streamlit offre une interface utilisateur moderne et intuitive pour accéder à toutes les fonctionnalités du projet EY :

- 🤖 **Machine Learning** : Entraînement et évaluation de modèles
- 📊 **Prédictions** : Interface de prédiction en temps réel
- 📚 **Recherche RAG** : Recherche dans les documents réglementaires
- 📈 **Rapports** : Génération de rapports multilingues
- ⚙️ **Pipeline Automatique** : Exécution complète du workflow

## 🚀 Installation et Lancement

### Prérequis
- Python 3.8+
- Environnement virtuel activé
- Toutes les dépendances installées

### Méthode 1 : Script de lancement automatique
```bash
python run_streamlit.py
```

### Méthode 2 : Lancement manuel
```bash
# Activer l'environnement virtuel
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Lancer Streamlit
streamlit run streamlit_app.py
```

L'application sera accessible sur : **http://localhost:8501**

## 🎯 Fonctionnalités Principales

### 🏠 Page d'Accueil
- Vue d'ensemble du projet
- Statistiques des modèles et données
- Navigation vers toutes les sections

### 🤖 Machine Learning
**Configuration du modèle :**
- Choix de la cible (LoanApproved/RiskClass)
- Sélection du type de modèle (auto, XGBoost, RandomForest, etc.)
- Optimisation des hyperparamètres
- Sélection de features (k-best, RFE)

**Paramètres avancés :**
- Nombre de features (k-best)
- Seuil de corrélation
- Taille du test set
- Random state

### 📊 Prédictions
**Interface intuitive pour :**
- Saisie des features (âge, revenu, score de crédit, etc.)
- Sélection du modèle et de la cible
- Affichage des résultats avec graphiques
- Gauge chart pour les probabilités

### 📚 Recherche RAG
**Recherche dans les documents réglementaires :**
- Questions en langage naturel
- Réponses basées sur Graph RAG
- Historique des recherches
- Exemples de questions prédéfinies

**Documents disponibles :**
- Bâle III (définitions)
- Solvabilité II (directive UE)
- Rémunération MRT 2019
- Politique de rémunération des mandataires

### 📈 Rapports
**Génération de rapports multilingues :**
- 7 langues supportées (FR, EN, ES, AR, DE, IT, PT)
- Types de rapports : Exécutif, Technique, Réglementaire, Complet
- Téléchargement des rapports générés

### ⚙️ Pipeline Automatique
**Exécution complète du workflow :**
1. Entraînement automatique du meilleur modèle
2. Recherche d'informations réglementaires
3. Génération d'un rapport exécutif multilingue

## 🎨 Interface Utilisateur

### Design
- Interface moderne et responsive
- Navigation par sidebar
- Cartes d'information stylisées
- Graphiques interactifs avec Plotly
- Messages de statut colorés

### Expérience Utilisateur
- Feedback visuel en temps réel
- Gestion des erreurs avec messages explicites
- Sauvegarde de l'état de session
- Historique des actions

## 🔧 Configuration

### Variables d'environnement
L'application utilise les mêmes configurations que le projet principal :
- API OpenRouter pour le LLM
- Modèles d'embedding SentenceTransformers
- Index FAISS pour la recherche

### Personnalisation
Vous pouvez modifier :
- Les couleurs dans le CSS personnalisé
- Les paramètres par défaut
- Les exemples de questions RAG
- Les langues disponibles

## 📊 Visualisations

### Graphiques disponibles
- **Gauge charts** : Probabilités de prédiction
- **Métriques** : Statistiques du projet
- **Graphiques Plotly** : Visualisations interactives
- **Indicateurs** : Statut des opérations

## 🛠️ Développement

### Structure du code
```
streamlit_app.py          # Application principale
run_streamlit.py          # Script de lancement
README_STREAMLIT.md       # Ce guide
```

### Ajout de nouvelles fonctionnalités
1. Ajouter la nouvelle page dans la navigation
2. Créer la section correspondante dans le code
3. Intégrer avec les modules existants
4. Tester l'interface

### Débogage
- Utiliser `st.write()` pour afficher des variables
- Vérifier les logs Streamlit dans le terminal
- Tester les modules individuellement

## 🚨 Dépannage

### Problèmes courants

**Erreur de module non trouvé :**
```bash
pip install -r requirements.txt
```

**Erreur de port déjà utilisé :**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Problème de chemin :**
Vérifier que vous êtes dans le répertoire racine du projet.

### Logs et Debug
- Les erreurs s'affichent dans l'interface
- Logs détaillés dans le terminal
- Utiliser `st.exception()` pour capturer les erreurs

## 📈 Performance

### Optimisations
- Chargement lazy des modules
- Mise en cache des résultats
- Session state pour éviter les recalculs
- Gestion efficace de la mémoire

### Monitoring
- Temps de réponse des opérations
- Utilisation mémoire
- Performance des modèles ML

## 🔐 Sécurité

### Bonnes pratiques
- Validation des entrées utilisateur
- Gestion sécurisée des API keys
- Sanitisation des données
- Logs d'audit

## 📞 Support

Pour toute question ou problème :
1. Vérifier ce guide
2. Consulter la documentation Streamlit
3. Examiner les logs d'erreur
4. Tester les modules individuellement

---

**🏦 EY - Assistant IA Réglementaire | Interface Streamlit** 