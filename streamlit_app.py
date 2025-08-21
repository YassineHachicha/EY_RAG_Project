import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import base64
import joblib

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent / "src"))

# Imports des modules du projet
# Remarque : Ces importations nécessitent que les fichiers correspondants (ml_pipeline.py, etc.) soient présents.
# Pour l'exemple, nous supposerons que ces fonctions existent. Si elles n'existent pas, le code lèvera une erreur.
# Vous devrez peut-être commenter ces lignes si vous n'avez pas ces fichiers.
from ml_pipeline import train_and_evaluate, predict_from_input, preprocess_data
from rag_multi_doc_generator import generate_answer
from agent_langchain_mcp import llm_rag, llm_report, generate_summary_structured
from config import USERS

# Vérification de la disponibilité du LLM
LLM_AVAILABLE = llm_rag is not None
LLM_REPORT_AVAILABLE = llm_report is not None
if not LLM_AVAILABLE:
    st.warning("⚠️ API LLM RAG non disponible. Les fonctionnalités RAG seront limitées.")
if not LLM_REPORT_AVAILABLE:
    st.warning("⚠️ API LLM Rapports non disponible. La génération de rapports sera limitée.")

# Configuration de la page
st.set_page_config(
    page_title="EY - Assistant AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins vers les images
data_dir = Path(__file__).parent / "data"
logo_path = data_dir / "EY logo.jpg"
ai_bg_path = data_dir / "AI.jpg"
logout_img_path = data_dir / "Log out.jpg"

# ─────────────────────────────
# Helpers UI
# ─────────────────────────────
def box_md(content: str):
    """Affiche un bloc de texte dans un petit cadre gris lisible."""
    st.markdown(f'<div class="text-box">{content}</div>', unsafe_allow_html=True)

def box_title(content: str, emoji: str = ""):
    """Titre dans un cadre léger."""
    st.markdown(
        f'<div class="text-title">{emoji} {content}</div>',
        unsafe_allow_html=True
    )

def file_to_base64(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return ""

def map_features_to_model(target: str, ui: dict) -> dict:
    """Convertit les champs UI en features attendues par le modèle."""
    if target == "LoanApproved":
        user = {}
        # Valeurs continues
        user["Age"] = ui.get("Age")
        user["AnnualIncome"] = ui.get("Income")
        user["LoanAmount"] = ui.get("LoanAmount")
        user["CreditScore"] = ui.get("CreditScore")
        employment_years = ui.get("EmploymentYears")
        if employment_years is not None:
            user["JobTenure"] = employment_years
            user["Experience"] = employment_years

        # One-hot EmploymentStatus
        emp = (ui.get("EmploymentStatus") or "").lower()
        user["EmploymentStatus_Self-Employed"] = 1 if emp in ["self-employed", "self employed"] else 0
        user["EmploymentStatus_Unemployed"] = 1 if emp == "unemployed" else 0

        # One-hot MaritalStatus
        mar = (ui.get("MaritalStatus") or "").lower()
        user["MaritalStatus_Single"] = 1 if mar == "single" else 0
        user["MaritalStatus_Married"] = 1 if mar == "married" else 0

        # One-hot HomeOwnershipStatus
        home = (ui.get("HomeOwnershipStatus") or "").lower()
        user["HomeOwnershipStatus_Own"] = 1 if home == "own" else 0
        user["HomeOwnershipStatus_Rent"] = 1 if home == "rent" else 0

        # One-hot LoanPurpose
        purpose = (ui.get("LoanPurpose") or "").lower()
        user["LoanPurpose_Education"] = 1 if purpose == "education" else 0
        user["LoanPurpose_Other"] = 1 if purpose not in ("education", "") else 0

        return user
    elif target == "RiskClass":
        expected = get_expected_features("RiskClass")
        user: dict = {}
        for key in ["Exposure", "PastDueDays", "UtilizationRatio"]:
            if key in expected and key in ui:
                user[key] = ui[key]

        categorical_fields = {
            "RiskLevel": ui.get("RiskLevel"),
            "Industry": ui.get("Industry"),
            "Region": ui.get("Region"),
        }
        for base, selected in categorical_fields.items():
            if not selected:
                continue
            for feat in expected:
                if feat.startswith(base + "_"):
                    value = feat.split("_", 1)[1]
                    user[feat] = 1 if str(selected) == value else 0
        return user
    return dict(ui)

def get_expected_features(target: str):
    try:
        feats = joblib.load(f"models/{target}/features.pkl")
        return feats.tolist() if hasattr(feats, "tolist") else list(feats)
    except Exception:
        return []

def _collect_one_hot_options(prefix: str, expected_features: list[str]) -> list[str]:
    options = []
    for name in expected_features:
        if name.startswith(prefix + "_"):
            options.append(name.split("_", 1)[1])
    return options

# Logo EY dans la sidebar
if logo_path.exists():
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] { padding-top: 8px !important; }
            [data-testid="stSidebarNav"] { padding-top: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.image(str(logo_path), width=110)

# Image de fond
if ai_bg_path.exists():
    try:
        with open(ai_bg_path, "rb") as _f:
            _bg_b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            f"""
            <style>
                .stApp {{ position: relative; }}
                .stApp::before {{
                    content: "";
                    position: fixed;
                    inset: 0;
                    background-image: url('data:image/jpg;base64,{_bg_b64}');
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    opacity: 0.2;
                    pointer-events: none;
                    z-index: 0;
                }}
                div[data-testid="stAppViewContainer"] {{
                    position: relative;
                    z-index: 1;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# CSS personnalisé
st.markdown("""
<style>
    html, body, .stApp {
        background-color: #000000 !important; /* Fond noir */
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    /* Contenu principal: fond semi-transparent */
    div[data-testid="stAppViewContainer"] .main .block-container {
        background-color: rgba(255,255,255,0.92) !important;
        border-radius: 10px;
        padding: 1.1rem 1.6rem;
        color: #000000 !important; /* Couleur de texte par défaut pour le conteneur principal */
        font-size: 1.05rem;
        line-height: 1.6;
    }
    /* S'assurer que les textes généraux dans le conteneur principal sont noirs */
    div[data-testid="stAppViewContainer"] .main .block-container p,
    div[data-testid="stAppViewContainer"] .main .block-container label,
    div[data-testid="stAppViewContainer"] .main .block-container span,
    div[data-testid="stAppViewContainer"] .main .block-container li,
    div[data-testid="stAppViewContainer"] .main .block-container a {
        color: #000000 !important;
    }

    /* Titre principal */
    .main-header {
        font-size: 2.4rem;
        color: #ffffff !important; /* Titre principal en blanc sur fond noir */
        text-align: center;
        margin: 0 0 1.2rem 0;
        font-weight: 800;
    }

    /* Cartes métriques / vignettes */
    .metric-card {
        background-color: #f5f7fb;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 0 rgba(0,0,0,0.02);
        color: #000000 !important; /* Texte en noir */
    }
    .metric-card * { color: #000000 !important; } /* Texte en noir pour tous les éléments enfants */

    /* Cadre gris pour le texte lisible */
    .text-box {
        background-color: #f5f5f5;
        padding: 12px 14px;
        border-radius: 10px;
        border: 1px solid #e1e1e1;
        margin: 0.6rem 0 1rem 0;
        color: #000000 !important; /* Texte en noir */
    }
    .text-box * { color: #000000 !important; } /* Texte en noir pour tous les éléments enfants */

    /* Titre encadré léger */
    .text-title {
        background: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 10px 14px;
        border-radius: 10px;
        font-weight: 700;
        margin: 0.6rem 0 0.8rem 0;
        color: #000000 !important; /* Texte en noir */
    }
    .text-title * { color: #000000 !important; } /* Texte en noir pour tous les éléments enfants */

    .success-message {
        background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 0.5rem; border: 1px solid #c3e6cb;
    }
    .info-message {
        background-color: #d1ecf1; color: #0c5460; padding: 1rem; border-radius: 0.5rem; border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

def render_header():
    left, right = st.columns([0.9, 0.1])
    with left:
        st.markdown('<h1 class="main-header">EY- Assistant AI</h1>', unsafe_allow_html=True)
    with right:
        st.write("")
        if logout_img_path.exists():
            img_b64 = file_to_base64(logout_img_path)
            btn = st.button(" ", key="logout_img_btn")
            st.markdown(
                f"""
                <style>
                .stButton button#logout_img_btn {{
                    background: transparent url('data:image/jpg;base64,{img_b64}') center/contain no-repeat;
                    width: 46px; height: 46px; border: none; cursor: pointer;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            if btn:
                st.session_state.auth = {"is_authenticated": False, "user": None}
                st.rerun()
        else:
            if st.button("🚪", key="logout_icon", help="Se déconnecter"):
                st.session_state.auth = {"is_authenticated": False, "user": None}
                st.rerun()

# ─────────────────────────────
# Authentification simple
# ─────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = {"is_authenticated": False, "user": None}

def login_view():
    render_header()
    box_title("Connexion", "🔐")
    with st.form("login_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Nom d'utilisateur", key="login_user")
        with col2:
            password = st.text_input("Mot de passe", type="password", key="login_pass")
        remember = st.checkbox("Se souvenir de moi", value=True)
        submitted = st.form_submit_button("Se connecter")
    if submitted:
        user = USERS.get(username)
        if user and password == user.get("password"):
            st.session_state.auth = {"is_authenticated": True, "user": {"username": username, "name": user.get("name", username), "remember": remember}}
            st.success("✅ Connexion réussie")
            st.rerun()
        else:
            st.error("❌ Identifiants invalides")

def logout_button():
    if st.sidebar.button("🚪 Se déconnecter"):
        st.session_state.auth = {"is_authenticated": False, "user": None}
        st.rerun()

if not st.session_state.auth["is_authenticated"]:
    login_view()
    st.stop()

render_header()

# Sidebar pour la navigation + info utilisateur
st.sidebar.markdown(f"**👤 {st.session_state.auth['user']['name']}**")
page = st.sidebar.selectbox("Choisissez une section :", ["🏠 Accueil", "🤖 Machine Learning", "📊 Prédictions", "📚 Recherche RAG", "⚙️ Pipeline Automatique"]) 

# Page d'accueil
if page == "🏠 Accueil":
    box_title("Bienvenue dans l'Assistant IA Réglementaire EY", "👋")
    box_md("Accédez rapidement à l’entraînement des modèles, à la recherche réglementaire (RAG) et aux prédictions en temps réel, le tout dans une interface unifiée.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Machine Learning</h3>
            <p>Entraînement et évaluation de modèles pour la prédiction de l'approbation de prêts et la classification des risques.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Recherche RAG</h3>
            <p>Recherche intelligente dans les documents réglementaires (Bâle III, Solvabilité II, etc.) avec Graph RAG.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Prédictions</h3>
            <p>Interface pour effectuer des prédictions en temps réel avec les modèles entraînés.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    box_title("📊 Statistiques du Projet")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modèles ML", "25+", "XGBoost, RandomForest, etc.")
    with col2:
        st.metric("Documents RAG", "4", "Bâle III, Solvabilité II, etc.")
    with col3:
        st.metric("Données", "2 Datasets", "LoanApproved & RiskClass")
    with col4:
        st.metric("Langues", "7", "FR, EN, ES, AR, DE, IT, PT")

# ... (Le reste du code reste inchangé, car les modifications étaient principalement dans le CSS et le titre)
# Page Machine Learning
elif page == "🤖 Machine Learning":
    box_title("🤖 Entraînement et Évaluation des Modèles")
    box_md("Configurez votre cible, le type de modèle et lancez l’entraînement. Activez l’optimisation des hyperparamètres pour de meilleures performances.")

    # Configuration du modèle
    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox(
            "Cible à prédire :",
            ["LoanApproved", "RiskClass"],
            help="LoanApproved : Prédiction d'approbation de prêt\nRiskClass : Classification des risques"
        )

        model_type = st.selectbox(
            "Type de modèle :",
            ["auto", "xgboost", "randomforest", "lightgbm", "logisticregression", "svc"],
            help="'auto' sélectionne automatiquement le meilleur modèle"
        )

    with col2:
        optimize = st.checkbox(
            "Optimisation des hyperparamètres",
            help="Active la recherche par grille pour optimiser les hyperparamètres"
        )

        feature_selection = st.selectbox(
            "Sélection de features :",
            ["none", "kbest", "rfe"],
            help="Méthode de sélection des variables les plus importantes"
        )

    # Paramètres avancés
    with st.expander("⚙️ Paramètres avancés"):
        col1, col2 = st.columns(2)

        with col1:
            k_best = st.slider("Nombre de features (k-best)", 5, 50, 15)
            corr_threshold = st.slider("Seuil de corrélation", 0.8, 0.99, 0.95, 0.01)

        with col2:
            test_size = st.slider("Taille du test set", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random state", 0, 1000, 42)

    # Bouton d'entraînement
    if st.button("🚀 Lancer l'entraînement", type="primary"):
        with st.spinner("Entraînement en cours..."):
            try:
                params = {
                    "target": target,
                    "model_type": model_type,
                    "optimize": optimize,
                    "feature_selection": feature_selection,
                    "k_best": k_best,
                    "corr_threshold": corr_threshold
                }
                result = train_and_evaluate(**params)
                st.success("✅ Entraînement terminé avec succès !")

                st.markdown("### 📊 Résultats de l'entraînement")
                if isinstance(result, str):
                    box_md(result)
                else:
                    st.json(result)

                st.session_state['last_training_params'] = params
                st.session_state['last_training_result'] = result

            except Exception as e:
                st.error(f"❌ Erreur lors de l'entraînement : {str(e)}")

    # Affichage des résultats précédents
    if 'last_training_result' in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Derniers résultats d'entraînement")
        if isinstance(st.session_state['last_training_result'], str):
            box_md(st.session_state['last_training_result'])
        else:
            st.json(st.session_state['last_training_result'])

# Page Prédictions
elif page == "📊 Prédictions":
    box_title("📊 Interface de Prédiction")
    box_md("Renseignez les features et obtenez une prédiction immédiate avec la probabilité associée.")

    # Sélection du modèle
    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox(
            "Cible :",
            ["LoanApproved", "RiskClass"],
            key="pred_target"
        )
        
        models_dir = os.path.join("models", target)
        available_models = []
        try:
            for f in os.listdir(models_dir):
                if f.endswith("_model.pkl"):
                    name = f.replace("_model.pkl", "")
                    available_models.append(name)
        except Exception:
            available_models = ["xgboost", "randomforest", "lightgbm", "logisticregression"]
        available_models = sorted(set(available_models))

        model_type = st.selectbox(
            "Modèle :",
            available_models,
            key="pred_model"
        )

    with col2:
        if target == "LoanApproved":
            st.markdown("### 📝 Saisie des features (LoanApproved)")
            col1b, col2b = st.columns(2)
            with col1b:
                age = st.number_input("Âge", 18, 100, 30)
                income = st.number_input("Revenu annuel", 10000, 500000, 50000)
                loan_amount = st.number_input("Montant du prêt", 1000, 1000000, 50000)
                credit_score = st.number_input("Score de crédit", 300, 850, 700)
                employment_years = st.number_input("Années d'emploi", 0, 50, 5)
            with col2b:
                education_level = st.selectbox("Niveau d'éducation", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
                employment_status = st.selectbox("Statut d'emploi", ["Employed", "Self-employed", "Unemployed"])
                marital_status = st.selectbox("Statut marital", ["Single", "Married", "Divorced"])
                home_ownership = st.selectbox("Propriété immobilière", ["Own", "Rent", "Mortgage"])
                loan_purpose = st.selectbox("Objectif du prêt", ["Home", "Car", "Education", "Business", "Personal"])

            features = {
                "Age": age, "Income": income, "LoanAmount": loan_amount,
                "CreditScore": credit_score, "EmploymentYears": employment_years,
                "EducationLevel": education_level, "EmploymentStatus": employment_status,
                "MaritalStatus": marital_status, "HomeOwnershipStatus": home_ownership,
                "LoanPurpose": loan_purpose
            }
        else:
            st.markdown("### 📝 Saisie des features (RiskClass)")
            exp_feats = get_expected_features("RiskClass")
            risk_level_opts = _collect_one_hot_options("RiskLevel", exp_feats) or ["Low", "Medium", "High"]
            industry_opts = _collect_one_hot_options("Industry", exp_feats) or ["Finance", "Tech", "Retail"]
            region_opts = _collect_one_hot_options("Region", exp_feats) or ["EU", "US", "APAC"]

            colA, colB = st.columns(2)
            with colA:
                exposure = st.number_input("Exposure", 0.0, 1e9, 100000.0)
                pdays = st.number_input("Past Due Days", 0, 360, 0)
                utilization = st.slider("Utilization Ratio", 0.0, 1.0, 0.35)
            with colB:
                risk_level = st.selectbox("Risk Level", risk_level_opts)
                industry = st.selectbox("Industry", industry_opts)
                region = st.selectbox("Region", region_opts)

            features = {
                "Exposure": exposure, "PastDueDays": pdays, "UtilizationRatio": utilization,
                "RiskLevel": risk_level, "Industry": industry, "Region": region,
            }

    if st.button("🔮 Effectuer la prédiction", type="primary"):
        with st.spinner("Prédiction en cours..."):
            try:
                user_features = map_features_to_model(target, features)
                prediction = predict_from_input(user_features, target=target, model_type=model_type)
                st.success("✅ Prédiction effectuée !")

                col1r, col2r = st.columns(2)
                with col1r:
                    st.markdown("### 📊 Résultat de la prédiction")
                    box_md(str(prediction))

                with col2r:
                    if target == "LoanApproved":
                        prob = prediction.get('probability', 0.5)
                        st.markdown("### 📈 Probabilité d'approbation")
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number", value = prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Probabilité (%)"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#ea4335"},
                                    {'range': [50, 75], 'color': "#fbbc05"},
                                    {'range': [75, 100], 'color': "#34a853"}
                                ],
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {str(e)}")

# Page Recherche RAG
elif page == "📚 Recherche RAG":
    box_title("📚 Recherche dans les Documents Réglementaires")
    box_md("Posez une question réglementaire. L’agent effectue une recherche augmentée (RAG) et synthétise la réponse.")

    question = st.text_area(
        "Posez votre question réglementaire :",
        placeholder="Ex: Quelle est la définition du capital Tier 1 dans Bâle III ?",
        height=120
    )
    st.markdown("---")
    st.markdown("**Exemples de questions :**")
    ex_q = [
        "Dans Bâle III, quelle est la définition du capital Tier 1 ?",
        "Quelle est la formule du ratio de levier selon Bâle III ?",
        "Comment calcule-t-on le SCR standard selon Solvabilité II ?",
    ]
    for q in ex_q:
        st.markdown(f"• {q}")

    if not LLM_AVAILABLE:
        st.error("❌ Fonctionnalité RAG non disponible.")
    elif st.button("🔍 Rechercher", type="primary") and question:
        with st.spinner("Recherche en cours..."):
            try:
                answer = generate_answer(question, llm=llm_rag, top_k=3)
                st.success("✅ Réponse trouvée !")
                st.markdown("### 📖 Réponse")
                box_md(answer.content if hasattr(answer, 'content') else str(answer))

                if 'rag_history' not in st.session_state:
                    st.session_state['rag_history'] = []
                st.session_state['rag_history'].append({
                    'question': question,
                    'answer': answer.content if hasattr(answer, 'content') else str(answer)
                })

            except Exception as e:
                st.error(f"❌ Erreur lors de la recherche : {str(e)}")

    if 'rag_history' in st.session_state and st.session_state['rag_history']:
        st.markdown("---")
        st.markdown("### 📚 Historique des recherches")
        for item in reversed(st.session_state['rag_history'][-5:]):
            with st.expander(f"Q: {item['question'][:50]}..."):
                box_md(f"**Question :** {item['question']}<br>**Réponse :** {item['answer']}")

# Page Pipeline Automatique
elif page == "⚙️ Pipeline Automatique":
    box_title("⚙️ Pipeline Automatique Complet")
    box_md("Uploadez un dataset, configurez les options et lancez un pipeline complet incluant l'entraînement, la recherche RAG et la génération de rapport.")

    st.markdown("### 📤 Étape 1 : Upload d'un Dataset")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV :", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Fichier '{uploaded_file.name}' uploadé.")
            st.dataframe(df.head())
            dataset_path = f"data/uploaded/{uploaded_file.name}"
            df.to_csv(dataset_path, index=False)
            st.session_state['uploaded_dataset'] = dataset_path
        except Exception as e:
            st.error(f"❌ Erreur de lecture du fichier : {e}")

    st.markdown("---")
    st.markdown("### ⚙️ Étape 2 : Configuration du Pipeline")

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Cible à prédire :", ["LoanApproved", "RiskClass"], key="pipe_target")
        optimize = st.checkbox("Optimisation des hyperparamètres", value=True, key="pipe_optimize")
    with col2:
        lang_options = ["Français","Anglais", "Espagnol", "Allemand", "Italien", "Portugais"]
        lang_label = st.selectbox("Langue du rapport :", lang_options, key="pipe_lang")
        lang_map = {
            "Français": "fr",
            "Anglais": "en",
            "Espagnol": "es",
            "Allemand": "de",
            "Italien": "it",
            "Portugais": "pt",
        }
        lang_code = lang_map.get(lang_label, "en")
        rag_question = st.text_input("Question RAG :", "Quelle est la définition du capital Tier 1 dans Bâle III ?", key="pipe_rag")

    if not LLM_AVAILABLE:
        st.error("❌ Pipeline non disponible sans API LLM.")
    elif st.button("🚀 Lancer le Pipeline Automatique", type="primary"):
        if 'uploaded_dataset' not in st.session_state:
            st.error("❌ Veuillez d'abord uploader un dataset.")
        else:
            with st.spinner("Pipeline automatique en cours..."):
                try:
                    # 1. Entraînement
                    st.info("Étape 1/3 : Entraînement du modèle...")
                    result_train = train_and_evaluate(
                        target=target, model_type="auto", optimize=optimize,
                        custom_dataset_path=st.session_state['uploaded_dataset']
                    )
                    st.success("✅ Entraînement terminé.")

                    # 2. RAG
                    st.info("Étape 2/3 : Recherche réglementaire...")
                    result_rag = generate_answer(rag_question, llm=llm_rag, top_k=1)
                    st.success("✅ Recherche RAG terminée.")
                    
                    # 3. Rapport
                    st.info("Étape 3/3 : Génération du rapport...")
                    report = generate_summary_structured(
                        ml_summary=str(result_train),
                        regulatory_summary=result_rag.content if hasattr(result_rag, 'content') else str(result_rag),
                        lang=lang_code
                    )
                    st.success("✅ Rapport généré.")
                    
                    st.markdown("---")
                    st.markdown("## 📋 Rapport Final")
                    report_content = report.content if hasattr(report, 'content') else str(report)
                    box_md(report_content)
                    st.download_button("💾 Télécharger le rapport", report_content, f"rapport_{target}_{lang_code}.txt")

                except Exception as e:
                    st.error(f"❌ Erreur lors du pipeline : {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>EY - Assistant AI | Développé avec Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)