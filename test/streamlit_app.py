import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="EY ML Agent Demo", layout="centered")

# --- Titre
st.title("🤖 Démo Streamlit : Agent ML avec SHAP")

# --- Chargement des données
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y, iris.target_names

X, y, target_names = load_data()
st.write("### Aperçu des données")
st.dataframe(X.head())

# --- Sélection utilisateur
selected_index = st.slider("Choisis un échantillon à prédire :", 0, len(X) - 1, 0)
sample = X.iloc[[selected_index]]

# --- Modèle
model = RandomForestClassifier()
model.fit(X, y)
prediction = model.predict(sample)[0]

st.write(f"### 🔍 Prédiction : {target_names[prediction]}")

# --- Interprétation SHAP
explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)


st.write("### 📊 Interprétation SHAP")
st.write(f"shap_values type: {type(shap_values)}")
st.write(f"shap_values shape: {getattr(shap_values, 'shape', None)}")
st.write(f"shap_values[0] type: {type(shap_values[0])}")
st.write(f"shap_values[0] shape: {getattr(shap_values[0], 'shape', None)}")
try:
    st.write(f"shap_values[0,0] type: {type(shap_values[0,0])}")
    st.write(f"shap_values[0,0] shape: {getattr(shap_values[0,0], 'shape', None)}")
except Exception as e:
    st.write(f"Erreur d'accès shap_values[0,0]: {e}")
fig, ax = plt.subplots()
try:
    classe_predite = prediction
    shap.plots.waterfall(shap_values[selected_index, classe_predite], show=False)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Erreur lors de l'affichage SHAP : {e}")
