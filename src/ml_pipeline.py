from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import os
import joblib

def preprocess_data(target):
    if target == "LoanApproved":
        df = pd.read_csv("data/preprocessed_loanapproved.csv")
        target_col = "LoanApproved"
    elif target == "RiskClass":
        df = pd.read_csv("data/preprocessed_riskclass.csv")
        target_col = "RiskClassEncoded"
    else:
        raise ValueError(f"Target inconnue : {target}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), X.columns

def train_and_evaluate(target: str, model_type: str = "auto") -> str:
    (X_train, X_test, y_train, y_test), feature_cols = preprocess_data(target)

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_type = model_type.lower()
    models = {
        "randomforest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "lightgbm": LGBMClassifier()
    }

    if model_type != "auto":
        if model_type not in models:
            raise ValueError(f"Modèle non supporté : {model_type}")
        model = models[model_type]
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    else:
        best_model = None
        best_score = 0
        best_name = ""
        for name, model in models.items():
            scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="roc_auc")
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_name = name
        model = best_model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        model_type = best_name.lower()

    # Sauvegarde modèle + scaler
    model_dir = f"models/{target}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/{model_type}_model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump(list(feature_cols), f"{model_dir}/features.pkl")

    return f"""
✅ Meilleur modèle : {model_type.upper()}
📊 Score AUC test : {round(auc, 3)}
🎯 Précision : {round(report['weighted avg']['precision'], 3)}
🔁 Recall : {round(report['weighted avg']['recall'], 3)}
🎯 F1-score : {round(report['weighted avg']['f1-score'], 3)}
"""

def predict_from_input(user_dict, target="LoanApproved", model_type="auto"):
    import os

    model_dir = f"models/{target}"
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    features_path = os.path.join(model_dir, "features.pkl")

    if model_type == "auto":
        # charger premier modèle dispo
        for f in os.listdir(model_dir):
            if f.endswith("_model.pkl"):
                model_path = os.path.join(model_dir, f)
                break
        else:
            raise FileNotFoundError("Aucun modèle trouvé.")
    else:
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pkl")

    # Chargement
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)

    # Préparation des données utilisateur
    X_input = pd.DataFrame([user_dict])
    X_input = X_input.reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)

    if target == "LoanApproved":
        return "Approved" if prediction[0] == 1 else "Rejected"
    elif target == "RiskClass":
        label_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        return label_map.get(prediction[0], prediction[0])
    else:
        return prediction[0]
