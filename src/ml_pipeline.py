from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import os
import joblib
import json
import shap
import matplotlib.pyplot as plt
import webbrowser
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
import csv
import random
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, RidgeClassifierCV,
    SGDClassifier, PassiveAggressiveClassifier
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

# optionnels si installés
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier








def remove_highly_correlated_features(X, threshold=0.95):
    import numpy as np
    import pandas as pd
    X_numeric = X.select_dtypes(include=["int64", "float64"])
    corr_matrix = X_numeric.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    print(f"📉 Variables supprimées pour corrélation > {threshold} : {to_drop}")
    return X.drop(columns=to_drop, errors='ignore'), to_drop

def preprocess_data(target, feature_selection="none", k_best=15, corr_threshold=0.95, custom_dataset_path=None):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from sklearn.ensemble import RandomForestClassifier

    # Si un dataset personnalisé est fourni, l'utiliser
    if custom_dataset_path and os.path.exists(custom_dataset_path):
        print(f"📁 Utilisation du dataset uploadé : {custom_dataset_path}")
        df = pd.read_csv(custom_dataset_path)
        
        # Vérifier si la colonne cible existe
        if target not in df.columns:
            # Essayer de trouver une colonne similaire
            possible_targets = [col for col in df.columns if target.lower() in col.lower()]
            if possible_targets:
                target = possible_targets[0]
                print(f"🎯 Colonne cible trouvée : {target}")
            else:
                raise ValueError(f"Colonne cible '{target}' non trouvée dans le dataset. Colonnes disponibles : {list(df.columns)}")
        
        y = df[target]
        X = df.drop(columns=[target])
        
        # Encodage automatique des variables catégorielles
        categorical_cols = X.select_dtypes(include=['object', 'string']).columns
        if len(categorical_cols) > 0:
            print(f"🔤 Encodage de {len(categorical_cols)} variables catégorielles : {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Gestion des valeurs manquantes
        if X.isnull().sum().sum() > 0:
            print("🧹 Nettoyage des valeurs manquantes...")
            X = X.fillna(X.median())
        
        # Winsorizing pour les variables numériques
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            print(f"🔢 Colonnes numériques détectées : {list(numeric_cols)}")
            Q1 = X[numeric_cols].quantile(0.25)
            Q3 = X[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X[numeric_cols] = X[numeric_cols].clip(lower=lower, upper=upper, axis=1)
        else:
            print("⚠️ Aucune colonne numérique trouvée dans le dataset uploadé.")

    elif target == "LoanApproved":
        df = pd.read_csv("data/P1M2_Yonathan_Anggraiwan.csv", sep=';')
        df = df.drop(columns=['ApplicationDate'])
        df['EducationLevel'] = df['EducationLevel'].astype(str).str.strip()
        edu_map = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
        df['EducationLevel'] = df['EducationLevel'].map(edu_map)
        y = df["LoanApproved"]
        df = df.drop(columns=["LoanApproved"])
        df_encoded = pd.get_dummies(df, columns=[
            'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'
        ], drop_first=True)
        # Winsorizing
        Q1 = df_encoded.select_dtypes(include=["int64", "float64"]).quantile(0.25)
        Q3 = df_encoded.select_dtypes(include=["int64", "float64"]).quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_encoded[Q1.index] = df_encoded[Q1.index].clip(lower=lower, upper=upper, axis=1)
        X = df_encoded

    elif target == "RiskClass":
        df = pd.read_csv("data/preprocessed_riskclass.csv")
        X = df.drop(columns=["RiskClassEncoded"])
        y = df["RiskClassEncoded"]
    else:
        raise ValueError(f"Target inconnue : {target}")

    # 🔁 Supprimer corrélations
    X, dropped_corr = remove_highly_correlated_features(X, threshold=corr_threshold)

    # 🔎 Feature selection
    if feature_selection == "kbest":
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            selector = SelectKBest(score_func=f_classif, k=min(k_best, len(numeric_cols)))
            X_new = selector.fit_transform(X[numeric_cols], y)
            selected_cols = numeric_cols[selector.get_support()]
            print(f"✅ SelectKBest : {len(selected_cols)} variables retenues : {list(selected_cols)}")
            X = pd.concat([X[selected_cols], X.select_dtypes(exclude=["int64", "float64"])], axis=1)
        else:
            print("⚠️ Aucune colonne numérique trouvée pour la sélection de features. Feature selection ignorée.")

    elif feature_selection == "rfe":
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k_best, len(numeric_cols)))
            X_new = selector.fit_transform(X[numeric_cols], y)
            selected_cols = numeric_cols[selector.get_support()]
            print(f"✅ RFE : {len(selected_cols)} variables retenues : {list(selected_cols)}")
            X = pd.concat([X[selected_cols], X.select_dtypes(exclude=["int64", "float64"])], axis=1)
        else:
            print("⚠️ Aucune colonne numérique trouvée pour la sélection de features. Feature selection ignorée.")

    else:
        print("ℹ️ Feature selection désactivée.")

    # Vérification finale
    if X.shape[1] == 0:
        raise ValueError("❌ Aucune feature restante après prétraitement. Vérifiez votre dataset.")
    
    print(f"✅ Dataset final : {X.shape[0]} lignes, {X.shape[1]} colonnes")
    print(f"📋 Colonnes finales : {list(X.columns)}")
    
    # Split final
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    return (X_train, X_test, y_train, y_test), X.columns







def optimize_models(X_train, y_train, X_test, y_test, csv_log_path="benchmark_results1.csv"):
    results = []
    rows_for_csv = []

    print("\n📌 🔍 Modèles générés et testés :\n")

    n_classes = len(np.unique(y_train))
    scoring = 'roc_auc_ovr' if n_classes > 2 else 'roc_auc'

    for name, model in get_all_classifiers():
        param_grid = get_param_grid(name)
        if not param_grid:
            continue  # skip modèles sans grille

        try:
            print(f"🧠 {name}")

            grid = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            probas = getattr(best_model, "predict_proba", None)

            if callable(probas):
                probas = best_model.predict_proba(X_test)
                if n_classes > 2:
                    auc = roc_auc_score(y_test, probas, multi_class="ovr")
                else:
                    auc = roc_auc_score(y_test, probas[:, 1] if probas.shape[1] == 2 else probas)
            else:
                auc = roc_auc_score(y_test, best_model.decision_function(X_test), multi_class="ovr" if n_classes > 2 else 'raise')

            report = classification_report(y_test, y_pred, output_dict=True)

            results.append({
                'Model': name,
                'AUC': auc,
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-score': report['weighted avg']['f1-score'],
                'TrainedModel': best_model,
                'Report': report,
                'Best Params': grid.best_params_
            })

            rows_for_csv.append({
                'Model': name,
                'AUC': round(auc, 4),
                'Precision': round(report['weighted avg']['precision'], 4),
                'Recall': round(report['weighted avg']['recall'], 4),
                'F1-score': round(report['weighted avg']['f1-score'], 4),
                'Best Params': str(grid.best_params_)
            })

        except Exception as e:
            print(f"⚠️ Erreur avec {name} : {e}")
            continue
    # 🔁 Réinitialiser le fichier CSV à chaque exécution
    os.makedirs("reports", exist_ok=True)
    csv_path = os.path.join("reports", csv_log_path)

    # Réinitialiser le fichier dès le début (avec en-têtes uniquement)
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "AUC", "Precision", "Recall", "F1-score", "Best Params"])
        writer.writeheader()
        
    # 🧾 Export CSV
    if rows_for_csv:
        os.makedirs("reports", exist_ok=True)
        csv_path = os.path.join("reports", csv_log_path)
        df = pd.DataFrame(rows_for_csv)
        df.to_csv(csv_path, index=False)
        print(f"\n📁 Résultats enregistrés dans : {csv_path}\n")

    if not results:
        raise ValueError("Aucun modèle n'a pu être évalué. Vérifiez vos données ou la logique de scoring.")

    return results

def get_all_classifiers():
    excluded = {"calibratedclassifiercv", "categoricalnb", "complementnb","mlpclassifier","svc","logisticregression"}
    classifiers = []
    for name, Clf in all_estimators(type_filter='classifier'):
        try:
            model = Clf()
            if isinstance(model, ClassifierMixin):
                classifiers.append((name.lower(), model))
        except:
            continue

    random.shuffle(classifiers)    
    return classifiers


import random

def get_whitelisted_models(sample_size=15):
    all_models = [
        ("RandomForest", RandomForestClassifier()),
        ("AdaBoost", AdaBoostClassifier()),
        ("Bagging", BaggingClassifier()),
        ("ExtraTrees", ExtraTreesClassifier()),
        ("GradientBoosting", GradientBoostingClassifier()),
        ("HistGradientBoosting", HistGradientBoostingClassifier()),

        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("LogisticRegressionCV", LogisticRegressionCV(cv=3, max_iter=1000)),
        ("RidgeClassifierCV", RidgeClassifierCV()),
        ("SGDClassifier", SGDClassifier(loss="log_loss", max_iter=1000)),
        ("PassiveAggressive", PassiveAggressiveClassifier()),

        ("GaussianNB", GaussianNB()),
        ("BernoulliNB", BernoulliNB()),

        ("KNeighbors", KNeighborsClassifier()),
        ("DecisionTree", DecisionTreeClassifier()),

        ("LDA", LinearDiscriminantAnalysis()),
        ("QDA", QuadraticDiscriminantAnalysis()),

        ("SVC", SVC(probability=True)),  # enable predict_proba

        ("Dummy", DummyClassifier(strategy="most_frequent")),

        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
        ("LGBM", LGBMClassifier())
    ]

    random.shuffle(all_models)
    selected = random.sample(all_models, min(sample_size, len(all_models)))
    return selected


def get_param_grid_for_model(model):
    name = model.__class__.__name__.lower()
    grids = {
        'randomforestclassifier': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        'xgbclassifier': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        },
        'lgbmclassifier': {
            'n_estimators': [100, 200],
            'max_depth': [-1, 10],
            'learning_rate': [0.01, 0.1]
        },
        'adaboostclassifier': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1.0]
        },
        'baggingclassifier': {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.5, 1.0]
        },
        'decisiontreeclassifier': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        },
        'extratreesclassifier': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        'bernoullinb': {
            'alpha': [0.1, 1.0, 10.0],
            'binarize': [0.0, 0.5, 1.0]
        }
    }
    return grids.get(name, None)


 

def train_and_evaluate(
    target: str,
    model_type: str = "auto",
    optimize: bool = False,
    feature_selection="none",
    k_best=15,
    corr_threshold=0.95,
    custom_dataset_path=None
):
    (X_train, X_test, y_train, y_test), feature_cols = preprocess_data(
        target,
        feature_selection=feature_selection,
        k_best=k_best,
        corr_threshold=corr_threshold,
        custom_dataset_path=custom_dataset_path
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_params = None
    model_type = model_type.lower()
    all_results = []

    try:
        if model_type != "auto":
            predefined_models = {
                "randomforest": RandomForestClassifier(n_estimators=100, random_state=42),
                "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0),
                "lightgbm": LGBMClassifier(verbose=-1)
            }

            if model_type not in predefined_models:
                raise ValueError(f"Modèle non supporté : {model_type}")

            model = predefined_models[model_type]
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, output_dict=True)

        else:
            best_model = None
            best_acc = 0
            best_name = ""
            n_classes = len(np.unique(y_train))

            for idx, (name, model) in enumerate(get_whitelisted_models(sample_size=10)):
                if idx >= 12:
                    break
                try:
                    if name.lower() in [
                        "mlpclassifier", "svc", "logisticregressioncv", "calibratedclassifiercv",
                        "categoricalnb", "complementnb", "labelpropagation", "labelspreading",
                        "passiveaggressiveclassifier", "ridgeclassifier", "ridgeclassifiercv",
                        "radiusneighborsclassifier"
                    ]:
                        continue

                    if hasattr(model, "verbose"):
                        model.set_params(verbose=0)

                    if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
                        print(f"⛔ Modèle ignoré (pas de proba ni decision_fn) : {name}")
                        continue

                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    all_results.append({
                        "Model": name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1-score": f1
                    })

                    # Choix du meilleur modèle avec logique inversée si tous les scores sont > 0.97
                    if acc >= 0.97:
                        if best_acc == 0 or acc < best_acc:
                            best_acc = acc
                            best_model = model
                            best_name = name
                    elif best_model is None and acc > best_acc:
                        best_acc = acc
                        best_model = model
                        best_name = name

                except Exception as e:
                    print(f"❌ {name} échoué : {e}")
                    continue

            if best_model is None:
                raise ValueError("Aucun modèle valide trouvé.")

            print(f"✅ Meilleur modèle (avant tuning) : {best_name.upper()} avec Accuracy = {best_acc:.3f}")

            param_grid = get_param_grid_for_model(best_model)
            if optimize and param_grid:
                print(f"🔧 Optimisation de {best_name.upper()}...")
                grid = GridSearchCV(best_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train_scaled, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                print(f"⚠️ Entraînement direct sans tuning pour {best_name.upper()}.")
                model = best_model.fit(X_train_scaled, y_train)

            model_type = best_name
            y_pred = model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, output_dict=True)

        # 📁 Sauvegarde des artefacts
        model_dir = f"models/{target}"
        report_dir = f"reports/{target}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        joblib.dump(model, f"{model_dir}/{model_type}_model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(list(feature_cols), f"{model_dir}/features.pkl")

        pd.DataFrame(all_results).to_csv(f"{report_dir}/benchmark_results.csv", index=False)

        try:
            RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
            plt.title(f"Courbe ROC - {model_type.upper()}")
            plt.savefig(f"{report_dir}/roc_curve.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Erreur génération courbe ROC : {e}")

        if hasattr(model, "feature_importances_") or "XGB" in type(model).__name__ or "LGBM" in type(model).__name__:
            explain_model(model, X_test_scaled, feature_cols, target_name=target)

        summary = f"""
✅ Meilleur modèle : {model_type.upper()}
🎯 Accuracy : {round(report['weighted avg']['precision'], 3)}
🔁 Recall : {round(report['weighted avg']['recall'], 3)}
🎯 F1-score : {round(report['weighted avg']['f1-score'], 3)}
"""
        return summary

    except Exception as e:
        return f"❌ Erreur dans train_and_evaluate : {str(e)}"









def predict_from_input(user_dict, target="LoanApproved", model_type="auto"):
    model_dir = f"models/{target}"
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    features_path = os.path.join(model_dir, "features.pkl")

    if model_type == "auto":
        for f in os.listdir(model_dir):
            if f.endswith("_model.pkl"):
                model_path = os.path.join(model_dir, f)
                break
        else:
            raise FileNotFoundError("Aucun modèle trouvé.")
    else:
        model_path = os.path.join(model_dir, f"{model_type.lower()}_model.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features_loaded = joblib.load(features_path)

    # 🧠 Defaults en fonction de la cible
    base_defaults = {
        "Age": 35,
        "AnnualIncome": 40000,
        "CreditScore": 650,
        "LoanAmount": 15000,
        "LoanDuration": 36,
        "MonthlyIncome": 3000,
        "Experience": 5,
        "DebtToIncomeRatio": 0.3,
        "PaymentHistory": 85,
        "LengthOfCreditHistory": 4,
        "RiskScore": 700,
        "JobTenure": 3,
        "LoanApproved": 1,
        "MaritalStatus_Single": 0,
        "MaritalStatus_Married": 1,
        "HomeOwnershipStatus_Rent": 0,
        "HomeOwnershipStatus_Own": 1,
        "LoanPurpose_Education": 0,
        "LoanPurpose_Other": 0,
        "EmploymentStatus_Self-Employed": 0,
        "EmploymentStatus_Unemployed": 0
    }

    # 👇 Adapte ici si certaines colonnes n’existent que dans un dataset
    # Liste de features attendue par le modèle (priorité au modèle sauvé)
    expected_features = None
    if hasattr(model, "feature_names_in_"):
        try:
            expected_features = list(model.feature_names_in_)
        except Exception:
            expected_features = None
    if expected_features is None:
        expected_features = list(features_loaded)

    # Si le modèle courant n'a pas la même dimensionalité que les features
    model_n = getattr(model, "n_features_in_", None)
    if model_n is not None and model_n != len(expected_features):
        # Essayer de trouver un modèle compatible dans le dossier
        try:
            for f in os.listdir(model_dir):
                if f.endswith("_model.pkl"):
                    cand_path = os.path.join(model_dir, f)
                    cand = joblib.load(cand_path)
                    cand_n = getattr(cand, "n_features_in_", None)
                    if cand_n == len(expected_features):
                        model = cand
                        model_n = cand_n
                        break
        except Exception:
            pass

    input_data = {}
    for col in expected_features:
        if col in user_dict:
            input_data[col] = user_dict[col]
        else:
            input_data[col] = base_defaults.get(col, 0)  # fallback à zéro

    # Respecter l'ordre des colonnes
    X_input = pd.DataFrame([[input_data.get(c) for c in expected_features]], columns=expected_features)

    # Appliquer le scaler uniquement si la dimension correspond
    try:
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == X_input.shape[1]:
            X_scaled = scaler.transform(X_input)
        else:
            X_scaled = X_input
    except Exception:
        X_scaled = X_input

    prediction = model.predict(X_scaled)

    if target == "LoanApproved":
        return "Approved" if prediction[0] == 1 else "Rejected"
    elif target == "RiskClass":
        label_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        return label_map.get(prediction[0], prediction[0])
    else:
        return prediction[0]

def explain_model(model, X_test, feature_names, target_name="LoanApproved", instance_index=0):
    # 📁 Créer le dossier s'il n'existe pas
    reports_dir = f"reports/{target_name}"
    os.makedirs(reports_dir, exist_ok=True)

    # 🧠 Créer l'explainer adapté à l'arbre
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
    except Exception as e:
        print(f"⚠️ Modèle non compatible avec SHAP : {type(model).__name__} → {e}")
        return  # skip shap pour ce modèle

    # ✅ Sauvegarde PNG - summary plot
    try:
        shap.plots.beeswarm(shap_values, show=False)
        summary_png_path = os.path.join(reports_dir, "summary_plot.png")
        plt.tight_layout()
        plt.savefig(summary_png_path)
        plt.clf()
        print(f"✅ Summary plot PNG sauvegardé : {summary_png_path}")
    except Exception as e:
        print(f"❌ Erreur beeswarm PNG : {e}")

    # ✅ Sauvegarde PNG - waterfall plot
    try:
        shap.plots.waterfall(shap_values[instance_index], show=False)
        waterfall_png_path = os.path.join(reports_dir, "waterfall_plot.png")
        plt.tight_layout()
        plt.savefig(waterfall_png_path)
        plt.clf()
        print(f"✅ Waterfall plot PNG sauvegardé : {waterfall_png_path}")
    except Exception as e:
        print(f"❌ Erreur waterfall PNG : {e}")

    # ✅ Sauvegarde HTML interactif
    try:
        summary_html_path = os.path.join(reports_dir, "summary_plot_interactive.html")
        summary_plot = shap.plots.beeswarm(shap_values, show=False)
        shap.save_html(summary_html_path, summary_plot)
        print(f"✅ HTML interactif sauvegardé : {summary_html_path}")
        webbrowser.open(f"file://{os.path.abspath(summary_html_path)}")
    except Exception as e:
        print(f"❌ Erreur HTML interactif : {e}")


def explain_model_from_disk(target="LoanApproved"):
    import joblib
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split
    from src.explainability import explain_model  # ajuste le chemin si besoin

    # 🔁 Vérification des chemins
    model_dir = f"models/{target}"
    model_file = next((f for f in os.listdir(model_dir) if f.endswith("_model.pkl")), None)
    if not model_file:
        raise FileNotFoundError(f"Aucun modèle entraîné trouvé dans {model_dir}")

    # 📦 Chargement des artefacts
    model = joblib.load(os.path.join(model_dir, model_file))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    features = joblib.load(os.path.join(model_dir, "features.pkl"))

    # 📄 Chargement du bon dataset
    if target == "LoanApproved":
        df = pd.read_csv("data/preprocessed_loanapproved.csv")
        y_col = "LoanApproved"
    elif target == "RiskClass":
        df = pd.read_csv("data/preprocessed_riskclass.csv")
        y_col = "RiskClassEncoded"
    else:
        raise ValueError(f"Cible inconnue : {target}")

    X = df.drop(columns=[y_col])
    _, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_scaled = scaler.transform(X_test)

    # 🧠 Génération des explications SHAP
    explain_model(model, X_scaled, features, target)

    return f"✅ Graphiques SHAP générés dans reports/{target}/"




if __name__ == "__main__":
    
    explain_model_from_disk("LoanApproved")
