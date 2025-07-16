# agent_core.py
import json
from datetime import datetime
import numpy as np
from ml_pipeline import predict_from_input
from eval_agent import evaluate_models_for_target
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def log_result(result, target):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def convert_np(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return obj

    if isinstance(result, (np.integer, np.int64, int, float, np.floating)):
        safe_result = {"prediction": convert_np(result)}
    elif isinstance(result, dict):
        safe_result = {k: convert_np(v) for k, v in result.items()}
    else:
        safe_result = {"output": str(result)}

    with open("log_predictions.jsonl", "a") as f:
        log_entry = {
            "timestamp": timestamp,
            "target": target,
            "result": safe_result
        }
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def agent_main():
    print("Bienvenue dans l’agent IA pour la classification.")
    while True:
        print("\n====================")
        print("Options disponibles :")
        print("1. Prédire à partir d’un input utilisateur")
        print("2. Évaluer les modèles pour LoanApproved")
        print("3. Évaluer les modèles pour RiskClass")
        print("4. Quitter")
        print("====================")

        choice = input("Votre choix : ")

        if choice == "1":
            print("\n✨ Exemple d'input utilisateur pour LoanApproved :")
            exemple_input = {
                "Age": 42,
                "AnnualIncome": 60000,
                "CreditScore": 700,
                "Experience": 10,
                "LoanAmount": 20000,
                "LoanDuration": 36,
                "NumberOfDependents": 2,
                "MonthlyDebtPayments": 500,
                "CreditCardUtilizationRate": 0.3,
                "NumberOfOpenCreditLines": 3,
                "MonthlyIncome": 5000,
                "UtilityBillsPaymentHistory": 0.9,
                "JobTenure": 5,
                "NetWorth": 100000,
                "BaseInterestRate": 0.1,
                "InterestRate": 0.12,
                "MonthlyLoanPayment": 600,
                "TotalDebtToIncomeRatio": 0.2,
                "EmploymentStatus_Self-Employed": 0,
                "EmploymentStatus_Unemployed": 0,
                "MaritalStatus_Married": 1,
                "MaritalStatus_Single": 0,
                "MaritalStatus_Widowed": 0,
                "HomeOwnershipStatus_Other": 0,
                "HomeOwnershipStatus_Own": 1,
                "HomeOwnershipStatus_Rent": 0,
                "LoanPurpose_Debt Consolidation": 1,
                "LoanPurpose_Education": 0,
                "LoanPurpose_Home": 0,
                "LoanPurpose_Other": 0
            }

            target = input("\nQuelle cible prédire ? (LoanApproved / RiskClass) : ")
            model = input("Quel modèle utiliser ? (randomforest / xgboost) : ")

            try:
                result = predict_from_input(exemple_input, target=target, model_type=model)
                print("\n🔮 Résultat :")
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
                log_result(result, target)
            except Exception as e:
                print(f"\n🚫 Erreur de prédiction : {e}")

        elif choice == "2" or choice == "3":
            target = "LoanApproved" if choice == "2" else "RiskClass"
            print(f"\n🔍 Test des modèles pour {target}")

            from ml_pipeline import preprocess_data
            from sklearn.ensemble import RandomForestClassifier
            from xgboost import XGBClassifier

            X_train, X_test, y_train, y_test = preprocess_data(target)

            models = {
                "randomforest": RandomForestClassifier(random_state=42),
                "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
            }

            for name, model in models.items():
                print(f"\n📊 Modèle : {name.upper()}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted")
                rec = recall_score(y_test, y_pred, average="weighted")

                print(f"✅ Accuracy : {acc:.4f}")
                print(f"🎯 Precision : {prec:.4f}")
                print(f"🔁 Recall : {rec:.4f}")
                print("\n🧾 Classification Report :\n", classification_report(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix - {name.upper()} ({target})")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.tight_layout()
                plt.show()

        elif choice == "4":
            print("\n🚩 Fin de l'agent.")
            break

        else:
            print("\n🚫 Choix invalide. Veuillez réessayer.")

if __name__ == "__main__":
    agent_main()
