
from ml_pipeline import train_and_evaluate

print("🔍 Test modèle XGBoost sur LoanApproved :")
print(train_and_evaluate("LoanApproved", "xgboost"))

print("\n🔍 Test modèle RandomForest sur RiskClass :")
print(train_and_evaluate("RiskClass", "randomforest"))
