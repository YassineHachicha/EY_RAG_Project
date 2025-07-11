import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Charger les données
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Entraîner un modèle simple
model = RandomForestClassifier()
model.fit(X, y)

# Prédire sur un échantillon
sample = X.iloc[[0]]
prediction = model.predict(sample)

# Afficher résultat
print("✅ Modèle entraîné avec succès.")
print(f"🎯 Prédiction sur le premier exemple : {prediction[0]} (classe réelle : {y[0]})")
