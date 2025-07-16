
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# Charger le dataset brut
df = pd.read_csv('P1M2_Yonathan_Anggraiwan.csv', sep=';')
df = df.drop(columns=['ApplicationDate'])


# Création de la variable cible catégorielle à partir de RiskScore
def categorize_risk(score):
    if score < 40:
        return "Low"
    elif score <= 60:
        return "Medium"
    else:
        return "High"

# Ajouter la colonne 'RiskClass'
df['RiskClass'] = df['RiskScore'].apply(categorize_risk)

# (Optionnel) Encodage numérique pour la classification
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['RiskClassEncoded'] = label_encoder.fit_transform(df['RiskClass'])

# Nettoyage des chaînes (strip)
df['EducationLevel'] = df['EducationLevel'].astype(str).str.strip()

# Mapping ordonné
edu_map = {
    'High School': 1,
    'Associate': 2,
    'Bachelor': 3,
    'Master': 4,
    'Doctorate': 5
}
df['EducationLevel'] = df['EducationLevel'].map(edu_map)




df_encoded = pd.get_dummies(df, columns=[
    'EmploymentStatus',
    'MaritalStatus',
    'HomeOwnershipStatus',
    'LoanPurpose'
], drop_first=True)

# Conversion booléens → entiers
# Ne convertir que les colonnes booléennes en int
for col in df_encoded.select_dtypes(include='bool').columns:
    df_encoded[col] = df_encoded[col].astype(int)

    

# 1. Séparer les colonnes non numériques AVANT le traitement
colonnes_non_numeriques = df_encoded.select_dtypes(exclude='number')

# 2. Sélectionner les colonnes numériques pour le capping
colonnes_numeriques = df_encoded.select_dtypes(include='number')

# 3. Calcul des bornes pour le capping (IQR)
Q1 = colonnes_numeriques.quantile(0.25)
Q3 = colonnes_numeriques.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 4. Appliquer le capping sur les colonnes numériques
colonnes_numeriques_capped = colonnes_numeriques.clip(lower=lower_bound, upper=upper_bound, axis=1)

# 5. Réintégrer les colonnes non numériques dans le DataFrame final
df_final = pd.concat([colonnes_non_numeriques.reset_index(drop=True), colonnes_numeriques_capped.reset_index(drop=True)], axis=1)



# Définition des features et de la nouvelle target RiskClassEncoded
features = df_encoded.drop(['RiskClass', 'RiskClassEncoded'], axis=1)
target = df_encoded['RiskClassEncoded']

# Mise à l'échelle des colonnes numériques
numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)



# Vérifie d’abord que la variable cible existe bien avec les deux classes
print(df_encoded["RiskClassEncoded"].value_counts())

# Séparation X/y
X = df_encoded.drop("RiskClassEncoded", axis=1)
y = df_encoded["RiskClassEncoded"]

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

print(pd.Series(y_resampled).value_counts())

X_resampled["RiskClassEncoded"] = y_resampled

X_resampled.to_csv("preprocessed_riskclass.csv", index=False)



























































































