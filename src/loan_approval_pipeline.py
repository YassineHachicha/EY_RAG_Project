#!/usr/bin/env python
# coding: utf-8

# ##  Data Preparation 

# In[72]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
df = pd.read_csv('P1M2_Yonathan_Anggraiwan.csv', sep=';')
df = df.drop(columns=['ApplicationDate'])
df.head()


# In[73]:


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

# Vérification
print(df['EducationLevel'].isnull().sum())
df['EducationLevel'].unique()


# ##  Encodage des variables catégorielles

# In[74]:


df_encoded = pd.get_dummies(df, columns=[
    'EmploymentStatus',
    'MaritalStatus',
    'HomeOwnershipStatus',
    'LoanPurpose'
], drop_first=True)

# Conversion booléens → entiers
df_encoded = df_encoded.astype(int)
df_encoded.head()


# ##  Traitement des outliers par Winsorizing (Capping)

# In[75]:


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

# 6. Afficher le résultat
df_final.head()


# ##  Standardisation des variables numériques

# In[76]:


features = df_final.drop('LoanApproved', axis=1)
target = df_final['LoanApproved']

numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
features.head()


# ##  Séparation Train/Test

# In[77]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)


# In[78]:


print(df_encoded["LoanApproved"].value_counts())


# In[79]:


from sklearn.model_selection import train_test_split

# Vérifie d’abord que la variable cible existe bien avec les deux classes
print(df_encoded["LoanApproved"].value_counts())

# Séparation X/y
X = df_encoded.drop("LoanApproved", axis=1)
y = df_encoded["LoanApproved"]

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Vérification
print("Répartition dans y_train :")
print(y_train.value_counts())



# ##  Rééquilibrage des classes avec SMOTE

# In[80]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Répartition après SMOTE :")
print(y_train_res.value_counts())

X_train_res["LoanApproved"] = y_train_res  # Recombiner features + target
X_train_res.to_csv("preprocessed_loanapproved.csv", index=False)

