
# 🧠 EY_RAG_Project – AI & Data Internship

Ce projet a été réalisé dans le cadre du **Summer Internship EY - AI & Data**, et porte sur le développement d’un **système de classification bancaire** enrichi par une architecture **RAG (Retrieval-Augmented Generation)** multi-document.

---

## 🚀 Objectifs du projet

- 🔍 Analyser un dataset de crédit (données client, scoring, statut de prêt)
- ⚙️ Mettre en place un pipeline complet de traitement des données
- 🧠 Déployer un système RAG pour interroger des documents réglementaires
- 📚 Exploiter plusieurs sources (`Basel III`, `Loan Policies`, `Risk Weights`) pour répondre à des questions via un LLM

---

## 🗃️ Structure du projet
EY_RAG_Project/
├── data/ # Dataset CSV brut
├── documents/ # Fichiers réglementaires (.txt)
├── faiss_indexes/ # Indexs FAISS et mapping JSON
├── notebooks/ # EDA, visualisations, PCA, etc.
├── outputs/ # Prompts générés, logs éventuels
├── src/ # Scripts Python (RAG, retrieval, FAISS)
├── .gitignore # Fichiers à ignorer
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier
