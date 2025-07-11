
import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 📁 Chemins
index_dir = Path("faiss_index")
chunk_map_path = index_dir / "chunk_mapping.json"

# 🧠 Charger modèle d'embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# 📦 Charger le mapping des chunks
with open(chunk_map_path, "r", encoding="utf-8") as f:
    chunk_mapping = json.load(f)

# 🔍 Fonction de routage très simple
def select_doc(question):
    q = question.lower()
    if "loan" in q or "approval" in q:
        return "loan_policy"
    elif "capital" in q or "tier 1" in q:
        return "basel3"
    elif "weight" in q or "asset" in q:
        return "risk_weights"
    else:
        return "loan_policy"  # défaut

# 🔁 Fonction principale
def retrieve_answer(question, top_k=3):
    doc_key = select_doc(question)
    print(f"📚 Document sélectionné : {doc_key}")

    # Charger index FAISS
    index_path = index_dir / f"{doc_key}.idx"
    index = faiss.read_index(str(index_path))

    # Encoder la question
    query_vec = model.encode([question]).astype("float32")

    # Recherche
    D, I = index.search(query_vec, top_k)
    chunks = chunk_mapping[doc_key]
    results = [chunks[i] for i in I[0]]

    print("\n🎯 Résultats pertinents :\n")
    for idx, chunk in enumerate(results):
        print(f"[{idx+1}] {chunk}\n")

# 🔘 Exécution manuelle
if __name__ == "__main__":
    print("Pose une question (ex: What are the conditions to approve a loan?)")
    question = input("🧠 Question : ")
    retrieve_answer(question)
