
import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 📁 Dossiers
index_dir = Path("faiss_index")
chunk_map_path = index_dir / "chunk_mapping.json"

# 🧠 Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 📦 Chunks
with open(chunk_map_path, "r", encoding="utf-8") as f:
    chunk_mapping = json.load(f)

# 🔍 Router
def select_doc(question):
    q = question.lower()
   

    if "bâle" in q or "basel" in q or "tier 1" in q or "capital" in q:
        return "bale3_definitions"
    elif "solvabilité" in q or "directive 2009" in q or "assurance" in q:
        return "reglement_solvabilite2_ue"
    elif "mrt" in q or "crd4" in q or "rémunération groupe" in q:
        return "rapport_remunerations_mrt_2019"
    elif "mandataire" in q or "président" in q or "variable long terme" in q:
        return "politique_remuneration_mandataires"
    else:
        return "bale3_definitions"


# 🔁 Retrieval + Prompt + LLM
def generate_answer(question, top_k=3):
    doc_key = select_doc(question)
    print(f"📚 Using document: {doc_key}")

    index_path = index_dir / f"{doc_key}.idx"
    index = faiss.read_index(str(index_path))

    query_vec = model.encode([question]).astype("float32")
    D, I = index.search(query_vec, top_k)

    chunks = chunk_mapping[doc_key]
    retrieved = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved)

    # 🔧 Prompt
    prompt = f"""Tu es un assistant réglementaire expert en conformité bancaire.

Contexte extrait de la réglementation :
{context}

Question :
{question}

Réponds de façon claire et concise, en citant uniquement le contexte fourni. Si le contexte est insuffisant, dis-le.
"""


    # 🧠 LLM via OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"  # 🔐 Remplace par ta clé OpenRouter
    )
    

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528",  # ou mistralai/mistral-7b-instruct
        messages=[
            {"role": "system", "content": "You are a regulatory assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.3,
        extra_headers={
            "HTTP-Referer": "https://votresite.com",
            "X-Title": "MultiDoc-RAG"
        }
    )

    reply = response.choices[0].message.content
    print("\n🧠 Réponse générée :\n")
    print(reply)

# 🔘 Interaction manuelle
if __name__ == "__main__":
    question = input("🧠 Question réglementaire : ")
    generate_answer(question)
