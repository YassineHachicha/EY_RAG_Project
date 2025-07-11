
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
    if "loan" in q or "approval" in q:
        return "loan_policy"
    elif "capital" in q or "tier 1" in q:
        return "basel3"
    elif "weight" in q or "asset" in q:
        return "risk_weights"
    else:
        return "loan_policy"

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
    prompt = f"""You are a regulatory assistant.

Context:
{context}

Question:
{question}

Answer clearly using only the provided context.
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
