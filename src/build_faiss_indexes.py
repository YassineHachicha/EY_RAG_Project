
import os
import json
import faiss
import numpy as np
import textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 📁 Répertoires
documents_dir = Path("documents")
index_dir = Path("faiss_index")
index_dir.mkdir(exist_ok=True)

# 🧠 Modèle d'embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# 📦 Mapping pour chunks
chunk_mapping = {}

# 🔁 Parcourir chaque .txt
for file_path in documents_dir.glob("*.txt"):
    doc_name = file_path.stem
    print(f"📄 Indexation : {doc_name}.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ✂️ Chunking (500 caractères)
    chunks = textwrap.wrap(text, width=500)
    chunk_mapping[doc_name] = chunks

    # 🔢 Embeddings
    embeddings = model.encode(chunks, show_progress_bar=False)
    embedding_matrix = np.array(embeddings).astype("float32")

    # 🔄 FAISS index
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)

    # 💾 Sauvegarde
    faiss.write_index(index, str(index_dir / f"{doc_name}.idx"))
# ✅ Création des graphes sémantiques pour chaque document
from rag_multi_doc_generator import build_graph_from_chunks

for doc_key, chunks in chunk_mapping.items():
    build_graph_from_chunks(chunks, doc_key)

# 💾 Mapping chunks ↔ documents
with open(index_dir / "chunk_mapping.json", "w", encoding="utf-8") as f:
    json.dump(chunk_mapping, f, indent=2)

print("✅ Indexation terminée pour tous les documents.")
