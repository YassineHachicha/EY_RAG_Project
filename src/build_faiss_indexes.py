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

# 🔁 Pour chaque document
for file_path in documents_dir.glob("*.txt"):
    doc_name = file_path.stem  # Exemple : basel3
    print(f"📄 Traitement de {doc_name}.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ✂️ Découper le texte (~300 caractères)
    chunks = textwrap.wrap(text, width=300)
    chunk_mapping[doc_name] = chunks

    # 🔢 Embedding des chunks
    embeddings = model.encode(chunks)
    embedding_matrix = np.array(embeddings).astype("float32")

    # 🧊 Création de l’index FAISS
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)

    # 💾 Sauvegarde de l'index
    faiss.write_index(index, str(index_dir / f"{doc_name}.idx"))


# 💾 Sauvegarder le mapping chunks ↔ documents
with open(index_dir / "chunk_mapping.json", "w", encoding="utf-8") as f:
    json.dump(chunk_mapping, f, indent=2)

print("✅ Indexation terminée.")
