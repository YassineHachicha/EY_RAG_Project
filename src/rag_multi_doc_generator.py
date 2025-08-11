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


def retrieve_from_graph(question, doc_key, chunk_mapping, model, top_k=3):
    import faiss
    import networkx as nx
    from pathlib import Path

    index_path = Path(f"faiss_index/{doc_key}.idx")
    import pickle

    with open(f"indices/{doc_key}_graph.gpickle", "rb") as f:
         G = pickle.load(f)


    query_vec = model.encode([question]).astype("float32")
    index = faiss.read_index(str(index_path))
    D, I = index.search(query_vec, top_k)

    initial_nodes = I[0]
    expanded_nodes = set(initial_nodes)

    for node in initial_nodes:
        neighbors = list(G.neighbors(node))
        expanded_nodes.update(neighbors[:2])

    chunks = chunk_mapping[doc_key]
    return "\n\n".join([chunks[i] for i in expanded_nodes if i < len(chunks)])


def generate_answer(question: str, llm, top_k=3) -> str:
    doc_key = select_doc(question)
    print(f"🔍 Mode : Graph RAG activé sur le document : {doc_key}")
    context = retrieve_from_graph(question, doc_key, chunk_mapping, model, top_k=top_k)

    prompt = f"""
Tu es un assistant réglementaire expert en conformité bancaire.

Contexte extrait des documents :
{context}

Question :
{question}

Réponds de façon claire et concise en citant uniquement le contexte. Si le contexte est insuffisant, dis-le.
"""
    return llm.invoke(prompt)

def build_graph_from_chunks(chunks: list[str], doc_key: str):
    import networkx as nx
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_tensor=True)

    G = nx.Graph()
    for i, chunk in enumerate(chunks):
        G.add_node(i, text=chunk)

    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
    threshold = 0.6

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if cos_scores[i][j] > threshold:
                G.add_edge(i, j, weight=float(cos_scores[i][j]))

    import os
    graph_path = f"indices/{doc_key}_graph.gpickle"
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    import pickle
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"✅ Graphe sauvé : {graph_path}")

if __name__ == "__main__":
    question = input("🧠 Question réglementaire : ")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
        model="deepseek/deepseek-chat",
        temperature=0.3,
        max_tokens=512
    )
    print(generate_answer(question, llm=llm))
