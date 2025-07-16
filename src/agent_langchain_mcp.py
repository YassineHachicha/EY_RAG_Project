# agent_langchain_mcp.py

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from ml_pipeline import train_and_evaluate, predict_from_input
import json
import os
from rag_multi_doc_generator import generate_answer

# ✅ LLM configuration (OpenRouter)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    model="deepseek/deepseek-chat",
    temperature=0.3
)

# 🛠️ Tool 1: Entraînement ML
def train_ml_model(query: str) -> str:
    target = "RiskClass" if "risk" in query.lower() else "LoanApproved"
    model_name = "xgboost" if "xgboost" in query.lower() else "randomforest"
    return str(train_and_evaluate(target=target, model_type=model_name))

ml_tool = Tool(
    name="MLTrainerTool",
    func=train_ml_model,
    description="Entraîne un modèle ML sur LoanApproved ou RiskClass (XGBoost ou RandomForest)."
)

# 🛠️ Tool 2: Prédiction ML
def ml_predictor(json_input: str) -> str:
    try:
        inputs = json.loads(json_input)
        prediction = predict_from_input(
            inputs["features"],
            target=inputs["target"],
            model_type=inputs["model_type"]
        )
        return json.dumps(prediction, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {str(e)}"

predict_tool = Tool(
    name="MLPredictorTool",
    func=ml_predictor,
    description="Prédit LoanApproved ou RiskClass à partir d’un input utilisateur JSON."
)

# 🛠️ Tool 3: Recherche RAG (FAISS)
  # 👈 importe la vraie fonction

# 🛠️ Tool 3: Recherche RAG (FAISS)
def retrieve_rag_info(question: str) -> str:
    return generate_answer(question, top_k=3)


rag_tool = Tool(
    name="RAGRetrieverTool",
    func=retrieve_rag_info,
    description="Récupère un passage réglementaire depuis des documents FAISS indexés."
)

# 🛠️ Tool 4: Résumé LLM
def generate_summary(input: str) -> str:
    return llm.predict(f"Rédige un rapport exécutif sur ce résultat :\n{input}")

report_tool = Tool(
    name="LLMReporterTool",
    func=generate_summary,
    description="Génère un résumé clair à partir des sorties ML et réglementaires."
)

# 🚀 Agent MCP final
tools = [ml_tool, predict_tool, rag_tool, report_tool]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_autonomous_pipeline():
    print("\n🚀 Lancement du pipeline autonome...\n")

    # Étape 0 : Demander la cible
    target = input("📌 Que souhaitez-vous prédire ? Tapez 'LoanApproved' ou 'RiskClass' : ").strip()
    if target.lower() not in ["loanapproved", "riskclass"]:
        print("❌ Cible non valide. Abandon.")
        return

    # Étape 1 : Benchmark et entraînement
    print(f"\n▶️ Étape 1 : Sélection automatique du meilleur modèle pour prédire {target}...")
    query = f"Benchmark plusieurs modèles pour prédire {target} et choisis le meilleur."
    result_train = agent.run(query)

    # Étape 2 : Récupération d’information réglementaire
    print("\n▶️ Étape 2 : Récupération d'informations réglementaires (ex: Bâle III)...")
    result_rag = agent.run("Quelle est la définition du capital Tier 1 dans Bâle III ?")

    # Étape 3 : Rapport exécutif
    print("\n▶️ Étape 3 : Génération du rapport exécutif...")
    result_summary = agent.run(f"Résume ce résultat : {result_train} + contexte : {result_rag}")

    # Résultat final
    print("\n✅ Pipeline terminé. Résumé exécutif :\n")
    print(result_summary)


# 🔖 Menu d’exemples
def afficher_menu_examples():
    print("\n🧪 Exemples d’instructions que tu peux tester :")
    print("1. Entraîne un modèle XGBoost pour prédire LoanApproved")
    print("2. Lance un entraînement RandomForest sur RiskClass")
    print("3. Prédit LoanApproved à partir d’un input : {\"features\": {...}, \"target\": \"LoanApproved\", \"model_type\": \"xgboost\"}")
    print("4. Quelle est la définition du capital Tier 1 dans Bâle III ?")
    print("5. Résume ce résultat : le modèle XGBoost atteint 99% de précision sur LoanApproved")
    print("6. Tape 'auto' pour lancer tout le pipeline automatiquement")
    print("7. Tape 'exit' pour quitter")

# 🚀 Interface utilisateur
if __name__ == "__main__":
    print("🧠 Agent MCP prêt ! Tape une instruction (ou 'menu' pour des exemples, 'exit' pour quitter)...")
    while True:
        q = input("\nInstruction : ")
        if q.lower() in ["exit", "quit"]:
            break
        elif q.lower() == "menu":
            afficher_menu_examples()
        elif q.lower() == "auto":
            run_autonomous_pipeline()
        else:
            try:
                result = agent.invoke(q)
                print("\n🧾 Réponse de l’agent :")
                print(result)
            except Exception as e:
                print(f"❌ Erreur : {e}")
