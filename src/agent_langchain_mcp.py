from langchain.agents import Tool
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from ml_pipeline import train_and_evaluate, predict_from_input
from rag_multi_doc_generator import generate_answer
import json
import os



# ✅ LLM configuration (OpenRouter)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    model="deepseek/deepseek-chat",
    temperature=0.3,
    max_tokens=512 
)


# 🛠️ Tool 1: Entraînement ML
def train_ml_model(query: str) -> str:
    try:
        parsed = json.loads(query)
        target = parsed.get("target", "LoanApproved")
        model_name = parsed.get("models", ["auto"])[0].lower()
        optimize = parsed.get("optimize", False)
        feature_selection = parsed.get("feature_selection", "none")
        k_best = parsed.get("k_best", 15)
        corr_threshold = parsed.get("corr_threshold", 0.95)
    except:
        target = "RiskClass" if "risk" in query.lower() else "LoanApproved"
        model_name = "xgboost" if "xgboost" in query.lower() else "randomforest"
        optimize = "optimize" in query.lower() or "hyperparam" in query.lower()

    return train_and_evaluate(
        target=target,
        model_type=model_name,
        optimize=optimize,
        feature_selection=feature_selection,
        k_best=k_best,
        corr_threshold=corr_threshold
    )

ml_tool = Tool(
    name="MLTrainerTool",
    func=train_ml_model,
    description=(
        "Entraîne un modèle ML sur LoanApproved ou RiskClass (XGBoost, RandomForest ou auto). "
        "Inclut l'optimisation des hyperparamètres si l'instruction contient 'optimize' ou 'hyperparam'."
    )
)

# 🛠️ Tool 2: Prédiction ML
def ml_predictor(json_input: str) -> str:
    try:
        inputs = json.loads(json_input)
        if "features" not in inputs:
            return "Erreur : la clé 'features' est manquante. Attendu format : {\"features\": {...}, \"target\": \"LoanApproved\", \"model_type\": \"xgboost\"}"
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

# 🛠️ Tool 3: Recherche RAG (Graph RAG uniquement)
from rag_multi_doc_generator import generate_answer
rag_tool = Tool(
    name="RAGRetrieverTool",
    func=lambda question: generate_answer(question, llm=llm, top_k=3),
    description="Récupère un passage réglementaire depuis un graphe sémantique (Graph RAG uniquement)."
)

# 🛠️ Tool 4: Résumé LLM (multilingue simplifié)
def generate_summary_structured(
    ml_summary: str,
    regulatory_summary: str,
    lang: str = "fr"  # "fr", "en", "es", "ar", etc.
) -> str:
    """
    Génère un rapport exécutif multilingue.
    - lang: langue de sortie (ex: "fr", "en", "es", "ar").
    """
    max_len = 4000
    ml_summary = (ml_summary or "")[:max_len]
    regulatory_summary = (regulatory_summary or "")[:max_len]

    lang_names = {
        "fr": "French",
        "en": "English",
        "es": "Spanish",
        "ar": "Arabic",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese"
    }
    lang_name = lang_names.get(lang.lower(), "English")

    prompt = f"""
You are an executive reporting assistant. Write the report in {lang_name}.

# Machine Learning Results
{ml_summary}

# Regulatory Context
{regulatory_summary}

Requirements:
- Single coherent narrative (no bullet points).
- Professional and factual.
- If regulatory context is partial, state limitations briefly.
- Include 1 sentence on model limitations/assumptions.
"""
    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


report_tool = StructuredTool.from_function(
    func=generate_summary_structured,
    name="LLMReporterTool",
    description="Génère un rapport exécutif multilangue à partir d’un résumé ML et d’un contexte réglementaire."
)

# 🚀 Agent MCP final
tools = [ml_tool, predict_tool, rag_tool, report_tool]

agent = StructuredChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    verbose=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


def run_autonomous_pipeline():
    print("\n🚀 Lancement du pipeline autonome...\n")

    target_input = input("📌 Que souhaitez-vous prédire ? Tapez 'LoanApproved' ou 'RiskClass' : ").strip().lower()
    if target_input not in ["loanapproved", "riskclass"]:
        print("❌ Cible non valide. Abandon.")
        return

    target = "LoanApproved" if target_input == "loanapproved" else "RiskClass"

    opt_choice = input("⚙️ Souhaitez-vous activer l’optimisation des hyperparamètres ? (oui/non) : ").strip().lower()
    optimize = opt_choice in ["oui", "yes", "y"]

    # 🆕 Choix de la langue du rapport
    lang_choice = input("🌍 En quelle langue souhaitez-vous le rapport ? (fr/en/es/ar...) : ").strip().lower()

    print(f"\n▶️ Étape 1 : Sélection {'optimisée' if optimize else 'rapide'} du meilleur modèle pour prédire {target}...\n")

    query = json.dumps({
        "model_type": "auto",
        "target": target,
        "optimize": optimize,
        "feature_selection": "kbest",
        "k_best": 15,
        "corr_threshold": 0.95
    })

    result_train = agent_executor.invoke(query)

    print("\n▶️ Étape 2 : Récupération d'informations réglementaires (ex: Bâle III)...")
    result_rag = agent_executor.invoke("Quelle est la définition du capital Tier 1 dans Bâle III ?")
    ml_text = result_train.get("output") if isinstance(result_train, dict) else str(result_train)
    rag_text = result_rag.get("output")   if isinstance(result_rag, dict)   else str(result_rag)
    print("\n▶️ Étape 3 : Génération du rapport exécutif multilingue...")
    result_summary = generate_summary_structured(ml_text, rag_text, lang=lang_choice)

    print("\n✅ Pipeline terminé. Résumé exécutif :\n")
    print(result_summary)



def afficher_menu_examples():
    print("\n🧪 Exemples d’instructions que tu peux tester :")
    print("1. Entraîne un modèle XGBoost pour prédire LoanApproved")
    print("2. Lance un entraînement RandomForest sur RiskClass")
    print("3. Prédit LoanApproved à partir d’un input : {\"features\": {...}, \"target\": \"LoanApproved\", \"model_type\": \"xgboost\"}")
    print("4. Quelle est la définition du capital Tier 1 dans Bâle III ?")
    print("5. Résume ce résultat : le modèle XGBoost atteint 99% de précision sur LoanApproved")
    print("6. Tape 'auto' pour lancer tout le pipeline automatiquement")
    print("7. Tape 'exit' pour quitter")


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
                result = agent_executor.invoke(q)
                print("\n🧾 Réponse de l’agent :")
                print(result)
            except Exception as e:
                print(f"❌ Erreur : {e}")

