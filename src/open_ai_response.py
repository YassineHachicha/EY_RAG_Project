from openai import OpenAI

# Configurer le client avec OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"  # remplace par ta clé réelle
)

# Charger le prompt depuis ton système RAG
with open("generated_prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

# Appel au modèle
response = client.chat.completions.create(
    model="deepseek/deepseek-r1-0528",  # ou mistralai/mistral-7b-instruct
    messages=[
        {"role": "system", "content": "You are a regulatory assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=512,
    temperature=0.3,
    extra_headers={
        "HTTP-Referer": "https://votresite.com",  # optionnel
        "X-Title": "RAGPrototype",                # optionnel
    }
)

# Afficher la réponse
print("🧠 Réponse générée :")
print(response.choices[0].message.content)
