"""
Agentic RAG — version LMStudio (google/gemma-4-12b-qat)

Stratégie de l'agent :
  1. Appelle search_documents → cherche dans les factures locales
  2. Si "CONFIANCE INSUFFISANTE" → appelle search_web (DuckDuckGo)
  3. Génère une réponse en indiquant la source utilisée

verbose=True affiche le raisonnement de l'agent étape par étape.

Usage :
  cd "Agentic RAG"
  python rag_agent.py
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from config import LM_STUDIO_BASE_URL, LM_STUDIO_MODEL
from tools import init_vectorstore, search_documents, search_web

SYSTEM_PROMPT = """Tu es un assistant intelligent avec accès à deux sources d'information :
- search_documents : base de données locale de factures clients
- search_web : internet (DuckDuckGo)

STRATÉGIE :
1. Appelle TOUJOURS search_documents en premier.
2. Si la réponse contient "CONFIANCE INSUFFISANTE", appelle search_web.
3. Ne cherche sur le web que si les documents locaux sont insuffisants.
4. Indique ta source dans la réponse finale : [Source : documents locaux] ou [Source : web].
5. Réponds en français, de façon concise et précise."""


def build_agent(llm) -> AgentExecutor:
    tools = [search_documents, search_web]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)


def main():
    init_vectorstore()

    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0,
    )
    agent_executor = build_agent(llm)

    print("\n=== Agentic RAG — LMStudio ===")
    print(f"Modèle    : {LM_STUDIO_MODEL}")
    print(f"Vectorstore : ../LMStudio/data/invoices/")
    print("Tapez 'exit' pour quitter.\n")

    while True:
        question = input("Question : ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        result = agent_executor.invoke({"input": question})
        print(f"\nRéponse finale : {result['output']}\n")
        print("─" * 60)


if __name__ == "__main__":
    main()
