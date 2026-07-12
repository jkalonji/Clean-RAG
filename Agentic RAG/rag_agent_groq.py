"""
Agentic RAG — version Groq (llama-3.3-70b-versatile)

Même architecture que rag_agent.py, seul le LLM change.
Permet de comparer : modèle local (LMStudio) vs modèle cloud (Groq).

Prérequis : clé API Groq dans les variables d'environnement
  Windows PowerShell : $env:GROQ_API_KEY = "gsk_..."
  Linux / macOS      : export GROQ_API_KEY="gsk_..."
  Clé gratuite sur   : https://console.groq.com

Usage :
  cd "Agentic RAG"
  python rag_agent_groq.py
"""

import os

from langchain_groq import ChatGroq

from config import GROQ_MODEL
from tools import init_vectorstore, search_documents, search_web
from rag_agent import SYSTEM_PROMPT, build_agent


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERREUR : variable d'environnement GROQ_API_KEY non définie.")
        print("Définissez-la avec :")
        print("  PowerShell : $env:GROQ_API_KEY = 'gsk_...'")
        print("  Bash       : export GROQ_API_KEY='gsk_...'")
        return

    init_vectorstore()

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    agent_executor = build_agent(llm)

    print("\n=== Agentic RAG — Groq ===")
    print(f"Modèle    : {GROQ_MODEL}")
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
