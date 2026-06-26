"""
RAG Map-Reduce sur factures TechSolutions
Map : 1 appel LLM par document
Reduce : synthèse de toutes les réponses partielles
Pas de vectorstore — tous les documents sont traités.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL = "http://192.168.1.59:1234/v1"
LM_STUDIO_MODEL    = "google/gemma-4-12b-qat"
DOCS_DIR           = Path("data/invoices")

# ---------------------------------------------------------------------------
# Questions de test (identiques à rag_app.py pour pouvoir comparer)
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "Quel est le montant total TTC de la facture de Thomas Lefebvre ?",
    "Quels services ont été facturés au cabinet médical des Trois Fontaines ?",
    "Quelle est la facture la plus élevée et à quel client correspond-elle ?",
    "Quels clients ont bénéficié d'une formation informatique ?",
    "Quel est le mode de paiement utilisé par la boulangerie Dupain ?",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
MAP_PROMPT = ChatPromptTemplate.from_template(
    """Tu analyses une facture. Extrais uniquement les informations utiles pour répondre à la question.
Si ce document ne contient aucune information pertinente, réponds exactement : AUCUNE INFO PERTINENTE

Document :
{document}

Question : {question}

Réponse (courte, factuelle) :"""
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """Tu reçois les réponses partielles de {n_docs} factures analysées individuellement.
Synthétise ces réponses pour fournir une réponse finale complète et précise.
Ignore les réponses "AUCUNE INFO PERTINENTE".
Si aucune réponse partielle n'est pertinente, dis-le clairement.

Question initiale : {question}

Réponses partielles :
{map_results}

Réponse finale :"""
)


def load_documents():
    print("Chargement des documents...")
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  {len(docs)} documents chargés.")
    return docs


def build_chains():
    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0.0,
    )
    map_chain    = MAP_PROMPT    | llm | StrOutputParser()
    reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()
    return map_chain, reduce_chain


def map_reduce(question, docs, map_chain, reduce_chain):
    # --- MAP ---
    print(f"  Map : {len(docs)} appels LLM en cours...", end="", flush=True)
    map_results = []
    for doc in docs:
        result = map_chain.invoke({
            "document": doc.page_content,
            "question": question,
        })
        map_results.append(result)
        print(".", end="", flush=True)
    print()

    # --- REDUCE ---
    formatted = "\n\n".join(
        f"[Doc {i+1}] {r}" for i, r in enumerate(map_results)
    )
    answer = reduce_chain.invoke({
        "question":    question,
        "map_results": formatted,
        "n_docs":      len(docs),
    })
    return answer


def run_tests(docs, map_chain, reduce_chain):
    print("\n" + "=" * 70)
    print("TEST DU SYSTÈME RAG — MAP-REDUCE")
    print("=" * 70)
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        print("-" * 60)
        answer = map_reduce(question, docs, map_chain, reduce_chain)
        print(f"[R{i}] {answer}")
    print("\n" + "=" * 70)


def main():
    docs = load_documents()
    map_chain, reduce_chain = build_chains()
    run_tests(docs, map_chain, reduce_chain)


if __name__ == "__main__":
    main()
