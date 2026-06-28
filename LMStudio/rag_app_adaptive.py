"""
RAG Adaptatif — LLM classifier + routage RAG standard / Map-Reduce
LOCAL  → Chroma + similarité (rapide)
GLOBAL → Map-Reduce sur tous les documents (exhaustif)
Défaut : GLOBAL si la classification est incertaine
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL = "http://192.168.1.59:1234/v1"
LM_STUDIO_MODEL    = "google/gemma-4-12b-qat"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR           = Path("data/invoices")
CHROMA_DIR         = Path("data/chroma_db_adaptive")
RETRIEVER_K        = 3

# ---------------------------------------------------------------------------
# Questions de test
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
CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """Tu dois classifier une question en deux catégories :

LOCAL  — la réponse se trouve dans UN SEUL document
         (ex: montant d'une facture précise, services d'un client nommé, mode de paiement d'une entreprise)

GLOBAL — la réponse nécessite de parcourir TOUS les documents
         (ex: comparaisons, agrégations, listes de plusieurs clients, maximum/minimum)

Réponds UNIQUEMENT par le mot LOCAL ou GLOBAL, sans explication.

Question : {question}"""
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un assistant spécialisé dans l'analyse de factures.
Réponds en français, de façon concise et précise, en te basant uniquement sur le contexte fourni.
Si l'information n'est pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}"""
)

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


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
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


def build_vectorstore(docs):
    print("Construction du vectorstore...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  {len(chunks)} chunks indexés.")
    return vectorstore


def build_chains(vectorstore):
    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        model=LM_STUDIO_MODEL,
        temperature=0.0,
    )

    classifier_chain = CLASSIFIER_PROMPT | llm | StrOutputParser()

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    map_chain    = MAP_PROMPT    | llm | StrOutputParser()
    reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()

    return classifier_chain, rag_chain, map_chain, reduce_chain


# ---------------------------------------------------------------------------
# Routage
# ---------------------------------------------------------------------------
def classify(question, classifier_chain):
    raw = classifier_chain.invoke({"question": question}).strip().upper()
    if "LOCAL" in raw:
        return "LOCAL"
    return "GLOBAL"  # défaut sûr


def run_map_reduce(question, docs, map_chain, reduce_chain):
    print(f"  Map ({len(docs)} docs)...", end="", flush=True)
    map_results = []
    for doc in docs:
        result = map_chain.invoke({"document": doc.page_content, "question": question})
        map_results.append(result)
        print(".", end="", flush=True)
    print()
    formatted = "\n\n".join(f"[Doc {i+1}] {r}" for i, r in enumerate(map_results))
    return reduce_chain.invoke({
        "question":    question,
        "map_results": formatted,
        "n_docs":      len(docs),
    })


def answer(question, docs, classifier_chain, rag_chain, map_chain, reduce_chain):
    route = classify(question, classifier_chain)
    print(f"  [ROUTE → {route}]")
    if route == "LOCAL":
        return rag_chain.invoke(question)
    return run_map_reduce(question, docs, map_chain, reduce_chain)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests(docs, classifier_chain, rag_chain, map_chain, reduce_chain):
    print("\n" + "=" * 70)
    print("TEST DU SYSTÈME RAG — ADAPTATIF (LLM Classifier)")
    print("=" * 70)
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        print("-" * 60)
        result = answer(question, docs, classifier_chain, rag_chain, map_chain, reduce_chain)
        print(f"[R{i}] {result}")
    print("\n" + "=" * 70)


def main():
    docs        = load_documents()
    vectorstore = build_vectorstore(docs)
    classifier_chain, rag_chain, map_chain, reduce_chain = build_chains(vectorstore)
    run_tests(docs, classifier_chain, rag_chain, map_chain, reduce_chain)


if __name__ == "__main__":
    main()
