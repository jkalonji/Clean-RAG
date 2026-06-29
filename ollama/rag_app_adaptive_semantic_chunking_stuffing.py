"""
RAG Adaptatif + Chunking Sémantique + Stuffing — version Ollama
- Chunking : 1 chunk résumé par facture (dense, ~100 tokens)
- Routage LLM : LOCAL → Chroma retrieval, GLOBAL → Stuffing (1 appel LLM)
- Métadonnées extraites : facture_id, client, total_ttc, type_client, date, mode_paiement
"""

import re
import time
from functools import wraps
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL    = "gemma4:latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR        = Path("../LMStudio/data/invoices")
CHROMA_DIR      = Path("data/chroma_db")
RETRIEVER_K     = 4

TEST_QUESTIONS = [
    "Quel est le montant total TTC de la facture de Thomas Lefebvre ?",
    "Quels services ont été facturés au cabinet médical des Trois Fontaines ?",
    "Quelle est la facture la plus élevée et à quel client correspond-elle ?",
    "Quels clients ont bénéficié d'une formation informatique ?",
    "Quel est le mode de paiement utilisé par la boulangerie Dupain ?",
]

# ---------------------------------------------------------------------------
# Décorateur de mesure de temps
# ---------------------------------------------------------------------------
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"  [{func.__name__}] durée : {time.perf_counter() - start:.2f}s")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Extraction de métadonnées
# ---------------------------------------------------------------------------
def _clean_amount(raw: str) -> float:
    try:
        return float(raw.replace(" ", "").replace(",", "."))
    except ValueError:
        return 0.0


def extract_metadata(text: str, source: str) -> dict:
    meta = {"source": source}

    m = re.search(r"FACTURE N°\s+(TS-\d{4}-\d+)", text)
    meta["facture_id"] = m.group(1) if m else ""

    m = re.search(r"Date\s*:\s*(\d{2}/\d{2}/\d{4})", text)
    meta["date"] = m.group(1) if m else ""

    m = re.search(r"Échéance\s*:\s*(\d{2}/\d{2}/\d{4})", text)
    meta["echeance"] = m.group(1) if m else ""

    client_block = re.search(
        r"CLIENT(?:\s*\(ENTREPRISE\))?\s*\n-+\n(.*?)(?:\n-{10,}|\Z)", text, re.DOTALL
    )
    if client_block:
        lines = [l.strip() for l in client_block.group(1).splitlines() if l.strip()]
        meta["client"] = lines[0] if lines else ""
    else:
        meta["client"] = ""

    meta["type_client"] = "entreprise" if "CLIENT (ENTREPRISE)" in text else "particulier"

    m = re.search(r"TOTAL TTC\s*\|?\s*([\d\s]+,\d+)", text)
    meta["total_ttc"] = _clean_amount(m.group(1)) if m else 0.0

    m = re.search(r"Mode de paiement\s*:\s*(.+)", text)
    meta["mode_paiement"] = m.group(1).strip() if m else ""

    m = re.search(r"OBJET\s*:\s*(.+)", text)
    meta["objet"] = m.group(1).strip() if m else ""

    return meta


# ---------------------------------------------------------------------------
# Chunking sémantique — résumés uniquement
# ---------------------------------------------------------------------------
def make_summary_chunk(meta: dict) -> str:
    return (
        f"RÉSUMÉ FACTURE {meta['facture_id']}\n"
        f"Client : {meta['client']} ({meta['type_client']})\n"
        f"Date : {meta['date']} | Échéance : {meta['echeance']}\n"
        f"Objet : {meta['objet']}\n"
        f"Total TTC : {meta['total_ttc']:.2f} €\n"
        f"Mode de paiement : {meta['mode_paiement']}"
    )


def split_invoice(doc: Document) -> list[Document]:
    text   = doc.page_content
    source = doc.metadata.get("source", "")
    meta   = extract_metadata(text, source)
    return [Document(page_content=make_summary_chunk(meta), metadata={**meta, "section": "resume"})]


def build_semantic_chunks(docs: list[Document]) -> list[Document]:
    return [chunk for doc in docs for chunk in split_invoice(doc)]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """Tu dois classifier une question en deux catégories :

LOCAL  — la réponse se trouve dans UN SEUL document
         (ex: montant d'une facture précise, services d'un client nommé, mode de paiement)

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

STUFFING_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un assistant spécialisé dans l'analyse de factures.
Réponds en français, de façon concise et précise, en te basant uniquement sur les résumés fournis.
Si l'information n'est pas dans le contexte, dis-le clairement.

Résumés de toutes les factures :
{context}

Question : {question}"""
)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def load_documents():
    print("Chargement des documents...")
    loader = DirectoryLoader(
        str(DOCS_DIR), glob="*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  {len(docs)} documents chargés.")
    return docs


def build_vectorstore(docs):
    print("Chunking sémantique et indexation...")
    chunks = build_semantic_chunks(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(CHROMA_DIR),
    )
    print(f"  {len(chunks)} chunks résumé indexés.")
    return vectorstore


def build_chains(vectorstore, model=OLLAMA_MODEL):
    llm = ChatOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        model=model,
        temperature=0.0,
    )
    classifier_chain = CLASSIFIER_PROMPT | llm | StrOutputParser()
    retriever        = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    rag_chain        = RAG_PROMPT | llm | StrOutputParser()
    stuffing_chain   = STUFFING_PROMPT | llm | StrOutputParser()
    return classifier_chain, retriever, rag_chain, stuffing_chain


# ---------------------------------------------------------------------------
# Routage
# ---------------------------------------------------------------------------
def classify(question, classifier_chain):
    raw = classifier_chain.invoke({"question": question}).strip().upper()
    return "LOCAL" if "LOCAL" in raw else "GLOBAL"


def answer(question, summary_chunks, classifier_chain, retriever, rag_chain, stuffing_chain):
    t_total = time.perf_counter()
    perf = {}

    t0    = time.perf_counter()
    route = classify(question, classifier_chain)
    perf["latency_classify"] = round(time.perf_counter() - t0, 3)
    print(f"  [ROUTE → {route}]")

    if route == "LOCAL":
        t0        = time.perf_counter()
        retrieved = retriever.invoke(question)
        perf["latency_retrieve"] = round(time.perf_counter() - t0, 3)
        perf["n_docs"]           = len(retrieved)
        perf["route"]            = "LOCAL"

        context = "\n\n---\n\n".join(d.page_content for d in retrieved)
        result  = rag_chain.invoke({"context": context, "question": question})

        perf["latency_total"] = round(time.perf_counter() - t_total, 3)
        print(f"  [answer] durée : {perf['latency_total']:.2f}s")
        return result, perf

    t0      = time.perf_counter()
    context = "\n\n---\n\n".join(c.page_content for c in summary_chunks)
    result  = stuffing_chain.invoke({"context": context, "question": question})
    perf["latency_retrieve"] = 0.0
    perf["n_docs"]           = len(summary_chunks)
    perf["route"]            = "GLOBAL"

    perf["latency_total"] = round(time.perf_counter() - t_total, 3)
    print(f"  [answer] durée : {perf['latency_total']:.2f}s")
    return result, perf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests(summary_chunks, classifier_chain, retriever, rag_chain, stuffing_chain):
    print("\n" + "=" * 70)
    print("TEST RAG — ADAPTATIF + CHUNKING RÉSUMÉ + STUFFING  [Ollama]")
    print("=" * 70)
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        print("-" * 60)
        result, perf = answer(question, summary_chunks, classifier_chain, retriever, rag_chain, stuffing_chain)
        print(f"[R{i}] {result}")
    print("\n" + "=" * 70)


def main():
    docs              = load_documents()
    vectorstore       = build_vectorstore(docs)
    summary_chunks    = build_semantic_chunks(docs)
    classifier_chain, retriever, rag_chain, stuffing_chain = build_chains(vectorstore)
    run_tests(summary_chunks, classifier_chain, retriever, rag_chain, stuffing_chain)


if __name__ == "__main__":
    main()
