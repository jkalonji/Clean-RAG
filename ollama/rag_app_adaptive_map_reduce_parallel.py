"""
RAG Adaptatif + Chunking Sémantique + Map-Reduce Parallèle — version Ollama
- Retrieval LOCAL  : cosine similarity (Chroma, K=4)
- Retrieval GLOBAL : hybride BM25 + cosine (EnsembleRetriever, K=20)
- GLOBAL route     : map-reduce avec appels LLM parallèles (asyncio + Semaphore)

Dépendance supplémentaire : pip install rank_bm25
"""

import asyncio
import re
import time
from functools import wraps
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL    = "http://localhost:11434/v1"
OLLAMA_MODEL       = "gemma4:latest"   # reduce + rag local
CLASSIFIER_MODEL   = "gemma3:4b"       # routing LOCAL/GLOBAL
MAP_MODEL          = "gemma3:4b"       # extraction factuelle pendant le map
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR           = Path("../LMStudio/data/invoices")
CHROMA_DIR         = Path("data/chroma_db_mapreduce")
LOCAL_RETRIEVER_K  = 4
GLOBAL_RETRIEVER_K = 20   # nb de docs passés au map (cosine + BM25 fusionnés)
MAX_CONCURRENT     = 4    # appels LLM simultanés — doit correspondre à OLLAMA_NUM_PARALLEL

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


GLOBAL — La question concerne une collection de documents, ou d'entités référencées dans les documents. 
        la réponse nécessite donc de parcourir plusieurs documents pour faire une comparaison, un calcul, ou une vérification au travers de plusiiers documents et leurs contenus.
         (ex: faire des comparaisons, agrégations, lister plusieurs clients, trouver un maximum/minimum)
         exemples : "Quel est le montant total de toutes les factures ?", "Quels clients ont bénéficié d'une formation informatique ?", "Quelle est la facture la plus élevée et à quel client correspond-elle ?"

LOCAL  — la réponse se trouve dans UN SEUL document ou chunk de document.
         (ex: montant d'une facture précise, services d'un client nommé, mode de paiement)
         
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
    print(f"  {len(chunks)} chunks résumé indexés dans {CHROMA_DIR}.")
    return vectorstore


def _rrf_fusion(results_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion — fusionne plusieurs listes de résultats en un seul ranking."""
    scores: dict[str, dict] = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1.0 / (k + rank + 1)
    return [v["doc"] for v in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]


def build_retrievers(vectorstore, summary_chunks):
    # LOCAL : cosine similarity seule (précision sur 1 doc)
    local_retriever = vectorstore.as_retriever(
        search_kwargs={"k": LOCAL_RETRIEVER_K}
    )

    # GLOBAL : hybride BM25 (mots-clés) + cosine (sens), fusionnés par RRF
    bm25 = BM25Retriever.from_documents(summary_chunks)
    bm25.k = GLOBAL_RETRIEVER_K

    chroma_global = vectorstore.as_retriever(
        search_kwargs={"k": GLOBAL_RETRIEVER_K}
    )

    def global_retriever(question: str) -> list[Document]:
        return _rrf_fusion([
            bm25.invoke(question),
            chroma_global.invoke(question),
        ])

    return local_retriever, global_retriever


def build_chains(vectorstore):
    classifier_llm = ChatOpenAI(
        base_url=OLLAMA_BASE_URL, api_key="ollama",
        model=CLASSIFIER_MODEL, temperature=0.0,
    )
    map_llm = ChatOpenAI(
        base_url=OLLAMA_BASE_URL, api_key="ollama",
        model=MAP_MODEL, temperature=0.0,
    )
    reduce_llm = ChatOpenAI(
        base_url=OLLAMA_BASE_URL, api_key="ollama",
        model=OLLAMA_MODEL, temperature=0.0,
    )
    classifier_chain = CLASSIFIER_PROMPT | classifier_llm | StrOutputParser()
    rag_chain        = RAG_PROMPT        | reduce_llm     | StrOutputParser()
    map_chain        = MAP_PROMPT        | map_llm        | StrOutputParser()
    reduce_chain     = REDUCE_PROMPT     | reduce_llm     | StrOutputParser()
    return classifier_chain, rag_chain, map_chain, reduce_chain


# ---------------------------------------------------------------------------
# Routage
# ---------------------------------------------------------------------------
def classify(question, classifier_chain):
    raw = classifier_chain.invoke({"question": question}).strip().upper()
    return "LOCAL" if "LOCAL" in raw else "GLOBAL"


async def _run_map_parallel(question, docs, map_chain):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    t0 = time.perf_counter()

    async def map_one(doc, idx):
        async with sem:
            t = time.perf_counter()
            result = await map_chain.ainvoke({
                "document": doc.page_content,
                "question": question,
            })
            print(f"    doc[{idx+1:02d}] terminé à +{time.perf_counter()-t0:.2f}s (durée : {time.perf_counter()-t:.2f}s)")
            return result

    print(f"  Map ({len(docs)} docs, concurrence={MAX_CONCURRENT})...")
    results = await asyncio.gather(*[map_one(doc, i) for i, doc in enumerate(docs)])
    print(f"  Map total : {time.perf_counter() - t0:.2f}s")
    return list(results)


@timeit
def answer(question, local_retriever, global_retriever, classifier_chain, rag_chain, map_chain, reduce_chain):
    route = classify(question, classifier_chain)
    print(f"  [ROUTE → {route}]")

    if route == "LOCAL":
        retrieved = local_retriever.invoke(question)
        context   = "\n\n---\n\n".join(d.page_content for d in retrieved)
        return rag_chain.invoke({"context": context, "question": question})

    # GLOBAL : retrieval hybride + map-reduce parallèle
    retrieved = global_retriever(question)
    print(f"  Docs récupérés (hybride BM25+cosine) : {len(retrieved)}")

    map_results = asyncio.run(_run_map_parallel(question, retrieved, map_chain))

    relevant   = [r for r in map_results if "AUCUNE INFO PERTINENTE" not in r]
    formatted  = "\n\n".join(f"[Doc {i+1}] {r}" for i, r in enumerate(relevant))

    t = time.perf_counter()
    result = reduce_chain.invoke({
        "question":    question,
        "map_results": formatted,
        "n_docs":      len(retrieved),
    })
    print(f"  Reduce : {time.perf_counter() - t:.2f}s")
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests(local_retriever, global_retriever, classifier_chain, rag_chain, map_chain, reduce_chain):
    print("\n" + "=" * 70)
    print("TEST RAG — ADAPTATIF + CHUNKING RÉSUMÉ + MAP-REDUCE PARALLÈLE  [Ollama]")
    print("=" * 70)
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        print("-" * 60)
        result = answer(
            question,
            local_retriever, global_retriever,
            classifier_chain, rag_chain, map_chain, reduce_chain,
        )
        print(f"[R{i}] {result}")
    print("\n" + "=" * 70)


def main():
    docs           = load_documents()
    vectorstore    = build_vectorstore(docs)
    summary_chunks = build_semantic_chunks(docs)

    local_retriever, global_retriever = build_retrievers(vectorstore, summary_chunks)
    classifier_chain, rag_chain, map_chain, reduce_chain = build_chains(vectorstore)

    run_tests(local_retriever, global_retriever, classifier_chain, rag_chain, map_chain, reduce_chain)


if __name__ == "__main__":
    main()
