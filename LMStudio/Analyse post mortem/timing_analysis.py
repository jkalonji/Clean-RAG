"""
Analyse détaillée des temps d'exécution — RAG Adaptatif + Map-Reduce
Chaque étape est chronométrée individuellement pour identifier les goulots.
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
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL = "http://192.168.1.59:1234/v1"
LM_STUDIO_MODEL    = "google/gemma-4-12b-qat"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR           = Path("../data/invoices")
CHROMA_DIR         = Path("../data/chroma_db_semantic")
RETRIEVER_K        = 4

TEST_QUESTIONS = [
    "Quel est le montant total TTC de la facture de Thomas Lefebvre ?",
    "Quels services ont été facturés au cabinet médical des Trois Fontaines ?",
    "Quelle est la facture la plus élevée et à quel client correspond-elle ?",
    "Quels clients ont bénéficié d'une formation informatique ?",
    "Quel est le mode de paiement utilisé par la boulangerie Dupain ?",
]

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
_timings: dict[str, list[float]] = {}

def record(label: str, duration: float):
    _timings.setdefault(label, []).append(duration)
    print(f"    ⏱  {label:<35} {duration:.3f}s")


def timed(label: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            record(label, time.perf_counter() - t0)
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Extraction de métadonnées (inchangée)
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

    # Émetteur : bloc entre les deux premières lignes ===
    emetteur_block = re.search(r"={20,}\n(.*?)\n={20,}", text, re.DOTALL)
    if emetteur_block:
        elines = [l.strip() for l in emetteur_block.group(1).splitlines() if l.strip()]
        meta["emetteur_nom"]    = elines[0] if len(elines) > 0 else ""
        meta["emetteur_adresse"] = elines[1] if len(elines) > 1 else ""
        m_siret = re.search(r"SIRET\s*:\s*([\d\s]+)", emetteur_block.group(1))
        meta["emetteur_siret"] = m_siret.group(1).strip() if m_siret else ""
    else:
        meta["emetteur_nom"] = meta["emetteur_adresse"] = meta["emetteur_siret"] = ""

    # Client : bloc entre les séparateurs ---
    client_block = re.search(
        r"CLIENT(?:\s*\(ENTREPRISE\))?\s*\n-+\n(.*?)(?:\n-{10,}|\Z)", text, re.DOTALL
    )
    if client_block:
        block = client_block.group(1)
        clines = [l.strip() for l in block.splitlines() if l.strip()]
        meta["client"] = clines[0] if clines else ""

        addr_lines = []
        for l in clines[1:]:
            if l.startswith("Email") or l.startswith("Tél"):
                break
            addr_lines.append(l)
        meta["client_adresse"] = ", ".join(addr_lines)

        m_email = re.search(r"Email\s*:\s*(.+)", block)
        meta["client_email"] = m_email.group(1).strip() if m_email else ""

        m_tel = re.search(r"Tél\s*:\s*(.+)", block)
        meta["client_tel"] = m_tel.group(1).strip() if m_tel else ""
    else:
        meta["client"] = meta["client_adresse"] = meta["client_email"] = meta["client_tel"] = ""

    meta["type_client"] = "entreprise" if "CLIENT (ENTREPRISE)" in text else "particulier"

    m = re.search(r"TOTAL TTC\s*\|?\s*([\d\s]+,\d+)", text)
    meta["total_ttc"] = _clean_amount(m.group(1)) if m else 0.0

    m = re.search(r"Mode de paiement\s*:\s*(.+)", text)
    meta["mode_paiement"] = m.group(1).strip() if m else ""

    m = re.search(r"OBJET\s*:\s*(.+)", text)
    meta["objet"] = m.group(1).strip() if m else ""

    # Lignes de facturation : lignes de tableau hors en-tête et totaux
    lignes = re.findall(
        r"\|\s+([^|]+?)\s+\|\s+(\d+)\s+\|\s+([\d\s,]+)\s+\|\s+([\d\s,]+)\s+\|",
        text,
    )
    meta["lignes"] = [
        {"desc": d.strip(), "qte": q.strip(), "pu": pu.strip(), "total": tot.strip()}
        for d, q, pu, tot in lignes
        if not d.strip().lower().startswith("description")
    ]

    return meta


# ---------------------------------------------------------------------------
# Chunking sémantique — résumés uniquement
# ---------------------------------------------------------------------------
def make_summary_chunk(meta: dict) -> str:
    lignes_str = "\n".join(
        f"  - {l['desc']} × {l['qte']} → {l['total']} €"
        for l in meta.get("lignes", [])
    )
    return (
        f"RÉSUMÉ FACTURE {meta['facture_id']}\n"
        f"Émetteur : {meta['emetteur_nom']} | {meta['emetteur_adresse']} | SIRET : {meta['emetteur_siret']}\n"
        f"Client : {meta['client']} ({meta['type_client']})\n"
        f"  Adresse : {meta['client_adresse']}\n"
        f"  Email : {meta['client_email']} | Tél : {meta['client_tel']}\n"
        f"Date : {meta['date']} | Échéance : {meta['echeance']}\n"
        f"Objet : {meta['objet']}\n"
        f"Lignes :\n{lignes_str}\n"
        f"Total TTC : {meta['total_ttc']:.2f} €\n"
        f"Mode de paiement : {meta['mode_paiement']}"
    )


def split_invoice(doc: Document) -> list[Document]:
    text   = doc.page_content
    source = doc.metadata.get("source", "")
    meta   = extract_metadata(text, source)
    chroma_meta = {k: v for k, v in meta.items() if k != "lignes"}
    return [Document(page_content=make_summary_chunk(meta), metadata={**chroma_meta, "section": "resume"})]


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

MAP_PROMPT = ChatPromptTemplate.from_template(
    """Tu analyses une facture. Extrais uniquement les informations utiles pour répondre à la question.
Si ce document ne contient aucune information pertinente, réponds exactement : AUCUNE INFO PERTINENTE

Document :
{document}

Question : {question}

Réponse (courte, factuelle) :"""
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
@timed("chargement documents")
def load_documents():
    print("  Chargement des documents...")
    loader = DirectoryLoader(
        str(DOCS_DIR), glob="*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  {len(docs)} documents chargés.")
    return docs


@timed("build vectorstore")
def build_vectorstore(docs):
    print("  Chunking sémantique et indexation...")
    chunks = build_semantic_chunks(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(CHROMA_DIR),
    )
    print(f"  {len(chunks)} chunks résumé indexés.")
    return vectorstore


def build_llm():
    return ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL, api_key="lm-studio",
        model=LM_STUDIO_MODEL, temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Étapes chronométrées individuellement
# ---------------------------------------------------------------------------
def classify(question: str, llm) -> str:
    t0 = time.perf_counter()
    raw = (CLASSIFIER_PROMPT | llm | StrOutputParser()).invoke({"question": question}).strip().upper()
    record("classification LLM", time.perf_counter() - t0)
    return "LOCAL" if "LOCAL" in raw else "GLOBAL"


def retrieve(question: str, retriever) -> list[Document]:
    t0 = time.perf_counter()
    docs = retriever.invoke(question)
    record("retrieval Chroma", time.perf_counter() - t0)
    return docs


def generate_local(question: str, context: str, llm) -> str:
    t0 = time.perf_counter()
    result = (RAG_PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": question})
    record("génération LLM (LOCAL)", time.perf_counter() - t0)
    return result


def run_stuffing(question: str, summary_chunks: list[Document], llm) -> str:
    context = "\n\n---\n\n".join(c.page_content for c in summary_chunks)
    t0 = time.perf_counter()
    result = (STUFFING_PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": question})
    record(f"génération LLM GLOBAL (stuffing, {len(summary_chunks)} docs)", time.perf_counter() - t0)
    return result


def answer(question: str, summary_chunks: list[Document], retriever, llm) -> str:
    t_total = time.perf_counter()

    route = classify(question, llm)
    print(f"    → ROUTE : {route}")

    if route == "LOCAL":
        retrieved = retrieve(question, retriever)
        context   = "\n\n---\n\n".join(d.page_content for d in retrieved)
        result    = generate_local(question, context, llm)
    else:
        result = run_stuffing(question, summary_chunks, llm)

    record("TOTAL answer()", time.perf_counter() - t_total)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests(summary_chunks, retriever, llm):
    print("\n" + "=" * 70)
    print("ANALYSE TIMING — RAG ADAPTATIF + CHUNKING RÉSUMÉ + MAP-REDUCE")
    print("=" * 70)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        print("-" * 60)
        result = answer(question, summary_chunks, retriever, llm)
        print(f"[R{i}] {result}")

    print("\n" + "=" * 70)
    print("RÉCAPITULATIF DES TEMPS (toutes questions confondues)")
    print("=" * 70)
    for label, values in _timings.items():
        if len(values) > 1:
            print(f"  {label:<40} total={sum(values):.2f}s  moy={sum(values)/len(values):.2f}s  n={len(values)}")
        else:
            print(f"  {label:<40} {values[0]:.2f}s")
    print("=" * 70)


def main():
    print("\n--- INITIALISATION ---")
    docs           = load_documents()
    vectorstore    = build_vectorstore(docs)
    summary_chunks = build_semantic_chunks(docs)
    llm            = build_llm()
    retriever      = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    run_tests(summary_chunks, retriever, llm)


if __name__ == "__main__":
    main()
