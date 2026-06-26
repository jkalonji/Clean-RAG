"""
RAG Adaptatif + Chunking Sémantique + Map-Reduce
- Chunking : 1 chunk résumé + chunks par section logique de la facture
- Métadonnées extraites : facture_id, client, total_ttc, type_client, date, mode_paiement
- Routage LLM : LOCAL → Chroma, GLOBAL → Map-Reduce
"""

import re
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
DOCS_DIR           = Path("data/invoices")
CHROMA_DIR         = Path("data/chroma_db_semantic")
RETRIEVER_K        = 4

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
# Extraction de métadonnées
# ---------------------------------------------------------------------------
def _clean_amount(raw: str) -> float:
    """Convertit '1 344,00' ou '27 240,00' en float."""
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

    # Client : première ligne non vide après le bloc CLIENT
    client_block = re.search(
        r"CLIENT(?:\s*\(ENTREPRISE\))?\s*\n-+\n(.*?)(?:\n-{10,}|\Z)",
        text,
        re.DOTALL,
    )
    if client_block:
        lines = [l.strip() for l in client_block.group(1).splitlines() if l.strip()]
        meta["client"] = lines[0] if lines else ""
    else:
        meta["client"] = ""

    meta["type_client"] = (
        "entreprise" if "CLIENT (ENTREPRISE)" in text else "particulier"
    )

    # Total TTC : ligne "| TOTAL TTC   |   ...   |"
    m = re.search(r"TOTAL TTC\s*\|?\s*([\d\s]+,\d+)", text)
    meta["total_ttc"] = _clean_amount(m.group(1)) if m else 0.0

    # Mode de paiement
    m = re.search(r"Mode de paiement\s*:\s*(.+)", text)
    meta["mode_paiement"] = m.group(1).strip() if m else ""

    # Objet
    m = re.search(r"OBJET\s*:\s*(.+)", text)
    meta["objet"] = m.group(1).strip() if m else ""

    return meta


# ---------------------------------------------------------------------------
# Chunking sémantique
# ---------------------------------------------------------------------------
def make_summary_chunk(meta: dict) -> str:
    """Chunk dense réunissant tous les faits clés — résout les questions précises."""
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

    chunks = []

    # 1. Chunk résumé (toujours en premier)
    chunks.append(Document(
        page_content=make_summary_chunk(meta),
        metadata={**meta, "section": "resume"},
    ))

    # 2. Chunks par section (séparés par les lignes ---)
    sections = re.split(r"-{20,}", text)
    for i, section in enumerate(sections):
        section = section.strip()
        # Ignorer les blocs trop courts (séparateurs vides, en-têtes seuls)
        if len(section) < 40:
            continue
        # Ignorer le bloc d'en-tête de l'entreprise (peu utile pour le retrieval)
        if "TECHSOLUTIONS SARL" in section and "SIRET" in section and len(section) < 300:
            continue
        chunks.append(Document(
            page_content=section,
            metadata={**meta, "section": f"section_{i}"},
        ))

    return chunks


def build_semantic_chunks(docs: list[Document]) -> list[Document]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(split_invoice(doc))
    return all_chunks


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
    print("Chunking sémantique et indexation...")
    chunks = build_semantic_chunks(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  {len(chunks)} chunks créés ({len(docs)} résumés + sections logiques).")
    print(f"  Vectorstore persisté dans : {CHROMA_DIR}")
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

    def format_docs(retrieved_docs):
        return "\n\n---\n\n".join(d.page_content for d in retrieved_docs)

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
    return "LOCAL" if "LOCAL" in raw else "GLOBAL"


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
    print("TEST RAG — ADAPTATIF + CHUNKING SÉMANTIQUE + MAP-REDUCE")
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
