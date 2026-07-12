"""
Outils disponibles pour l'agent RAG :

  search_documents — cherche dans le vectorstore local (factures)
  search_web       — fallback internet via DuckDuckGo

L'agent décide lui-même quel outil appeler selon la confiance retournée
par search_documents.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, DOCS_DIR, CHROMA_DIR, RETRIEVER_K, CONFIDENCE_THRESHOLD

# Vectorstore partagé — initialisé une seule fois via init_vectorstore()
_vectorstore: Chroma = None


def init_vectorstore() -> None:
    """Charge le vectorstore depuis le disque, ou le construit si absent."""
    global _vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if CHROMA_DIR.exists():
        print("Chargement du vectorstore existant...")
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        return

    print("Construction du vectorstore...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    _vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=str(CHROMA_DIR)
    )
    print(f"{len(chunks)} chunks indexés.")


@tool
def search_documents(query: str) -> str:
    """Recherche des informations dans la base documentaire locale (factures clients).
    Appelle cet outil EN PREMIER pour toute question.
    Retourne les passages pertinents, ou un signal de confiance insuffisante si
    la question ne concerne pas les documents locaux."""
    results = _vectorstore.similarity_search_with_score(query, k=RETRIEVER_K)

    if not results:
        return (
            "CONFIANCE INSUFFISANTE : aucun document trouvé. "
            "Utilise search_web pour répondre à cette question."
        )

    best_doc, best_score = results[0]

    if best_score < CONFIDENCE_THRESHOLD:
        return (
            f"CONFIANCE INSUFFISANTE (score={best_score:.2f} < seuil={CONFIDENCE_THRESHOLD}). "
            "Les documents locaux ne contiennent pas de réponse pertinente. "
            "Utilise search_web."
        )

    context = "\n\n---\n\n".join(
        f"[score={score:.2f}]\n{doc.page_content}"
        for doc, score in results
    )
    return (
        f"Documents pertinents trouvés (meilleur score={best_score:.2f} ≥ seuil={CONFIDENCE_THRESHOLD}) :\n\n"
        f"{context}"
    )


@tool
def search_web(query: str) -> str:
    """Recherche des informations sur internet via DuckDuckGo.
    Utilise cet outil UNIQUEMENT si search_documents a retourné une confiance insuffisante."""
    return DuckDuckGoSearchRun().run(query)
