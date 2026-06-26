"""
RAG sur factures TechSolutions — LM Studio + HuggingFace embeddings + Chroma
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------------------------
# Configuration — adapter si besoin
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL = "http://192.168.1.59:1234/v1"
LM_STUDIO_MODEL    = "google/gemma-4-12b-qat"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR           = Path("data/invoices")
CHROMA_DIR         = Path("data/chroma_db")
RETRIEVER_K        = 3   # nombre de chunks retournés par la recherche

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


def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("Découpage des documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks créés.")
    return chunks


def build_vectorstore(chunks):
    print("Création des embeddings et indexation dans Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  Vectorstore persisté dans : {CHROMA_DIR}")
    return vectorstore


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",          # LM Studio n'exige pas de vraie clé
        model=LM_STUDIO_MODEL,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_template(
        """Tu es un assistant spécialisé dans l'analyse de factures.
Réponds en français, de façon concise et précise, en te basant uniquement sur le contexte fourni.
Si l'information n'est pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}"""
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_tests(chain):
    print("\n" + "=" * 70)
    print("TEST DU SYSTÈME RAG")
    print("=" * 70)
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        print("-" * 60)
        answer = chain.invoke(question)
        print(f"[R{i}] {answer}")
    print("\n" + "=" * 70)


def main():
    docs   = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain  = build_rag_chain(vectorstore)
    run_tests(chain)


if __name__ == "__main__":
    main()
