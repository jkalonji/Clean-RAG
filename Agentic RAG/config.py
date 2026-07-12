from pathlib import Path

# --- LMStudio ---
LM_STUDIO_BASE_URL   = "http://192.168.1.59:1234/v1"
LM_STUDIO_MODEL      = "google/gemma-4-12b-qat"

# --- Groq ---
GROQ_MODEL           = "llama-3.3-70b-versatile"

# --- Embeddings & données ---
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR             = Path(__file__).parent.parent / "LMStudio" / "data" / "invoices"
CHROMA_DIR           = Path(__file__).parent / "data" / "chroma_db"
RETRIEVER_K          = 4

# Seuil de confiance vectorstore (similarité : 1 = identique, 0 = aucune similarité)
# Si le meilleur score < CONFIDENCE_THRESHOLD → les docs locaux ne sont pas pertinents → web
CONFIDENCE_THRESHOLD = 0.5
