"""
Évaluateur RAG — RAGAS + métriques de performance
Dépendances : pip install "ragas==0.1.21" datasets

LLM juge : gemma4 via Ollama
Métriques : faithfulness, answer_relevancy, context_precision, context_recall
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Les chemins relatifs du RAG (DOCS_DIR, CHROMA_DIR) sont définis depuis ollama/.
# On change le CWD avant les imports pour que os.chdir soit actif dès le chargement.
os.chdir(Path(__file__).parent.parent)

# Permet d'importer le RAG depuis le dossier parent (ollama/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_app_adaptive_map_reduce_parallel import (
    load_documents,
    build_vectorstore,
    build_semantic_chunks,
    build_retrievers,
    build_chains,
    classify,
    answer,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

EVAL_MODEL     = "gemma4:latest"
EVAL_DIR       = Path(__file__).parent.resolve()
GROUND_TRUTH_F = EVAL_DIR / "ground_truth.json"
RESULTS_DIR    = EVAL_DIR / "results"


def load_ground_truth():
    with open(GROUND_TRUTH_F, encoding="utf-8") as f:
        return json.load(f)


def collect_rag_result(
    question, local_retriever, global_retriever,
    classifier_chain, rag_chain, map_chain, reduce_chain,
):
    """
    Appelle le RAG pour une question.
    classify + retrieve avant answer() pour capturer les contextes (RAGAS).
    """
    route = classify(question, classifier_chain)
    docs  = (
        local_retriever.invoke(question)
        if route == "LOCAL"
        else global_retriever(question)
    )

    result, perf = answer(
        question,
        local_retriever, global_retriever,
        classifier_chain, rag_chain, map_chain, reduce_chain,
    )

    return {
        "route":    perf["route"],
        "contexts": [d.page_content for d in docs],
        "answer":   result,
        "latency":  perf["latency_total"],
        "perf":     perf,
    }


def print_table(ground_truth_items, rag_results, scores_df):
    print("\n" + "=" * 95)
    print("RÉSULTATS D'ÉVALUATION RAG")
    print("=" * 95)
    print(
        f"{'#':<3} {'Route':<7} {'Latence':>8}  "
        f"{'Faithful':>8}  {'AnsRel':>6}  {'CtxPrec':>7}  {'CtxRec':>6}"
    )
    print("-" * 95)

    for i, (gt, res) in enumerate(zip(ground_truth_items, rag_results)):
        row = scores_df.iloc[i]
        print(
            f"Q{i+1:<2} {res['route']:<7} {res['latency']:>7.2f}s  "
            f"{row.get('faithfulness',        float('nan')):>8.3f}  "
            f"{row.get('answer_relevancy',    float('nan')):>6.3f}  "
            f"{row.get('context_precision',   float('nan')):>7.3f}  "
            f"{row.get('context_recall',      float('nan')):>6.3f}"
        )

    print("-" * 95)
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    avail = [c for c in cols if c in scores_df.columns]
    means = scores_df[avail].mean()
    print(
        f"{'Moy':<3} {'':>7} {'':>9}  "
        f"{means.get('faithfulness',        float('nan')):>8.3f}  "
        f"{means.get('answer_relevancy',    float('nan')):>6.3f}  "
        f"{means.get('context_precision',   float('nan')):>7.3f}  "
        f"{means.get('context_recall',      float('nan')):>6.3f}"
    )
    print("=" * 95)


def main():
    ground_truth = load_ground_truth()

    # --- Initialisation du RAG ---
    print("Initialisation du RAG...")
    docs           = load_documents()
    vectorstore    = build_vectorstore(docs)
    summary_chunks = build_semantic_chunks(docs)
    local_retriever, global_retriever = build_retrievers(vectorstore, summary_chunks)
    classifier_chain, rag_chain, map_chain, reduce_chain = build_chains(vectorstore)

    # --- Exécution des questions ---
    print("\nExécution des questions de test...")
    rag_results = []
    for i, gt in enumerate(ground_truth, 1):
        print(f"\n{'='*65}\n[Q{i}] {gt['question']}")
        res = collect_rag_result(
            gt["question"],
            local_retriever, global_retriever,
            classifier_chain, rag_chain, map_chain, reduce_chain,
        )
        preview = res["answer"][:120].replace("\n", " ")
        print(f"\n  → Réponse : {preview}{'...' if len(res['answer']) > 120 else ''}")
        rag_results.append(res)

    # --- Évaluation RAGAS ---
    print("\n\nÉvaluation RAGAS (juge : gemma4)...")
    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        base_url=OLLAMA_BASE_URL, api_key="ollama",
        model=EVAL_MODEL, temperature=0.0,
    ))
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    dataset = Dataset.from_dict({
        "question":     [gt["question"]     for gt in ground_truth],
        "answer":       [r["answer"]        for r in rag_results],
        "contexts":     [r["contexts"]      for r in rag_results],
        "ground_truth": [gt["ground_truth"] for gt in ground_truth],
    })

    # RAGAS 0.1.x : singletons configurés via .llm / .embeddings
    faithfulness.llm        = eval_llm
    answer_relevancy.llm        = eval_llm
    answer_relevancy.embeddings = eval_embeddings
    context_precision.llm   = eval_llm
    context_recall.llm      = eval_llm

    result    = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    scores_df = result.to_pandas()

    print_table(ground_truth, rag_results, scores_df)

    # --- Sauvegarde JSON ---
    RESULTS_DIR.mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = RESULTS_DIR / f"{ts}.json"

    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    output = {
        "timestamp":  datetime.now().isoformat(),
        "eval_model": EVAL_MODEL,
        "questions": [
            {
                "id":                 f"Q{i+1}",
                "question":           gt["question"],
                "ground_truth":       gt["ground_truth"],
                "relevant_docs":      gt.get("relevant_docs", []),
                "route":              r["route"],
                "latency_s":          r["latency"],
                "answer":             r["answer"],
                "faithfulness":       float(scores_df.iloc[i].get("faithfulness",      0) or 0),
                "answer_relevancy":   float(scores_df.iloc[i].get("answer_relevancy",  0) or 0),
                "context_precision":  float(scores_df.iloc[i].get("context_precision", 0) or 0),
                "context_recall":     float(scores_df.iloc[i].get("context_recall",    0) or 0),
            }
            for i, (gt, r) in enumerate(zip(ground_truth, rag_results))
        ],
        "averages": {
            col: float(scores_df[col].mean())
            for col in cols
            if col in scores_df.columns
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés → {out_path}\n")


if __name__ == "__main__":
    main()
