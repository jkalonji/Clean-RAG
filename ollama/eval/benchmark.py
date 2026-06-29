"""
Benchmark RAG — multi-pipeline × multi-modèle
Usage : python eval/benchmark.py [--config eval/benchmark_config.json]

Pour chaque combinaison {pipeline, model} définie dans la config :
  1. Initialise le pipeline avec le modèle spécifié
  2. Exécute toutes les questions du ground_truth
  3. Évalue avec RAGAS (juge = judge_model de la config)
  4. Affiche un tableau comparatif + matrice de décision
  5. Sauvegarde les résultats dans eval/results/benchmark_{timestamp}.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from rag_app_adaptive_map_reduce_parallel import (
    load_documents,
    build_vectorstore,
    build_semantic_chunks,
    build_retrievers,
    build_chains as build_chains_mr,
    classify,
    answer as answer_mr,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
)
from rag_app_adaptive_semantic_chunking_stuffing import (
    build_chains as build_chains_st,
    answer as answer_st,
)

EVAL_DIR        = Path(__file__).parent.resolve()
GROUND_TRUTH_F  = EVAL_DIR / "ground_truth.json"
CONFIG_F        = EVAL_DIR / "benchmark_config.json"
RESULTS_DIR     = EVAL_DIR / "results"


def _unload_model(model: str) -> None:
    """Libère le modèle de la VRAM Ollama (keep_alive=0) pour éviter la saturation."""
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
        print(f"  [mem] '{model}' libéré de la VRAM")
    except Exception as e:
        print(f"  [mem] avertissement : impossible de libérer '{model}' : {e}")


# ---------------------------------------------------------------------------
# Collecte résultat pour une question — map_reduce
# ---------------------------------------------------------------------------
def _collect_mr(question, local_retriever, global_retriever, classifier_chain, rag_chain, map_chain, reduce_chain):
    route = classify(question, classifier_chain)
    docs  = local_retriever.invoke(question) if route == "LOCAL" else global_retriever(question)
    result, perf = answer_mr(
        question, local_retriever, global_retriever,
        classifier_chain, rag_chain, map_chain, reduce_chain,
    )
    return {"answer": result, "contexts": [d.page_content for d in docs], "perf": perf}


# ---------------------------------------------------------------------------
# Collecte résultat pour une question — stuffing
# ---------------------------------------------------------------------------
def _collect_st(question, summary_chunks, classifier_chain, retriever, rag_chain, stuffing_chain):
    from rag_app_adaptive_semantic_chunking_stuffing import classify as classify_st
    route = classify_st(question, classifier_chain)
    docs  = retriever.invoke(question) if route == "LOCAL" else summary_chunks
    result, perf = answer_st(
        question, summary_chunks, classifier_chain, retriever, rag_chain, stuffing_chain,
    )
    return {"answer": result, "contexts": [d.page_content for d in docs], "perf": perf}


# ---------------------------------------------------------------------------
# Exécution d'un run complet {pipeline, model}
# ---------------------------------------------------------------------------
def run_one(pipeline, model, ground_truth, docs, vectorstore, summary_chunks):
    print(f"\n{'='*70}")
    print(f"  PIPELINE: {pipeline}  |  MODÈLE: {model}")
    print(f"{'='*70}")

    if pipeline == "map_reduce":
        local_ret, global_ret = build_retrievers(vectorstore, summary_chunks)
        classifier_chain, rag_chain, map_chain, reduce_chain = build_chains_mr(
            vectorstore, model=model, map_model=model,
        )
        collect = lambda q: _collect_mr(q, local_ret, global_ret, classifier_chain, rag_chain, map_chain, reduce_chain)

    elif pipeline == "stuffing":
        classifier_chain, retriever, rag_chain, stuffing_chain = build_chains_st(
            vectorstore, model=model,
        )
        collect = lambda q: _collect_st(q, summary_chunks, classifier_chain, retriever, rag_chain, stuffing_chain)

    else:
        raise ValueError(f"Pipeline inconnu : {pipeline}")

    results = []
    for i, gt in enumerate(ground_truth, 1):
        print(f"\n[Q{i}] {gt['question']}")
        res = collect(gt["question"])
        preview = res["answer"][:100].replace("\n", " ")
        print(f"  → {preview}{'...' if len(res['answer']) > 100 else ''}")
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# Évaluation RAGAS pour un run
# ---------------------------------------------------------------------------
def ragas_eval(ground_truth, run_results, judge_model, embeddings_model):
    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        base_url=OLLAMA_BASE_URL, api_key="ollama",
        model=judge_model, temperature=0.0,
    ))
    eval_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=embeddings_model)
    )

    dataset = Dataset.from_dict({
        "question":     [gt["question"]     for gt in ground_truth],
        "answer":       [r["answer"]        for r in run_results],
        "contexts":     [r["contexts"]      for r in run_results],
        "ground_truth": [gt["ground_truth"] for gt in ground_truth],
    })

    faithfulness.llm            = eval_llm
    answer_relevancy.llm        = eval_llm
    answer_relevancy.embeddings = eval_emb
    context_precision.llm       = eval_llm
    context_recall.llm          = eval_llm

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    return result.to_pandas()


# ---------------------------------------------------------------------------
# Tableau comparatif ASCII
# ---------------------------------------------------------------------------
def _score(val):
    if val != val:  # NaN
        return "  N/A "
    return f"{val:.3f}"


def print_comparison_table(summary_rows):
    W = 90
    print("\n" + "=" * W)
    print("BENCHMARK RAG — RÉSULTATS COMPARATIFS")
    print("=" * W)
    print(f"{'Pipeline':<12} {'Modèle':<26} {'Lat.moy':>7}  {'LOCAL':>6}  {'GLOBAL':>6}  {'Faith':>6}  {'AnsRel':>6}  {'CtxRec':>6}")
    print("-" * W)
    for r in summary_rows:
        local_lat  = f"{r['lat_local']:.1f}s"  if r["lat_local"]  is not None else "  N/A"
        global_lat = f"{r['lat_global']:.1f}s" if r["lat_global"] is not None else "  N/A"
        print(
            f"{r['pipeline']:<12} {r['model']:<26} {r['lat_avg']:>6.1f}s"
            f"  {local_lat:>6}  {global_lat:>6}"
            f"  {_score(r['faithfulness'])}"
            f"  {_score(r['answer_relevancy'])}"
            f"  {_score(r['context_recall'])}"
        )
    print("=" * W)


# ---------------------------------------------------------------------------
# Matrice de décision automatique
# ---------------------------------------------------------------------------
def print_decision_matrix(summary_rows, thresholds):
    lat_fast = thresholds["latency_fast_s"]
    lat_ok   = thresholds["latency_ok_s"]
    q_good   = thresholds["quality_good"]
    q_ok     = thresholds["quality_ok"]

    def quality(r):
        f = r["faithfulness"]
        a = r["answer_relevancy"]
        if f != f or a != a:
            return 0.0
        return (f + a) / 2

    fast    = [r for r in summary_rows if r["lat_avg"] <= lat_fast]
    ok_lat  = [r for r in summary_rows if lat_fast < r["lat_avg"] <= lat_ok]
    slow    = [r for r in summary_rows if r["lat_avg"] > lat_ok]

    best_quality = max(summary_rows, key=quality, default=None)
    best_fast    = max(fast,         key=quality, default=None)
    best_balance = max(ok_lat,       key=quality, default=None)

    print("\nMATRICE DE DÉCISION")
    print(f"  Seuils : vitesse rapide < {lat_fast}s | acceptable < {lat_ok}s | qualité bonne > {q_good}")
    print("-" * 60)

    if best_fast:
        q = quality(best_fast)
        tag = "★" if q >= q_good else ("✓" if q >= q_ok else "~")
        print(f"  {tag} Vitesse max (< {lat_fast}s)         → {best_fast['pipeline']} + {best_fast['model']}  (qualité: {q:.2f})")
    else:
        print(f"  – Aucun run sous {lat_fast}s")

    if best_balance:
        q = quality(best_balance)
        tag = "★" if q >= q_good else ("✓" if q >= q_ok else "~")
        print(f"  {tag} Équilibre vitesse/qualité    → {best_balance['pipeline']} + {best_balance['model']}  (qualité: {q:.2f}, lat: {best_balance['lat_avg']:.1f}s)")

    if best_quality:
        q = quality(best_quality)
        tag = "★" if q >= q_good else ("✓" if q >= q_ok else "~")
        print(f"  {tag} Qualité max                  → {best_quality['pipeline']} + {best_quality['model']}  (qualité: {q:.2f}, lat: {best_quality['lat_avg']:.1f}s)")

    # Par type de question — minimum de latence parmi les runs de bonne qualité
    good_quality = [r for r in summary_rows if quality(r) >= q_ok] or summary_rows
    local_best  = min((r for r in good_quality if r["lat_local"]  is not None), key=lambda r: r["lat_local"],  default=None)
    global_best = min((r for r in good_quality if r["lat_global"] is not None), key=lambda r: r["lat_global"], default=None)
    if local_best:
        print(f"\n  Questions LOCAL uniquement     → {local_best['pipeline']} + {local_best['model']} ({local_best['lat_local']:.1f}s, qualité: {quality(local_best):.2f})")
    if global_best:
        print(f"  Questions GLOBAL uniquement    → {global_best['pipeline']} + {global_best['model']} ({global_best['lat_global']:.1f}s, qualité: {quality(global_best):.2f})")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_F))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    with open(GROUND_TRUTH_F, encoding="utf-8") as f:
        ground_truth = json.load(f)

    judge_model = config.get("judge_model", "gemma4:latest")
    thresholds  = config.get("thresholds", {"latency_fast_s": 3.0, "latency_ok_s": 8.0, "quality_good": 0.85, "quality_ok": 0.75})
    runs        = config["runs"]

    # Initialisation commune (partagée entre tous les runs)
    print("\nInitialisation commune (documents + vectorstore)...")
    docs           = load_documents()
    vectorstore    = build_vectorstore(docs)
    summary_chunks = build_semantic_chunks(docs)

    all_run_data  = []
    summary_rows  = []

    for run_cfg in runs:
        pipeline = run_cfg["pipeline"]
        model    = run_cfg["model"]

        run_results = run_one(pipeline, model, ground_truth, docs, vectorstore, summary_chunks)

        # Libère le modèle de la VRAM avant de charger le juge RAGAS (évite saturation GPU)
        _unload_model(model)

        print(f"\nÉvaluation RAGAS [{pipeline} / {model}]...")
        scores_df = ragas_eval(ground_truth, run_results, judge_model, EMBEDDING_MODEL)

        # Libère le juge avant le prochain run
        _unload_model(judge_model)

        # Agrégation des métriques perf par route
        lats_local  = [r["perf"]["latency_total"] for r in run_results if r["perf"].get("route") == "LOCAL"]
        lats_global = [r["perf"]["latency_total"] for r in run_results if r["perf"].get("route") == "GLOBAL"]
        all_lats    = [r["perf"]["latency_total"] for r in run_results]

        cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        row = {
            "pipeline":        pipeline,
            "model":           model,
            "lat_avg":         sum(all_lats) / len(all_lats) if all_lats else None,
            "lat_local":       sum(lats_local) / len(lats_local) if lats_local else None,
            "lat_global":      sum(lats_global) / len(lats_global) if lats_global else None,
            "faithfulness":        float(scores_df["faithfulness"].mean())        if "faithfulness"        in scores_df else float("nan"),
            "answer_relevancy":    float(scores_df["answer_relevancy"].mean())    if "answer_relevancy"    in scores_df else float("nan"),
            "context_precision":   float(scores_df["context_precision"].mean())   if "context_precision"   in scores_df else float("nan"),
            "context_recall":      float(scores_df["context_recall"].mean())      if "context_recall"      in scores_df else float("nan"),
        }
        summary_rows.append(row)

        # Détail par question
        questions_detail = []
        for i, (gt, res) in enumerate(zip(ground_truth, run_results)):
            score_row = scores_df.iloc[i]
            questions_detail.append({
                "id":               f"Q{i+1}",
                "question":         gt["question"],
                "ground_truth":     gt["ground_truth"],
                "answer":           res["answer"],
                "perf":             res["perf"],
                **{c: float(score_row.get(c, float("nan")) or 0) for c in cols},
            })

        all_run_data.append({
            "pipeline":   pipeline,
            "model":      model,
            "summary":    row,
            "questions":  questions_detail,
        })

    # Affichage
    print_comparison_table(summary_rows)
    print_decision_matrix(summary_rows, thresholds)

    # Sauvegarde
    RESULTS_DIR.mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = RESULTS_DIR / f"benchmark_{ts}.json"

    output = {
        "timestamp":   datetime.now().isoformat(),
        "judge_model": judge_model,
        "thresholds":  thresholds,
        "summary":     summary_rows,
        "runs":        all_run_data,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nRésultats sauvegardés → {out_path}\n")


if __name__ == "__main__":
    main()
