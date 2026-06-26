# Résultats de test — Map-Reduce sur chunks résumé
**Date :** 2026-06-27  
**Modèle :** google/gemma-4-12b-qat (LM Studio, thread unique)  
**Corpus :** 2 factures (Q1-Q2), 10 factures (Q3+)  
**Chunking :** résumés uniquement (~80-100 tokens/doc)

---

## Timings par question

### Q1 — LOCAL
> Quel est le montant total TTC de la facture de Thomas Lefebvre ?

| Étape | Durée |
|---|---|
| Classification LLM | 4.792s |
| Retrieval Chroma | 0.026s |
| Génération LLM | 6.071s |
| **TOTAL** | **10.889s** |

**Réponse :** Le montant total TTC est de 348.00 €. ✅

---

### Q2 — LOCAL
> Quels services ont été facturés au cabinet médical des Trois Fontaines ?

| Étape | Durée |
|---|---|
| Classification LLM | 5.267s |
| Retrieval Chroma | 0.011s |
| Génération LLM | 5.461s |
| **TOTAL** | **10.739s** |

**Réponse :** Cette information n'est pas présente dans le contexte fourni. ❌  
*(Le chunk résumé ne détaille pas les services — limite du chunking résumé seul)*

---

### Q3 — GLOBAL (Map-Reduce, 10 docs)
> Quelle est la facture la plus élevée et à quel client correspond-elle ?

| Étape | Durée |
|---|---|
| Classification LLM | 4.040s |
| map doc 1/10 | 9.141s |
| map doc 2/10 | 40.134s |
| map doc 3/10 | 111.213s |
| map doc 4/10 | 175.143s |
| map doc 5/10 | 22.397s |
| map doc 6/10 | 11.929s |
| map doc 7/10 | 9.108s |
| map doc 8-10 | non mesuré (test interrompu) |

> **Observation :** Les docs 2-4 présentent des pics anormaux (40s, 111s, 175s) — probablement dus à la file d'attente interne de LM Studio sous charge prolongée.

---

## Conclusions

| Route | Temps moyen | Goulot principal |
|---|---|---|
| LOCAL | ~10.8s | Classification LLM (~5s) + génération (~6s) |
| GLOBAL (Map-Reduce) | >200s estimé | N appels LLM séquentiels + dégradation LM Studio |

### Problèmes identifiés
1. **Map-Reduce incompatible avec LM Studio** : les N appels séquentiels saturent le thread unique → dégradation exponentielle des temps de réponse.
2. **Chunking résumé seul insuffisant pour LOCAL détaillé** : Q2 sans réponse car les services ne figurent pas dans le résumé.
3. **Classification coûteuse** : ~4-5s pour un seul token de sortie (LOCAL/GLOBAL).

### Prochaine étape
Tester la stratégie **Stuffing** pour les questions GLOBAL :  
→ 1 seul appel LLM avec tous les résumés concaténés (~1000 tokens)  
→ Objectif : passer de >200s à ~10-15s sur les questions GLOBAL.
