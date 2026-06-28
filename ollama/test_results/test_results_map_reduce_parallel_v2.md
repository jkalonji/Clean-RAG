# Résultats v2 — Map-Reduce Parallèle + Modèles spécialisés [Ollama]

**Date :** 2026-06-28  
**Modèles :**
- Classifier : gemma3:4b
- Map (extraction) : gemma3:4b
- Reduce + RAG local : gemma4:latest

**Architecture :** Chunking résumé + routing LOCAL/GLOBAL + Map-Reduce parallèle (asyncio, concurrence=4)  
**Retrieval GLOBAL :** hybride BM25 + cosine, K=20, fusion RRF  

---

## Résultats par question

| # | Route | Map | Reduce | Total | Correct |
|---|---|---:|---:|---:|:---:|
| Q1 — Montant TTC Thomas Lefebvre | LOCAL | — | — | 15.46s | ✓ |
| Q2 — Services cabinet médical Trois Fontaines | GLOBAL | 4.29s | 4.28s | 12.38s | ✓ |
| Q3 — Facture la plus élevée | GLOBAL | 2.40s | 7.30s | 10.10s | ✓ |
| Q4 — Clients formation informatique | GLOBAL | 2.00s | 5.77s | 8.15s | ✗ |
| Q5 — Mode de paiement boulangerie Dupain | LOCAL | — | — | 3.93s | ✓ |

**Précision : 4/5**

> Q1 lent (15.46s) : chargement du modèle en mémoire GPU au premier appel (warm-up).

---

## Parallélisation confirmée — détail Q4

```
doc[03] terminé à +0.71s   ┐
doc[04] terminé à +0.79s   ├─ batch 1 (4 docs simultanés)
doc[02] terminé à +0.84s   │
doc[01] terminé à +0.89s   ┘
doc[07] terminé à +1.50s   ┐
doc[05] terminé à +1.50s   ├─ batch 2
doc[08] terminé à +1.51s   │
doc[06] terminé à +1.54s   ┘
doc[09] terminé à +1.98s   ┐
doc[10] terminé à +1.99s   ┘ batch 3
Map total : 2.00s
```

10 docs traités en 3 batches de 4 — ~0.7s par doc en moyenne. Sans parallélisation : ~7s estimé.

---

## Analyse de l'erreur Q4

**Question :** *"Quels clients ont bénéficié d'une formation informatique ?"*

**Réponse obtenue (7 clients)** — dont 2 incorrects :
- ✓ M. Karim Benzara
- ✓ ÉCOLE PRIVÉE SAINT-EXUPÉRY
- ✓ CABINET MÉDICAL DES TROIS FONTAINES
- ✓ BOULANGERIE DUPAIN & FILS SARL
- ✓ Mme Isabelle Morand
- ✗ Mme Sarah Nguyen ← hallucination
- ✗ AGENCE CRÉATIVE PIXEL & CO ← hallucination

**Cause :** hallucination dans la phase **REDUCE** (gemma4). Le modèle a inventé des clients à partir du contexte général des factures, au lieu de se limiter strictement aux réponses MAP. Le prompt REDUCE doit être renforcé pour interdire toute inférence hors des réponses partielles.

---

## Comparaison des architectures (modèle chaud, requêtes GLOBAL)

| Architecture | Précision | Map 10 docs | Reduce | Total GLOBAL |
|---|:---:|---:|---:|---:|
| Stuffing — gemma4 seul | 5/5 | — | ~9s | ~9s |
| Map-Reduce — gemma4 seul | 3/5 | ~18s | ~5s | ~23s |
| **Map-Reduce — gemma3:4b/gemma4** | **4/5** | **~2.5s** | **~5s** | **~10s** |

---

## Prochaine action

Renforcer le prompt REDUCE pour éliminer les hallucinations :
> *"Base-toi UNIQUEMENT sur les réponses partielles fournies. N'ajoute aucune information qui ne figure pas explicitement dans ces réponses."*
