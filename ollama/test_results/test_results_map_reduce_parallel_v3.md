# Résultats v3 — Map-Reduce Parallèle + gemma3:4b full stack [Ollama]

**Date :** 2026-06-28  
**Modèles :**
- Classifier : gemma3:4b
- Map (extraction) : gemma3:4b
- Reduce + RAG local : gemma3:4b

**Architecture :** Chunking résumé + routing LOCAL/GLOBAL + Map-Reduce parallèle (asyncio, concurrence=4)  
**Retrieval GLOBAL :** hybride BM25 + cosine, K=20, fusion RRF  

---

## Résultats par question

| # | Route | Map | TTFT | Génération | Total | Correct |
|---|---|---:|---:|---:|---:|:---:|
| Q1 — Montant TTC Thomas Lefebvre | LOCAL | — | — | — | 1.21s | ✓ |
| Q2 — Services cabinet médical Trois Fontaines | GLOBAL | 1.95s | 0.38s | 0.14s | 2.87s | ✓ |
| Q3 — Facture la plus élevée | GLOBAL | 2.59s | 0.38s | 0.33s | 3.68s | ✓ |
| Q4 — Clients formation informatique | GLOBAL | 2.35s | 0.36s | 0.44s | 3.54s | ✗ |
| Q5 — Mode de paiement boulangerie Dupain | LOCAL | — | — | — | 1.01s | ✓ |

**Précision : 4/5**

---

## Analyse des temps — phase Reduce

Le TTFT (~0.38s) et la génération (0.14–0.44s) sont négligeables. Le reduce n'est plus un goulot d'étranglement.

```
Reduce TTFT       : ~0.38s  (constant, indépendant du contexte)
Reduce génération : 0.14–0.44s selon la longueur de la réponse
Reduce total      : 0.52–0.80s
```

Le temps dominant est désormais entièrement dans la phase **Map (~2s)**.

---

## Parallélisation — détail Q2

```
doc[04] terminé à +0.73s  ┐
doc[02] terminé à +0.73s  ├─ batch 1
doc[03] terminé à +0.74s  │
doc[01] terminé à +0.80s  ┘
doc[07] terminé à +1.40s  ┐
doc[06] terminé à +1.45s  ├─ batch 2
doc[05] terminé à +1.45s  │
doc[08] terminé à +1.46s  ┘
doc[09] terminé à +1.93s  ┐
doc[10] terminé à +1.95s  ┘ batch 3 (2 docs)
Map total : 1.95s
```

---

## Analyse de l'erreur Q4

**Question :** *"Quels clients ont bénéficié d'une formation informatique ?"*  
**Réponse :** 6 clients listés dont 2 incorrects (Sarah Nguyen, AGENCE CRÉATIVE PIXEL & CO).

La cause a changé par rapport à v2 : ce n'est plus le **Reduce** qui hallucine, c'est le **Map**.  
gemma3:4b interprète trop largement le champ `Objet` de certaines factures (ex: "RGPD", "infrastructure réseau") comme relevant d'une "formation informatique". Le Reduce n'invente rien — il synthétise fidèlement des réponses MAP incorrectes.

**Prochaine action :** renforcer le MAP_PROMPT pour exiger une citation exacte du texte source.

---

## Comparaison globale des architectures

| Architecture | Modèles | Précision | LOCAL | GLOBAL |
|---|---|:---:|---:|---:|
| Stuffing | gemma4 seul | 5/5 | ~6s | ~9s |
| Map-Reduce v1 | gemma4 seul | 3/5 | ~6s | ~23s |
| Map-Reduce v2 | gemma3:4b/gemma4 | 4/5 | ~4s | ~10s |
| **Map-Reduce v3** | **gemma3:4b full** | **4/5** | **~1.1s** | **~3.4s** |
