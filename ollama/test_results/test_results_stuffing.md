# Résultats — RAG Adaptatif + Chunking Résumé + Stuffing [Ollama]

**Date :** 2026-06-28  
**Modèle :** gemma4:latest (9.6 GB) via Ollama (mode serveur local)  
**Architecture :** Chunking sémantique résumé + routage LOCAL/GLOBAL + Stuffing  

---

## Résultats par question

| # | Route | Run 1 | Run 2 | Correct |
|---|-------|------:|------:|:-------:|
| Q1 — Montant TTC Thomas Lefebvre | LOCAL | 14.15s | 6.19s | ✓ |
| Q2 — Services cabinet médical Trois Fontaines | GLOBAL | 10.89s | 8.59s | ✓ |
| Q3 — Facture la plus élevée | GLOBAL | 10.04s | 10.06s | ✓ |
| Q4 — Clients formation informatique | GLOBAL | 9.41s | 9.33s | ✓ |
| Q5 — Mode de paiement boulangerie Dupain | LOCAL | 5.74s | 5.89s | ✓ |
| **TOTAL** | | **50.23s** | **40.06s** | **5/5** |

> Le Run 1 est plus lent sur Q1 (14.15s vs 6.19s) : modèle en cours de chargement en mémoire GPU (warm-up).

---

## Moyennes par type de route (Run 2, modèle chaud)

| Route | Questions | Temps moyen |
|-------|-----------|------------:|
| LOCAL | Q1, Q5 | 6.04s |
| GLOBAL | Q2, Q3, Q4 | 9.33s |

---

## Observations

- **5/5 réponses correctes** — qualité identique à la version LMStudio
- Le routage LOCAL/GLOBAL fonctionne correctement sur toutes les questions
- Les requêtes GLOBAL (stuffing) sont ~3s plus lentes que LOCAL, ce qui est attendu (contexte plus large)
- Stabilité bonne entre les deux runs (écart < 2s sauf Q1 warm-up)

---

## Comparaison LMStudio vs Ollama

| | LMStudio | Ollama |
|---|---|---|
| Modèle | gemma-4-12b-qat | gemma4:latest |
| Précision | 5/5 | 5/5 |
| Parallélisation | Non | Oui (`OLLAMA_NUM_PARALLEL`) |
| Accès réseau | IP fixe requise | localhost |
