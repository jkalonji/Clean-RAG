# Résultats — RAG Adaptatif + Map-Reduce Parallèle [Ollama]

**Date :** 2026-06-28  
**Modèles :** gemma4:latest (génération) + gemma3:1b (classification)  
**Architecture :** Chunking résumé + routing LOCAL/GLOBAL + Map-Reduce parallèle (asyncio, concurrence=4)  
**Retrieval GLOBAL :** hybride BM25 + cosine, K=20, fusion RRF  

---

## Résultats par question

| # | Route attendue | Route obtenue | Durée | Correct |
|---|---|---|------:|:-------:|
| Q1 — Montant TTC Thomas Lefebvre | LOCAL | LOCAL | 3.52s | ✓ |
| Q2 — Services cabinet médical Trois Fontaines | LOCAL | LOCAL | 4.74s | ✓ |
| Q3 — Facture la plus élevée | **GLOBAL** | LOCAL | 5.31s | ✗ |
| Q4 — Clients formation informatique | **GLOBAL** | LOCAL | 3.98s | ✗ |
| Q5 — Mode de paiement boulangerie Dupain | LOCAL | LOCAL | 3.78s | ✓ |

**Précision : 3/5** — erreurs sur Q3 et Q4 dues à une mauvaise classification.

---

## Analyse des erreurs

Q3 et Q4 sont des questions d'**agrégation/comparaison** qui nécessitent de parcourir tous les documents.  
Le classifier gemma3:1b les a routées vers LOCAL, ce qui a limité la recherche à 4 chunks Chroma — insuffisant pour une question comparative.

- **Q3** : *"Quelle est la facture la plus élevée ?"* → le modèle a renvoyé une facture au hasard parmi les 4 chunks récupérés (624 € au lieu de 27 240 €)
- **Q4** : *"Quels clients ont bénéficié d'une formation informatique ?"* → réponse partielle (1 client sur N possibles)

**Cause probable :** gemma3:1b est trop petit pour distinguer fiablement les questions d'agrégation des questions ciblées avec le prompt actuel.

---

## Points positifs

- **Vitesse LOCAL nettement améliorée** : 3.5–5.3s vs 5.7–14.2s (stuffing) — gain du classifier gemma3:1b
- Les questions LOCAL correctement classifiées sont toutes bien répondues
- La route GLOBAL + map-reduce parallèle n'a pas été exercée sur ce run

---

## Comparaison des architectures (modèle chaud)

| Architecture | Précision | Temps moyen LOCAL | Temps moyen GLOBAL |
|---|:---:|---:|---:|
| Stuffing (gemma4 seul) | 5/5 | ~6s | ~9s |
| Map-Reduce parallèle (gemma4 + gemma3:1b) | 3/5 | **~4s** | — (non testé) |

---

## Prochaine action

Améliorer la fiabilité du classifier gemma3:1b :
- Revoir le prompt de classification (exemples plus explicites pour les agrégations)
- Ou remplacer gemma3:1b par un modèle légèrement plus capable (ex: gemma3:4b)
