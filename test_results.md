# Suivi des tests RAG — TechSolutions Factures

**Stack :** LM Studio / google/gemma-4-12b-qat · HuggingFace all-MiniLM-L6-v2 · Chroma · chunk_size=500 / overlap=50

---

## Résultats

| # | Question | Statut | Diagnostic |
|---|----------|--------|------------|
| Q1 | Quel est le montant total TTC de la facture de Thomas Lefebvre ? | ❌ Faux | Mauvaise réponse — cause inconnue (chunking ? retrieval ?) |
| Q2 | Quels services ont été facturés au cabinet médical des Trois Fontaines ? | ⚠️ Partiel | Certains services manquants — la facture est probablement splitée sur plusieurs chunks dont seul une partie est récupérée |
| Q3 | Quelle est la facture la plus élevée et à quel client correspond-elle ? | ❌ Faux | Question globale : nécessite de comparer tous les documents. Le RAG ne récupère que k=3 chunks — les montants de toutes les factures ne sont jamais tous en contexte simultanément |
| Q4 | Quels clients ont bénéficié d'une formation informatique ? | ❌ Faux | Question globale : nécessite de parcourir tous les documents. Même problème que Q3 — le retrieval par similarité ne couvre pas l'ensemble du corpus |
| Q5 | Quel est le mode de paiement utilisé par la boulangerie Dupain ? | ❌ Faux | Mauvaise réponse — cause inconnue (chunking ? retrieval ?) |

---

## Légende

| Symbole | Signification |
|---------|---------------|
| ✅ Correct | Réponse exacte et complète |
| ⚠️ Partiel | Réponse incomplète ou approximative |
| ❌ Faux | Réponse incorrecte |

---

## Pistes d'amélioration identifiées

### Q3 & Q4 — Requêtes globales (agrégation / comparaison sur l'ensemble du corpus)
Le RAG standard par similarité est structurellement inadapté à ces questions :
il ne récupère que les `k` chunks les plus proches sémantiquement d'une requête,
sans garantie de couvrir tous les documents.

Pistes à évaluer :
- **Augmenter k** temporairement pour ces questions (risque : contexte LLM saturé)
- **Map-Reduce chain** : interroger chaque document séparément puis synthétiser
- **Self-query / metadata filtering** : ajouter les métadonnées (montant, client) à l'index pour permettre des requêtes structurées
- **Reranking** : récupérer plus de chunks puis reclasser

### Q1 & Q5 — Mauvaise récupération sur des questions précises
- Possiblement un problème de **chunking** : la ligne "TOTAL TTC" ou "Mode de paiement" se retrouve dans un chunk différent du nom du client
- À investiguer avec les **techniques de chunking** (session suivante)

### Q2 — Réponse partielle
- La facture 007 est longue → splitée en plusieurs chunks → seul une partie est retournée par le retriever (k=3)
- Augmenter k ou retravailler le chunking pour garder les blocs de lignes ensemble
