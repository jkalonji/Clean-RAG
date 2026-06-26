# Suivi des tests RAG — Adaptatif + Chunking Sémantique + Map-Reduce

**Stack :** LM Studio / google/gemma-4-12b-qat · HuggingFace all-MiniLM-L6-v2 · Chroma · LLM Classifier · Chunking sémantique (résumé + sections)
**Fichier :** `rag_app_adaptive_semantic_chunking_map_reduce.py`

---

## Résultats

| # | Question | Route | Statut | Performance | Diagnostic |
|---|----------|-------|--------|-------------|------------|
| Q1 | Quel est le montant total TTC de la facture de Thomas Lefebvre ? | LOCAL | ✅ Correct | Rapide | Chunk résumé retourné — client + montant dans le même chunk |
| Q2 | Quels services ont été facturés au cabinet médical des Trois Fontaines ? | LOCAL | ✅ Correct | Rapide | Chunk de section "prestations" correctement récupéré |
| Q3 | Quelle est la facture la plus élevée et à quel client correspond-elle ? | GLOBAL | ✅ Correct | ⚠️ Lent | Map-Reduce : 10 appels LLM + 1 reduce — correct mais trop long |
| Q4 | Quels clients ont bénéficié d'une formation informatique ? | GLOBAL | ✅ Correct | ⚠️ Lent | Map-Reduce : 10 appels LLM + 1 reduce — correct mais trop long |
| Q5 | Quel est le mode de paiement utilisé par la boulangerie Dupain ? | LOCAL | ✅ Correct | Rapide | Chunk résumé retourné — mode de paiement extrait dans les métadonnées |

---

## Légende

| Symbole | Signification |
|---------|---------------|
| ✅ Correct | Réponse exacte et complète |
| ⚠️ Partiel | Réponse incomplète ou approximative |
| ❌ Faux | Réponse incorrecte |

---

## Analyse

### Ce qui fonctionne
- Le chunking sémantique (chunk résumé) résout définitivement les échecs LOCAL de l'architecture précédente.
- Le classifier route correctement toutes les questions.
- Toutes les questions passent pour la première fois — **5/5 correct**.

### Problème identifié : latence Map-Reduce
Les questions GLOBAL déclenchent 10 appels LLM séquentiels (Map) + 1 appel Reduce.
Avec un modèle local (Gemma 4 12b sur LM Studio), chaque appel prend plusieurs secondes.
Sur 10 documents c'est acceptable, mais cette approche ne passera pas à l'échelle.

### Pistes pour réduire la latence Map-Reduce

| Approche | Principe | Complexité |
|----------|----------|------------|
| **Appels Map en parallèle** | Lancer les 10 appels LLM simultanément via `asyncio` ou `ThreadPoolExecutor` | Faible |
| **Chunks résumés pour le GLOBAL** | Au lieu d'envoyer le document entier au Map, envoyer uniquement le chunk résumé (plus court → réponse plus rapide) | Faible |
| **Pré-filtrage vectoriel avant Map-Reduce** | Récupérer les k documents les plus pertinents via Chroma, puis faire le Map-Reduce uniquement sur ceux-là | Moyen |
| **Multi-Query** | Générer plusieurs reformulations de la question pour améliorer le recall du RAG standard — éviter le Map-Reduce pour certains cas GLOBAL | Moyen |
