# Suivi des tests RAG — Architecture Adaptative (LLM Classifier)

**Stack :** LM Studio / google/gemma-4-12b-qat · HuggingFace all-MiniLM-L6-v2 · Chroma · LLM Classifier · chunk_size=500 / overlap=50
**Fichier :** `rag_app_adaptive.py`

---

## Résultats

| # | Question | Route détectée | Statut | Diagnostic |
|---|----------|---------------|--------|------------|
| Q1 | Quel est le montant total TTC de la facture de Thomas Lefebvre ? | LOCAL | ❌ Faux | Même échec que le RAG simple — le chunk retourné ne contient pas le bon montant ou le bon client |
| Q2 | Quels services ont été facturés au cabinet médical des Trois Fontaines ? | LOCAL | ❌ Faux | Même échec que le RAG simple — réponse partielle ou incorrecte |
| Q3 | Quelle est la facture la plus élevée et à quel client correspond-elle ? | GLOBAL | ✅ Correct | Map-Reduce parcourt tous les docs — agrégation réussie |
| Q4 | Quels clients ont bénéficié d'une formation informatique ? | GLOBAL | ✅ Correct | Map-Reduce parcourt tous les docs — liste complète retournée |
| Q5 | Quel est le mode de paiement utilisé par la boulangerie Dupain ? | LOCAL | ❌ Faux | Même échec que le RAG simple — chunk pertinent non récupéré |

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
Le classifier LLM route correctement Q3 et Q4 vers Map-Reduce.
Map-Reduce répond correctement à toutes les questions globales — résultat identique à `rag_app_map_reduce.py`.

### Problème confirmé : la route LOCAL est défaillante
Q1, Q2 et Q5 échouent pour la même raison que dans `rag_app.py` (RAG simple).
Le problème n'est pas le LLM ni le classifier — c'est la **qualité du retrieval** :
le chunk retourné par Chroma ne contient pas les informations clés de la facture cible.

**Cause probable — chunking naïf :**
Avec `chunk_size=500` et un découpage par caractères, une facture peut être splitée ainsi :
- Chunk A : en-tête + nom du client
- Chunk B : lignes de détail
- Chunk C : totaux + mode de paiement

La recherche par similarité retourne le chunk le plus proche sémantiquement de la question,
mais pas nécessairement le chunk qui contient la réponse (ex: "TOTAL TTC" dans chunk C,
alors que la requête matche sur le nom du client dans chunk A).

### Piste prioritaire : améliorer le chunking pour la route LOCAL
- **Chunking par document entier** : ne pas splitter les factures (elles sont courtes ~50 lignes)
- **Chunking sémantique** : découper par sections logiques (en-tête / lignes / totaux) plutôt que par nombre de caractères
- **Metadata enrichie** : indexer nom du client, numéro de facture, montant TTC comme métadonnées pour filtrer avant la recherche vectorielle
