# Key Learnings — RAG Local sur Factures

## Où on en est

### Architecture actuelle
RAG Adaptatif + Chunking résumé + Stuffing  
Modèle : `google/gemma-4-12b-qat` via LM Studio (thread unique)  
Corpus : 10 factures texte

### Ce qu'on a construit
1. **Chunking résumé** : 1 chunk par facture contenant les métadonnées clés (client, montant, objet, mode de paiement, lignes de facturation, coordonnées, émetteur)
2. **Routeur LLM** : classifie chaque question en LOCAL (→ Chroma retrieval) ou GLOBAL (→ Stuffing)
3. **Stuffing** : pour les questions GLOBAL, tous les chunks résumé sont concaténés en un seul appel LLM

### Évolution des stratégies GLOBAL testées
| Stratégie | Temps GLOBAL | Raison d'abandon |
|---|---|---|
| Map-Reduce (textes bruts) | >200s | N appels LLM séquentiels, dégradation LM Studio |
| Map-Reduce (chunks résumé) | ~100s | Toujours trop lent |
| Stuffing (chunks résumé courts) | ~15s | Retenu |
| Stuffing (chunks résumé enrichis) | ~20s | Retenu — meilleure précision |

### Performances actuelles
| Route | Temps moyen |
|---|---|
| LOCAL | ~10s |
| GLOBAL | ~20s |

Le temps LOCAL se décompose ainsi : ~4-5s classification + ~0.01s retrieval + ~6s génération.

### Score de précision
| # | Question | Résultat |
|---|---|---|
| Q1 | Montant TTC Thomas Lefebvre | ✅ |
| Q2 | Services cabinet médical des Trois Fontaines | ❌ |
| Q3 | Facture la plus élevée | ✅ |
| Q4 | Clients ayant eu une formation informatique | ✅ |
| Q5 | Mode de paiement boulangerie Dupain | ❌ |

**Score : 3/5**

---

## Problèmes identifiés

### Q2 — Services non trouvés
- La ligne "Audit de sécurité (2 jours)" disparaît du chunk : le regex d'extraction des lignes de facturation attend un entier en quantité, mais la facture contient `2j`.
- Même avec 5/6 services présents, la réponse échoue — cause exacte non confirmée (route ? retriever ? LLM ?).

### Q5 — Mode de paiement non trouvé
- Le champ `mode_paiement` est extrait et présent dans le chunk.
- Cause probable : mauvais routage (GLOBAL au lieu de LOCAL) ou nom client mal normalisé ("boulangerie Dupain" vs nom exact dans le fichier).

### Classification coûteuse
- ~4-5s par question pour produire un seul token (LOCAL/GLOBAL).
- Représente 40-50% du temps total sur les questions LOCAL.

### LM Studio — thread unique
- Pas de parallélisme possible : les appels LLM sont strictement séquentiels.
- Toute stratégie multi-appels (Map-Reduce, Refine) est prohibitive sur ce setup.

---

## Next Steps

### 1. Changer de modèle pour un plus rapide
Le goulot principal est le TPS (tokens per second) du modèle. Un modèle plus léger ou plus optimisé (quantization plus agressive, architecture plus petite) réduirait directement les ~6s de génération et les ~4s de classification.

### 2. Benchmarks de modèles locaux en RAG
Avant de choisir un nouveau modèle, consulter les benchmarks de modèles locaux sur des tâches RAG (précision + vitesse). Quelques pistes : MTEB leaderboard, LM Studio model rankings, benchmarks communautaires sur des tâches d'extraction structurée.

### 3. Repenser l'approche pour les questions GLOBAL
Le stuffing fonctionne tant que le corpus reste petit. À mesure qu'il grandit, le contexte sature et les temps augmentent. Il faut peut-être penser différemment :
- **Répondre sans LLM pour les questions GLOBAL simples** : puisque les métadonnées sont déjà extraites (total_ttc, client, mode_paiement...), des questions comme "quelle est la facture la plus élevée" peuvent être résolues directement par tri/filtre sur les métadonnées — zéro appel LLM.
- **Index inversé sur les métadonnées** : construire un index structuré en mémoire pour les agrégations et comparaisons.
- **Classifier sans LLM** : remplacer le classifieur LLM (~4-5s) par une règle légère basée sur des mots-clés ou des embeddings locaux.
