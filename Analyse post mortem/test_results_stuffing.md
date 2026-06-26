# Résultats de test — Stuffing sur chunks résumé
**Date :** 2026-06-27  
**Modèle :** google/gemma-4-12b-qat (LM Studio)  
**Corpus :** 10 factures  
**Chunking :** résumés uniquement (~80-100 tokens/doc)  
**Stratégie GLOBAL :** Stuffing (tous les résumés en un seul appel LLM)

---

## Timings

| Route | Temps moyen |
|---|---|
| LOCAL (classification + retrieval + génération) | ~10s |
| GLOBAL (classification + stuffing) | ~15s |

**Amélioration vs Map-Reduce :** GLOBAL passe de >200s à ~15s (-92%)

---

## Résultats par question

### Q1 — LOCAL ✅
> Quel est le montant total TTC de la facture de Thomas Lefebvre ?

**Résultat :** Correct

---

### Q2 — LOCAL ❌
> Quels services ont été facturés au cabinet médical des Trois Fontaines ?

**Résultat :** Incorrect  
**Cause :** Le chunk résumé ne contient que l'objet général de la facture (`meta["objet"]`), pas le détail ligne par ligne des services facturés. Cette information est perdue lors du chunking résumé seul.

---

### Q3 — GLOBAL ✅
> Quelle est la facture la plus élevée et à quel client correspond-elle ?

**Résultat :** Correct  
**Note :** Le `total_ttc` est extrait et présent dans chaque résumé — la comparaison est triviale pour le LLM.

---

### Q4 — GLOBAL ⚠️ Partiel
> Quels clients ont bénéficié d'une formation informatique ?

**Résultat :** 1 client retrouvé sur 3 attendus  
**Cause probable :** Le champ `objet` dans les résumés est trop générique (ex: "Formation informatique" vs "Formation bureautique avancée"). Le LLM ne fait pas le lien entre les variantes de libellé et le concept de "formation informatique".

---

### Q5 — LOCAL/GLOBAL ❌
> Quel est le mode de paiement utilisé par la boulangerie Dupain ?

**Résultat :** Non trouvé  
**Cause probable :** Erreur de routage (classifié GLOBAL au lieu de LOCAL), ou le nom "boulangerie Dupain" ne correspond pas exactement au champ `client` extrait — problème de normalisation du nom client dans les métadonnées.

---

## Bilan

| # | Question | Résultat | Route |
|---|---|---|---|
| Q1 | Montant TTC Thomas Lefebvre | ✅ Correct | LOCAL |
| Q2 | Services cabinet médical | ❌ Incorrect | LOCAL |
| Q3 | Facture la plus élevée | ✅ Correct | GLOBAL |
| Q4 | Clients formation informatique | ⚠️ Partiel (1/3) | GLOBAL |
| Q5 | Mode de paiement boulangerie Dupain | ❌ Non trouvé | LOCAL/GLOBAL |

**Score global : 2/5** (+ 1 partiel)

---

## Analyse des limites

| Limite | Impact | Piste d'amélioration |
|---|---|---|
| Résumé sans détail des lignes | Q2 sans réponse | Enrichir le chunk résumé avec les intitulés de services |
| Libellés d'objet trop variés | Q4 partiel | Normaliser l'objet ou ajouter un champ `tags` |
| Extraction du nom client fragile | Q5 peut-être | Améliorer le regex client ou stocker le nom de fichier comme fallback |
| Classification coûteuse (~4-5s) | 40-50% du temps total LOCAL | Classifier localement sans LLM (règles, embeddings) |
