# rag-lab — design

## Contexte

Projet personnel d'apprentissage : construire un RAG (Retrieval-Augmented
Generation) avec LangChain et LangGraph, en partant d'un pipeline simple puis
en l'enrichissant progressivement vers un RAG agentique — sans réécriture
entre les deux étapes.

Vérification faite avant de démarrer : aucun projet RAG existant dans les
repos GitHub (`St4r4x`, `St4r4x-NV`, org `Missia-Neural-Vision`) ni en local
dans `~/Projects`. C'est un nouveau projet.

## Cas d'usage

Un assistant qui répond à des questions sur la documentation officielle de
LangChain et LangGraph elles-mêmes : corpus de démo, pas de donnée métier.
Choix volontairement "meta" — apprendre les frameworks en construisant un
outil qui répond sur les frameworks.

## Objectif d'apprentissage

Progression en une seule base de code, en deux étapes sur le même graphe :

1. **v1 — RAG simple** : graphe LangGraph à un chemin linéaire
   `retrieve → generate`.
2. **v2 — RAG agentique** : ajout de nœuds sur le même graphe
   (`grade_documents`, `rewrite_query`, boucle conditionnelle retrieve↔rewrite)
   sans rien jeter de v1.

Ce choix (plutôt que deux pipelines séparés, ou un prototypage notebook puis
réécriture) évite le code jetable : le graphe grandit, il n'est jamais
remplacé.

## Stack

- **LLM & embeddings** : configurables via variables d'environnement, en
  s'appuyant sur les fabriques natives de LangChain — `init_chat_model()` et
  `init_embeddings()`. Pas de couche d'abstraction custom : ce sont déjà des
  fonctions qui prennent une string de config
  (`"ollama:llama3.1"` vs `"openai:gpt-4o-mini"`) et retournent l'objet
  compatible LangChain adapté au provider.
- **Vector store** : Qdrant, lancé via Docker Compose (service unique). Une
  collection `langchain_docs`.
- **Orchestration** : LangGraph (`StateGraph`).
- **API** : FastAPI, endpoint `POST /query` + `GET /health`.
- **UI** : Gradio `ChatInterface`, qui appelle l'API en HTTP plutôt que
  d'invoquer le graphe en direct — même pattern service que le projet
  existant `InferAPI`.
- **Ollama** : supposé lancé nativement sur la machine (`ollama serve`), pas
  containerisé — ne pas ajouter un conteneur pour quelque chose qui tourne
  déjà simplement en local.

## Structure du repo

```
rag-lab/
├── ingestion/
│   └── ingest.py          # fetch docs LangChain/LangGraph → chunk → embed → upsert Qdrant
├── graph/
│   ├── state.py           # schema d'état (question, documents, generation, retries)
│   └── build.py           # StateGraph : v1 = retrieve→generate, v2 = +grade/rewrite/loop
├── config.py                # init_chat_model() / init_embeddings() via variables d'env
├── api/
│   └── main.py               # FastAPI, endpoint POST /query (+ /health)
├── ui/
│   └── app.py                 # Gradio ChatInterface, appelle l'API en HTTP
├── docker-compose.yml          # service Qdrant uniquement
├── .env.example
├── pyproject.toml
└── tests/
```

## Flux de données

### Ingestion (offline, à la demande : `python -m ingestion.ingest`)

1. Récupération des pages doc LangChain + LangGraph : shallow clone
   (`git clone --depth 1`) du repo unifié `langchain-ai/docs`, puis lecture
   des fichiers `.md`/`.mdx` sous `src/oss/langchain/` et `src/oss/langgraph/`
   — pas de scraper HTML. (Correction du 2026-08-11 : `langchain-ai/langchain`
   et `langchain-ai/langgraph` n'ont plus de dossier `docs/` avec du contenu ;
   la doc des deux frameworks a été consolidée dans ce repo unique.)
2. Split en chunks (`RecursiveCharacterTextSplitter`, ~500-800 tokens).
3. Embedding de chaque chunk via `init_embeddings()`.
4. Upsert dans Qdrant (collection `langchain_docs`), métadonnées = source
   (`langchain` / `langgraph`) + URL de la page d'origine.

### Requête (runtime)

```
UI (Gradio) → POST /query {question} → FastAPI
  → graph.invoke({question})
      → retrieve (top-k Qdrant)
      → [v2] grade_documents → pertinent ? generate : rewrite_query → retour retrieve (max 2 boucles)
      → generate (LLM + contexte)
  → FastAPI renvoie {answer, sources}
→ UI affiche la réponse + les sources citées
```

Les `sources` (URLs des pages doc utilisées pour la réponse) remontent
jusqu'à l'UI, pour vérifier que le RAG cite ses documents plutôt que
d'inventer.

## Gestion d'erreurs

- **Retrieval vide ou peu pertinent** : boucle `rewrite_query` bornée à 2
  tentatives maximum, puis réponse explicite ("je n'ai pas assez
  d'information dans la doc indexée") plutôt qu'une hallucination.
- **Provider LLM/embeddings en erreur** (clé API manquante, Ollama non
  lancé, timeout) : erreur propagée clairement (FastAPI 502), affichée dans
  l'UI. Pas de retry silencieux qui masquerait le vrai problème.
- **Qdrant inaccessible** : vérifié par `/health` au démarrage de l'API,
  échec rapide et explicite plutôt qu'une erreur obscure au premier query.

## Tests

Le seul point avec une logique de branchement non triviale est le routing du
graphe (`grade_documents` → pertinent ou pas → `generate` vs
`rewrite_query`, et la limite de boucles). Un test unitaire dédié couvre ce
routing avec un LLM mocké : documents jugés non pertinents → passage par
`rewrite_query` → boucle au plus 2 fois → sortie en `generate` (fallback).

Pas de suite de tests exhaustive au-delà de ce point : l'ingestion et les
appels réseau réels (Ollama/API/Qdrant) ne sont pas couverts par des tests
unitaires — projet d'apprentissage, YAGNI sur la couverture de test.

## Hors scope (pour l'instant)

- Authentification sur l'API/UI.
- Déploiement (Jetson, cloud) — reste local pour cette itération.
- Ingestion incrémentale / mise à jour de la doc déjà indexée (on repart de
  zéro à chaque `ingest.py`).
