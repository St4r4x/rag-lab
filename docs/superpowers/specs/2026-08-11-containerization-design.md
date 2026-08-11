# rag-lab — containerization design

## Contexte

`rag-lab` existe déjà et fonctionne (voir `docs/superpowers/specs/2026-08-11-rag-lab-design.md`
et `docs/superpowers/plans/2026-08-11-rag-lab-implementation.md`). Seul Qdrant
tourne en conteneur ; l'API FastAPI, l'UI Gradio, le script d'ingestion et
Ollama tournent tous nativement sur la machine (venv local).

Objectif : mieux containeriser le projet pour qu'il soit présentable comme
une pièce de portfolio démontrant une maîtrise du RAG — `docker compose up`
doit suffire à faire tourner l'essentiel du système.

## Périmètre

- **Containerisés** : API (FastAPI), UI (Gradio), Qdrant (déjà fait),
  ingestion (comme job à la demande, pas un service permanent).
- **Reste natif** : Ollama. C'est un choix délibéré, pas une limitation —
  c'est le pattern courant même en production (les serveurs LLM tournent
  souvent à part des services applicatifs), et containeriser Ollama
  ajouterait une image lourde et une gestion GPU (passthrough
  nvidia-container-toolkit) sans bénéfice pour ce projet.
- **Hors périmètre** : basculer entre Ollama et un provider API reste géré
  par les variables d'environnement existantes (`LLM_MODEL`,
  `EMBEDDING_MODEL`), indépendamment de la containerisation — pas de
  profils docker-compose dédiés à ce switch.
- **Ajouts demandés au-delà de la containerisation stricte** : un diagramme
  d'architecture (Mermaid) dans le README, et une CI GitHub Actions qui
  lance la suite de tests à chaque push/PR.

## Architecture

Une image Docker unique (`Dockerfile` à la racine), construite depuis
`pyproject.toml` qui installe déjà toutes les dépendances du projet en un
seul package. Trois services du `docker-compose.yml` réutilisent cette
image avec des commandes différentes (`api`, `ui`, `ingestion`), plus le
service `qdrant` existant.

Une seule image plutôt qu'une par service : le projet est un seul package
Python, pas plusieurs bibliothèques indépendantes — trois `Dockerfile`
distincts n'apporteraient qu'une réduction marginale de taille d'image au
prix d'une maintenance triplée. Pas de multi-stage build non plus : sans
dépendance lourde (pas de torch/CUDA dans l'image), un `python:3.13-slim`
single-stage reste simple et suffisant.

```
rag-lab/
├── Dockerfile              # image unique, python:3.13-slim, pip install -e .
├── .dockerignore           # .venv, .git, __pycache__, *.egg-info, docs/
├── docker-compose.yml      # qdrant (existant) + api + ui + ingestion (profil "tools")
├── config.py               # + support OLLAMA_BASE_URL optionnel
├── .github/workflows/ci.yml
└── README.md               # quickstart Docker + diagramme Mermaid
```

## Composants

**`Dockerfile`** : base `python:3.13-slim`, copie `pyproject.toml` + code,
`pip install -e .` (sans `[dev]` — pytest n'est nécessaire qu'en CI, pas
dans l'image d'exécution). Pas de `CMD` fixe ; chaque service du compose
définit sa propre commande.

`ingestion/ingest.py` appelle `git clone` en `subprocess` — `git` n'est pas
présent dans `python:3.13-slim` par défaut. Le `Dockerfile` doit l'installer
(`apt-get install -y --no-install-recommends git`) avant `pip install`,
sinon le service `ingestion` échoue au premier `docker compose run`.

**`config.py`** : ajout d'un support pour `OLLAMA_BASE_URL` (variable
d'env optionnelle, vide par défaut). Quand elle est définie et que le
provider sélectionné (`LLM_MODEL`/`EMBEDDING_MODEL`) commence par
`"ollama:"`, sa valeur est passée en kwarg `base_url=...` à
`init_chat_model()`/`init_embeddings()`. Comportement natif (sans variable
définie) inchangé. Pas d'effet sur les providers API (OpenAI, etc.).

**`docker-compose.yml`** (étendu) :

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]
    healthcheck:
      # qdrant/qdrant has no curl/wget — bash's /dev/tcp is the only HTTP-client-free
      # way to probe it. Verified: the image has bash but no curl/wget/nc (checked
      # 2026-08-11). A bare TCP connect is a proxy for "server accepting connections",
      # good enough to gate api/ingestion startup.
      test: ["CMD-SHELL", "bash -c 'echo > /dev/tcp/localhost/6333'"]
      interval: 5s
      timeout: 3s
      retries: 10

  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    environment:
      QDRANT_URL: http://qdrant:6333
      OLLAMA_BASE_URL: http://host.docker.internal:11434
    extra_hosts: ["host.docker.internal:host-gateway"]
    depends_on:
      qdrant: {condition: service_healthy}

  ui:
    build: .
    command: python -m ui.app
    ports: ["7860:7860"]
    environment:
      RAG_API_URL: http://api:8000
    depends_on: [api]

  ingestion:
    build: .
    command: python -m ingestion.ingest
    profiles: ["tools"]
    environment:
      QDRANT_URL: http://qdrant:6333
      OLLAMA_BASE_URL: http://host.docker.internal:11434
    extra_hosts: ["host.docker.internal:host-gateway"]
    depends_on:
      qdrant: {condition: service_healthy}

volumes:
  qdrant_data:
```

**`.github/workflows/ci.yml`** : sur push et pull request, checkout,
`setup-python` (3.13), `pip install -e ".[dev]"`, `pytest tests/ -v`. Ne
build ni ne push aucune image Docker — seule la suite de tests existante
(mockée, sans dépendance réseau) tourne en CI.

**Diagramme (README, Mermaid)** : services conteneurisés (UI, API, Qdrant,
job d'ingestion) reliés à Ollama, mis en évidence visuellement comme seul
composant natif.

## Flux réseau

- `ui` → `api` : nom de service Docker (`http://api:8000`), remplace
  `localhost` utilisé en dev natif.
- `api` / `ingestion` → `qdrant` : nom de service (`http://qdrant:6333`).
- `api` / `ingestion` → Ollama : `http://host.docker.internal:11434`, via
  `extra_hosts: ["host.docker.internal:host-gateway"]` (mécanisme standard
  pour joindre l'hôte depuis un conteneur sous Docker Engine Linux).

`.env` reste utilisable pour surcharger ces valeurs (ex: changer
`LLM_MODEL`), chargé par `config.py` comme avant — la containerisation ne
change rien à ce mécanisme.

## Gestion d'erreurs

- Le healthcheck sur `qdrant` + `depends_on: condition: service_healthy`
  remplace le besoin d'un retry applicatif au démarrage : `api` et
  `ingestion` n'essaient de se connecter qu'une fois Qdrant réellement prêt.
- Ollama injoignable (Ollama non lancé sur l'hôte, ou mauvaise
  configuration de `extra_hosts`) : aucune nouvelle gestion d'erreur
  nécessaire, le chemin d'exception existant dans `api/main.py`
  (`try/except` → `HTTPException(502, ...)`) capture déjà toute erreur de
  connexion du provider LLM/embeddings, quelle qu'en soit la cause.

## Tests

Aucun nouveau test automatisé : la containerisation se vérifie
manuellement (`docker compose up`, puis requêtes réelles sur les
endpoints), dans le même esprit YAGNI que le reste du projet — ce n'est pas
un comportement applicatif à couvrir par des tests unitaires. La CI se
limite à faire tourner la suite déjà existante
(`tests/test_graph_routing.py`, 5 tests mockés, aucun appel réseau réel) à
chaque push/PR.

## Hors scope (pour l'instant)

- Containerisation d'Ollama (GPU passthrough) — décision explicite, pas un
  report.
- Profils docker-compose pour basculer Ollama ↔ provider API — déjà géré
  par les variables d'environnement, indépendamment de Docker.
- Build/push d'image Docker en CI, déploiement (Jetson, cloud, registry).
