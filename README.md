# Assistente de Saúde Informativo (Production-Grade RAG)

Sistema de RAG (Retrieval-Augmented Generation) desenvolvido para responder perguntas informativas com base exclusiva em documentos médicos (PDFs), estruturado com arquitetura híbrida, cache semântico, memória conversacional, guardrails de segurança, observabilidade completa e ciclo contínuo de avaliação (*Eval Harness* + *Data Flywheel*).

---

## Arquitetura do Sistema

* **Inferência LLM (Nuvem - Groq):** Utiliza `openai/gpt-oss-120b` via API da Groq para geração de respostas rápidas e reescrita contextual de perguntas de acompanhamento (*Standalone Question Rewriting*).
* **Embeddings & Vector Store (Local):** Utiliza `nomic-embed-text` rodando localmente via Ollama e persistência vetorial em ChromaDB (`./storage_chroma`).
* **Cache Semântico:** Coleção dedicada no ChromaDB com similaridade cosseno (`threshold = 0.80`) que responde perguntas recorrentes em milissegundos sem consumir tokens do LLM.
* **Guardrails & Segurança Médica:** Prompt restritivo combinado com pós-processamento que bloqueia diagnósticos, omite fontes quando há abstenção por falta de contexto e garante o aviso médico obrigatório em todas as respostas.
* **Observabilidade & Data Flywheel:**
  * Métricas em tempo real expostas em `/metrics` (latência p95, taxa de requisições HTTP, intervenções de guardrail e hits/misses do cache semântico) visualizadas no **Prometheus** e **Grafana**.
  * Auditoria assíncrona (`BackgroundTasks`) registrando todas as interações em `eval/audit_log.jsonl` e capturando abstenções, erros ou feedbacks negativos (`👎` no frontend) em `eval/production_failures.json`.

---

## Estrutura do Projeto

```text
assistente-saude-pdf/
├── Dockerfile                        # Imagem Docker da API FastAPI
├── docker-compose.yml                # Orquestração (Ollama + API + Prometheus + Grafana)
├── prometheus.yml                    # Configuração de scraping do Prometheus
├── api.py                            # Endpoints (/chat, /feedback), rewriting, guardrails e auditoria
├── main.py                           # Inicialização FastAPI com lifespan, CORS e instrumentação
├── models.py                         # Schemas Pydantic (ChatRequest com suporte a histórico)
├── rag.py                            # Pipeline RAG (ChatGroq + OllamaEmbeddings + ChromaDB)
├── semantic_cache.py                 # Cache semântico vetorial com métricas Prometheus
├── grafana/provisioning/             # Datasource e dashboard provisionados automaticamente
├── eval/                             # Eval Harness & Data Flywheel
│   ├── dataset.json                  # Golden dataset para testes de regressão
│   └── run_eval.py                   # Executor automatizado de avaliação (single-turn e multi-turn)
├── frontend/                         # Interface web estática (HTML/CSS/JS com feedback 👎)
├── pdfs/                             # Documentos médicos base (PDFs)
└── requirements.txt                  # Dependências Python
```

---

## Como Executar com Docker Compose

### 1. Pré-requisitos
* Docker e Docker Compose instalados.
* Uma chave de API gratuita da Groq (`GROQ_API_KEY`).

Crie um arquivo `.env` na raiz do projeto com sua chave:

```bash
GROQ_API_KEY=gsk_sua_chave_aqui
```

### 2. Subir toda a Stack
Na raiz do projeto, execute:

```bash
sudo docker compose up --build -d
```

O container `ollama-init` baixará automaticamente o modelo `nomic-embed-text` antes de iniciar a API.

* **API FastAPI (Swagger):** `http://localhost:8000/docs`
* **Métricas Prometheus:** `http://localhost:8000/metrics`
* **Prometheus UI:** `http://localhost:9090`
* **Grafana:** `http://localhost:3000` (Login: `admin` / `admin`)

### 3. Rodar o Frontend
Em um terminal na pasta `frontend/`:

```bash
python3 -m http.server 5500
```

Acesse `http://127.0.0.1:5500` no navegador.

---

## Eval Harness & Regressão Contínua

Para validar exatidão semântica, aderência ao aviso médico, memória conversacional e latência:

```bash
python3 eval/run_eval.py
```

Para promover um prompt capturado em `eval/production_failures.json` para o conjunto permanente de testes de regressão (`eval/dataset.json`):

```bash
python3 eval/run_eval.py --add-question "Sua pergunta aqui" --keywords palavra1 palavra2
```