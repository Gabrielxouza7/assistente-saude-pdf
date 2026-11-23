# Assistente de Saúde Informativo (Production Grade RAG)

Este projeto é um chatbot de RAG (Retrieval-Augmented Generation) focado em responder perguntas sobre documentos médicos (PDFs) com padrão de produção, observabilidade e robustez.

O sistema utiliza LangChain, Ollama para execução local de LLMs, ChromaDB, FastAPI com injeção de dependência nativa e instrumentação completa para Prometheus e Grafana, além de um **Eval Harness** automatizado.

## 🚀 Funcionalidades Principais

* **100% Local e Privado:** Roda em hardware de consumidor com Ollama (`gemma:2b` e `nomic-embed-text`) e ChromaDB.
* **Guarda de Segurança (Guardrail) Robusta:** Verificação em duas etapas para impedir diagnósticos, alucinações e garantir fontes reais.
* **Observabilidade Completa:** Métricas expostas via Prometheus (`/metrics`) e painéis prontos no Grafana, incluindo contadores de intervenções do Guardrail.
* **Eval Harness Automatizado:** Script de avaliação contínua (`eval/run_eval.py`) para medir latência, exatidão por palavras-chave e aderência aos avisos médicos.
* **Containerização com Docker Compose:** Orquestração completa da API, Prometheus e Grafana.

---

## 📦 Estrutura do Projeto

```bash
assistente-saude-pdf/
├── Dockerfile              # Imagem Docker da API FastAPI
├── docker-compose.yml      # Orquestração (API + Prometheus + Grafana)
├── prometheus.yml          # Configuração de scraping do Prometheus
├── api.py                  # Endpoints da API com métricas Prometheus
├── main.py                 # Inicialização FastAPI com lifespan e state
├── models.py               # Modelos Pydantic
├── rag.py                  # Lógica do RAG (ChromaDB + Ollama)
├── eval/                   # Eval Harness
│   ├── dataset.json        # Casos de teste dourados (golden dataset)
│   └── run_eval.py         # Executor de testes de avaliação
├── frontend/               # Interface web estática
├── pdfs/                   # Documentos PDF base
└── requirements.txt        # Dependências Python
```

---

## ⚙️ Como Executar com Docker Compose (Recomendado)

A maneira mais profissional de rodar o projeto é utilizando Docker Compose para gerenciar a API e a observabilidade.

### 1. Pré-requisitos
Certifique-se de ter o Docker e o Ollama instalados localmente na sua máquina (rodando na porta 11434).

Baixe os modelos necessários no Ollama:
```bash
ollama pull gemma:2b
ollama pull nomic-embed-text
```

### 2. Subir os Serviços (API + Prometheus + Grafana)
Execute o Docker Compose na raiz do projeto:
```bash
docker compose up --build -d
```

* **API FastAPI:** `http://localhost:8000` (Documentação em `http://localhost:8000/docs`)
* **Métricas Prometheus:** `http://localhost:8000/metrics`
* **Prometheus UI:** `http://localhost:9090`
* **Grafana:** `http://localhost:3000` (Usuário/Senha: `admin` / `admin`)

---

## 🧪 Executando o Eval Harness (Avaliação)

Para testar a qualidade das respostas, latência e funcionamento dos guardrails do RAG:

1. Certifique-se de que a API está rodando (via Docker ou `uvicorn`).
2. Execute o script de avaliação:
```bash
python3 eval/run_eval.py
```
O script rodará os casos de teste definidos em `eval/dataset.json` e exibirá um relatório detalhado de sucesso, latência e conformidade.
