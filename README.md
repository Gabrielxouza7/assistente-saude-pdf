# Assistente de Saúde Informativo

Este projeto é um chatbot de RAG (Retrieval-Augmented Generation) focado em responder perguntas sobre documentos médicos (PDFs).

O objetivo principal é construir um assistente 100% local, gratuito e privado, que funcione em hardware de consumidor (com pouco espaço em disco) e que possua fortes barreiras de segurança (guardrails) para impedir diagnósticos e alucinações.

O sistema utiliza o LangChain para orquestrar o fluxo, Ollama para executar LLMs leves localmente, e o FastAPI para servir uma API para um frontend de chat simples.

## 🚀 Funcionalidades Principais

* **100% Local e Gratuito:** Nenhuma chave de API é necessária. Não há custos.
* **100% Privado:** Os seus PDFs e as suas perguntas nunca saem da sua máquina.
* **Leve:** Utiliza o Ollama para rodar modelos leves (como `gemma:2b` e `nomic-embed-text`), eliminando a necessidade de `torch` e `sentence-transformers` (que consomem muito espaço em disco).
* **Guarda de Segurança (Guardrail) Robusta:** Implementa uma verificação em duas etapas (no *prompt* e na *API*) para evitar que o bot invente respostas ou fontes que não existem nos seus documentos.

## ⚙️ Arquitetura do Projeto

O fluxo de dados funciona da seguinte maneira:

1.  **Frontend (HTML/JS)** envia a `pergunta` para o backend.
2.  **FastAPI** recebe a requisição.
3.  O `api.py` monta a **`chat_chain`** (a "linha de montagem" do LangChain) com os componentes em `app_state`.
4.  A `chat_chain` executa o RAG:

    a.  Usa o `nomic-embed-text` para vetorizar a pergunta e buscar chunks (k: 3) relevantes no ChromaDB
    
    b.  Insere os chunks encontrados e a pergunta no CUSTOM_PROMPT_TEMPLATE.

    c.  Envia o prompt final para o modelo, que gera a resposta.
    
5.  O `api.py` recebe a resposta, aplica a verificação de segurança, para garantir que as fontes são reais, e a devolve ao Frontend.

## ⚠️ A Barreira de Segurança (Guardrail)

Modelos pequenos como o `gemma:2b` tendem a "alucinar" (inventar respostas) para serem prestáveis, mesmo que o contexto (os PDFs) esteja vazio. Eles também inventam fontes para fingir que seguiram as regras.

Implementámos uma solução em duas etapas para combater isto:

1. O `CUSTOM_PROMPT_TEMPLATE` instrui o LLM a responder *apenas* com o contexto e a dizer "Desculpe, não encontrei..." se o contexto estiver vazio.
2. O `api.py` não confia no LLM. Ele verifica manualmente a lista de `source_documents` que o *retriever* devolveu. Se o LLM deu uma resposta, mas a lista de fontes reais está vazia, a API descarta a resposta do LLM e substitui-a pela nossa mensagem de segurança. Isto garante que o utilizador nunca receba uma informação alucinada.

## 📦 Estrutura do Projeto

## 📦 Estrutura do Projeto

```bash
assistente-saude-pdf/
├── .venv/                  # O ambiente virtual
├── frontend/               # O nosso frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── pdfs_medicos/           # <-- COLOQUE OS SEUS PDFs AQUI
│   └── exemplo.pdf
├── storage_chroma/         # O banco vetorial (criado automaticamente)
├── api.py                  # Define o endpoint /chat
├── app_state.py            # Guarda a 'chain'
├── main.py                 # O servidor FastAPI (com lifespan)
├── models.py               # Define os modelos Pantic (ChatRequest)
├── rag.py                  # O "coração" - Constrói a chain do LangChain
└── requirements.txt        # As dependências Python
```

## 🚀 Como Executar

### Pré-requisitos (Fazer 1 vez)

**1. Instalar o Ollama (A Aplicação)**
```bash
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
```

**2. Baixar o modelo gemma:2b e o all-minilm (vetorizador leve).**

```bash
ollama pull gemma:2b
ollama pull nomic-embed-text
```
**3. Criar o Ambiente Python**
```bash
# Criar o .venv
python3 -m venv .venv
# Ativar
source .venv/bin/activate
# Instalar os pacotes
pip install --upgrade pip
pip install -r requirements.txt
```
### Executando o Projeto (2 Terminais)

Você precisa de dois terminais abertos na pasta assistente-saude-pdf/.

Terminal 1: O Backend (API)

Ative o ambiente virtual:
```bash 
source .venv/bin/activate
```
Inicie o servidor Uvicorn:
```bash
uvicorn main:app --reload
```
Navegue para a pasta do frontend:
```bash
cd frontend
```
Inicie um servidor HTTP simples:
```bash
python3 -m http.server 9000
```
Abra o seu navegador e vá para: http://localhost:9000