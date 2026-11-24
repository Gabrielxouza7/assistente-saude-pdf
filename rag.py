import os
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

CUSTOM_PROMPT_TEMPLATE = """
Você é um assistente de saúde informativo. Sua função é responder a perguntas 
sobre saúde com base *exclusivamente* nos documentos fornecidos.

Instruções Críticas:
1. NÃO forneça diagnósticos, prognósticos ou conselhos de tratamento.
2. Se a pergunta pedir um diagnóstico (ex: "Eu tenho X?"), RECUSE-SE
   educadamente e lembre o usuário de consultar um médico.
3. Se o contexto não contiver a resposta, diga que você não sabe.
4. Baseie TODAS as respostas nos fatos do contexto.
5. SEMPRE termine sua resposta com a frase: 
   "Esta informação não substitui uma consulta médica profissional. 
   Consulte seu médico."
6. Dê respostas bem elaboradas, evitando gerar respostas muito curtas
   mas não fornecendo resposta muito grandes

Contexto Fornecido:
{context}

Pergunta: {question}

Resposta Informativa:
"""

def load_components():
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 1. Configurar LLM (Llama 3.3 70B via Groq)
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1
    )

    # 2. Configurar Modelo de Embedding local (Ollama)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)

    # 3. Lógica "Carregar OU Criar" o Índice (com ChromaDB)
    if os.path.exists("./storage_chroma") and os.listdir("./storage_chroma"):
        vector_store = Chroma(
            persist_directory="./storage_chroma",
            embedding_function=embeddings
        )
    else:
        if not os.path.exists("./pdfs") or not os.listdir("./pdfs"):
            print("Aviso: Pasta ./pdfs vazia ou inexistente.")
            vector_store = None
        else:
            loader_pdf = PyPDFDirectoryLoader("./pdfs", glob="**/*.pdf")
            docs_pdf = loader_pdf.load()
            
            loader_PDF = PyPDFDirectoryLoader("./pdfs", glob="**/*.PDF")
            docs_PDF = loader_PDF.load()
            
            documents = docs_pdf + docs_PDF

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./storage_chroma"
            )

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    return llm, vector_store, prompt, embeddings