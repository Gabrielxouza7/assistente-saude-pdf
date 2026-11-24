import os

# Configure the Ollama host environment variable using fallback default URL
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Import LangChain modules for document loading, splitting, models, and vector storage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# Define the system prompt template enforcing strict context-only medical disclaimers
CUSTOM_PROMPT_TEMPLATE = """
Você é um assistente de saúde informativo. Sua função é responder a perguntas 
sobre saúde com base *exclusivamente* nos documentos fornecidos.

Instruções Críticas:
1. NÃO forneça diagnósticos, prognósticos ou conselhos de tratamento.
2. Se a pergunta pedir um diagnóstico (ex: "Eu tenho X?"), RECUSE-SE
   educadamente e lembre o usuário de consultar um médico.
3. Se o contexto não contiver a resposta, inicie sua resposta EXATAMENTE com:
   "Desculpe, não encontrei informações sobre isso nos documentos fornecidos."
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

# Initialize and return the LLM, vector store, prompt template, and embeddings model
def load_components():
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Initialize the Groq LLM client with low temperature for deterministic answers
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1
    )

    # Initialize the local Ollama embeddings model
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)

    # Load existing Chroma vector store from disk if the storage directory is populated
    if os.path.exists("./storage_chroma") and os.listdir("./storage_chroma"):
        vector_store = Chroma(
            persist_directory="./storage_chroma",
            embedding_function=embeddings
        )

    # Build a new vector store from PDFs when no existing Chroma storage is found
    else:

        # Set vector store to None if the PDF source directory is missing or empty
        if not os.path.exists("./pdfs") or not os.listdir("./pdfs"):
            print("Aviso: Pasta ./pdfs vazia ou inexistente.")
            vector_store = None

        # Load lowercase and uppercase PDF files and combine them into a single list
        else:
            loader_pdf = PyPDFDirectoryLoader("./pdfs", glob="**/*.pdf")
            docs_pdf = loader_pdf.load()
            loader_PDF = PyPDFDirectoryLoader("./pdfs", glob="**/*.PDF")
            docs_PDF = loader_PDF.load()
            documents = docs_pdf + docs_PDF

            # Split loaded documents into overlapping chunks and persist them in Chroma
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./storage_chroma"
            )

    # Build the LangChain prompt template and return all initialized components
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return llm, vector_store, prompt, embeddings