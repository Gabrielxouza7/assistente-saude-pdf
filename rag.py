import os
import re
import time
import html
import requests
import xml.etree.ElementTree as ET
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure the Ollama host environment variable using fallback default URL
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Import LangChain modules for documents, splitting, models, and vector storage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# Define the system prompt template enforcing strict context-only medical disclaimers
CUSTOM_PROMPT_TEMPLATE = """
Você é um assistente de saúde informativo. Sua função é responder a perguntas 
sobre saúde com base *exclusivamente* nos guias médicos oficiais fornecidos no contexto.

Instruções Críticas:
1. NÃO forneça diagnósticos individuais, prognósticos ou prescrições de tratamento.
2. Se a pergunta pedir um diagnóstico pessoal (ex: "Eu tenho X?"), RECUSE-SE
   educadamente e lembre o usuário de consultar um médico.
3. Se o contexto não contiver NENHUMA informação relacionada ao tema da pergunta, inicie sua resposta EXATAMENTE com:
   "Desculpe, não encontrei informações sobre isso nos documentos fornecidos."
   Caso o contexto traga informações sobre a doença perguntada, sintetize os fatos disponíveis de forma clara e completa.
4. Baseie TODAS as respostas exclusivamente nos fatos do contexto fornecido, traduzindo-os para uma linguagem clara, empática e acessível ao público geral (responda sempre em português).
5. SEMPRE termine sua resposta com a frase: 
   "Esta informação não substitui uma consulta médica profissional. 
   Consulte seu médico."
6. Dê respostas bem elaboradas, evitando gerar respostas muito curtas
   mas não fornecendo respostas muito grandes.

Contexto Fornecido:
{context}

Pergunta: {question}

Resposta Informativa:
"""

# Define default list of 100 specific medical conditions in Spanish for MedlinePlus queries
DEFAULT_MEDICAL_TOPICS = (
    # Oncologia específica (10)
    "cáncer de mama,cáncer de próstata,cáncer de pulmón,cáncer colorrectal,"
    "cáncer de cuello uterino,cáncer de estómago,cáncer de hígado,"
    "melanoma,leucemia,linfoma,"
    # Cardiovasculares (10)
    "presión arterial alta,ataque al corazón,insuficiencia cardíaca,"
    "accidente cerebrovascular,fibrilación auricular,enfermedad de las arterias coronarias,"
    "trombosis venosa profunda,embolia pulmonar,aterosclerosis,miocarditis,"
    # Endócrinas e Metabólicas (10)
    "diabetes tipo 1,diabetes tipo 2,obesidad,colesterol alto,"
    "hipotiroidismo,hipertiroidismo,síndrome metabólico,gota,"
    "hígado graso,síndrome de ovario poliquístico,"
    # Respiratórias (9)
    "asma,enfermedad pulmonar obstructiva crónica,neumonía,tuberculosis,"
    "rinitis alérgica,sinusitis,bronquitis crónica,"
    "apnea del sueño,fibrosis pulmonar,"
    # Infecciosas e Parasitárias (21)
    "covid-19,gripe,dengue,zika,chikungunya,fiebre amarilla,malaria,"
    "enfermedad de chagas,leishmaniasis,leptospirosis,vih sida,hepatitis b,hepatitis c,"
    "sífilis,virus del papiloma humano,herpes simple,meningitis,"
    "sepsis,lepra,sarampión,varicela,"
    # Neurológicas e Psiquiátricas (15)
    "enfermedad de alzheimer,enfermedad de parkinson,epilepsia,migraña,"
    "esclerosis múltiple,esclerosis lateral amiotrófica,neuropatía diabética,"
    "depresión,trastorno de ansiedad,trastorno bipolar,"
    "esquizofrenia,trastorno del espectro autista,"
    "trastorno de déficit de atención e hiperactividad,"
    "trastorno obsesivo-compulsivo,trastorno de estrés postraumático,"
    # Gastrointestinais e Hepáticas (10)
    "reflujo gastroesofágico,gastritis,úlcera péptica,"
    "enfermedad de crohn,colitis ulcerativa,síndrome del intestino irritable,"
    "cirrosis,pancreatitis,cálculos biliares,enfermedad celíaca,"
    # Renais, Urológicas e Ginecológicas (5)
    "enfermedad renal crónica,cálculos renales,infecciones de las vías urinarias,"
    "hiperplasia prostática benigna,endometriosis,"
    # Reumatológicas, Dermatológicas, Hematológicas e Oftalmológicas (10)
    "artritis reumatoide,osteoartritis,osteoporosis,lupus,"
    "fibromialgia,psoriasis,eczema,anemia,glaucoma,cataratas"
)

# Strip HTML tags and normalize whitespace from MedlinePlus XML summary blocks
def _clean_html_summary(raw_html: str) -> str:
    text = html.unescape(raw_html or "")
    text = re.sub(r"<li[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|h\d|div|ul|ol)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

# Query MedlinePlus NIH Web Service for a single medical condition with retry handling
def _fetch_single_topic(topic: str, count_per_topic: int, max_retries: int = 2) -> Tuple[str, List[dict]]:
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {
        "db": "healthTopicsSpanish",
        "term": topic,
        "retmax": count_per_topic,
        "rettype": "all",
        "tool": "AssistenteSaudeRAG"
    }

    # Attempt to fetch XML results with retry backoff on timeouts or transient errors
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            parsed_topics = []

            # Iterate through each returned document node in the MedlinePlus XML response
            for doc_elem in root.findall(".//document"):
                topic_url = doc_elem.attrib.get("url", "").strip()
                title = ""
                full_summary = ""
                also_called = []

                # Extract title, synonyms, and full patient-oriented summary fields
                for content_elem in doc_elem.findall("content"):
                    name = content_elem.attrib.get("name", "")
                    text_val = "".join(content_elem.itertext()).strip()
                    if name == "title" and text_val:
                        title = _clean_html_summary(text_val)
                    elif name == "FullSummary" and text_val:
                        full_summary = _clean_html_summary(text_val)
                    elif name == "also-called" and text_val:
                        also_called.append(_clean_html_summary(text_val))

                if topic_url and title and len(full_summary) > 100:
                    parsed_topics.append({
                        "url": topic_url,
                        "title": title,
                        "also_called": ", ".join(dict.fromkeys(also_called)),
                        "summary": full_summary
                    })

            return topic, parsed_topics

        # Wait briefly before retrying on transient failures, or log final warning
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"Aviso: Falha definitiva ao consultar MedlinePlus para '{topic}': {e}", flush=True)

    return topic, []

# Fetch patient-oriented health guides across all configured medical conditions
def fetch_medlineplus_topics(topics: List[str], count_per_topic: int = 3, max_retries: int = 2) -> List[Document]:
    documents: List[Document] = []
    seen_urls = set()
    clean_topics = [t.strip() for t in topics if t.strip()]

    print(f"Iniciando coleta paralela para {len(clean_topics)} condições na MedlinePlus (NIH)...", flush=True)

    # Execute concurrent HTTP queries using a bounded thread pool to respect NLM rate limits
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_fetch_single_topic, topic, count_per_topic, max_retries): topic
            for topic in clean_topics
        }

        # Process completed topic responses and deduplicate retrieved health guides
        for idx, future in enumerate(as_completed(futures), start=1):
            topic, results = future.result()
            added_for_topic = 0

            # Build LangChain Document objects with official MedlinePlus URLs and titles
            for item in results:
                topic_url = item["url"]

                # Skip duplicate health topics already collected from another query
                if topic_url in seen_urls:
                    continue

                seen_urls.add(topic_url)
                synonyms_line = f"También conocido como: {item['also_called']}\n" if item["also_called"] else ""

                page_content = (
                    f"Condición / Tema: {item['title']} ({topic})\n"
                    f"{synonyms_line}"
                    f"Guía Médica para Pacientes (MedlinePlus / NIH):\n{item['summary']}"
                )
                documents.append(Document(
                    page_content=page_content,
                    metadata={
                        "source": topic_url,
                        "title": item["title"],
                        "journal": "MedlinePlus (NIH)",
                        "year": "2026"
                    }
                ))
                added_for_topic += 1

            print(f"[{idx}/{len(clean_topics)}] '{topic}': +{added_for_topic} guias únicos", flush=True)

    print(f"Total de guias médicos únicos carregados: {len(documents)}", flush=True)
    return documents

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
        print("Carregando base vetorial existente do ChromaDB...", flush=True)
        vector_store = Chroma(
            persist_directory="./storage_chroma",
            embedding_function=embeddings
        )

    # Build a new vector store from MedlinePlus health guides when no storage is found
    else:
        raw_topics = os.getenv("MEDICAL_TOPICS", DEFAULT_MEDICAL_TOPICS)
        topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
        count_per_topic = int(os.getenv("DOCS_PER_TOPIC", "3"))

        # Fetch patient-oriented medical guides across all configured conditions
        documents = fetch_medlineplus_topics(topics=topics, count_per_topic=count_per_topic)

        # Set vector store to None if no documents were retrieved from the API
        if not documents:
            print("Aviso: Nenhum guia médico retornado pela API.", flush=True)
            vector_store = None

        # Split fetched guides into overlapping chunks and persist them in Chroma
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
            print(f"Indexando {len(chunks)} chunks no ChromaDB...", flush=True)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./storage_chroma"
            )
            print("Indexação concluída com sucesso!", flush=True)

    # Build the LangChain prompt template and return all initialized components
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return llm, vector_store, prompt, embeddings