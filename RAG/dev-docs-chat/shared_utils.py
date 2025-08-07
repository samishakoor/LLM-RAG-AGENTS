from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "dev_docs"

UPLOADED_URL_RECORD = "uploaded_urls.txt"
UPLOADED_FILE_RECORD = "uploaded_files.txt"

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_metadata={"hnsw:space": "cosine"},
)

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits
