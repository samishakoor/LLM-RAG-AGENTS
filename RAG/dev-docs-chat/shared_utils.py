from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

COLLECTION_NAME = "dev_docs"
CHROMA_DB_DIR = "./chroma_db"

UPLOADED_URL_RECORD = "uploaded_urls.txt"
UPLOADED_FILE_RECORD = "uploaded_files.txt"
UPLOADS_DIR = "uploads"

upload_file_dir = os.path.join(UPLOADS_DIR, UPLOADED_FILE_RECORD)
upload_url_dir = os.path.join(UPLOADS_DIR, UPLOADED_URL_RECORD)

# Initialize persistent vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_DIR,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_metadata={"hnsw:space": "cosine"},
)


def chunk_and_embed_docs(docs, source_type, source_path):
    """Chunk and embed documents to the vector store"""
    # Add metadata to documents
    for doc in docs:
        doc.metadata["source_type"] = source_type
        doc.metadata["source_path"] = source_path

    # Chunk documents
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Add documents to vector store
    print("Adding docs to vector store...")
    vector_store.add_documents(chunks)
