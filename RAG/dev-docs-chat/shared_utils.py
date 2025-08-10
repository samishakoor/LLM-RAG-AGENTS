import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

PINECONE_INDEX_NAME = "dev-docs-chat"

UPLOADS_DIR = "uploads"
UPLOADED_URL_RECORD = "uploaded_urls.txt"
UPLOADED_FILE_RECORD = "uploaded_files.txt"

upload_file_dir = os.path.join(UPLOADS_DIR, UPLOADED_FILE_RECORD)
upload_url_dir = os.path.join(UPLOADS_DIR, UPLOADED_URL_RECORD)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SUPPORTED_FILE_TYPES = [
    ".pdf",
    ".md",
    ".txt",
    ".xlsx",
    ".xls",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".csv",
]

def chunk_documents(docs):
    """Chunk documents into smaller pieces"""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)
    return chunks
