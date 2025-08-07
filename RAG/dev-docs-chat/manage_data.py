import shutil
from shared_utils import vector_store, UPLOADS_DIR, upload_file_dir, upload_url_dir
from handle_file_ingestion import (
    delete_file_record,
)
from handle_url_ingestion import (
    delete_url_record,
)
import os


def clear_vector_store():
    docs = vector_store.get()
    if docs.get("ids", None):
        # Delete all documents by their IDs
        vector_store.delete(ids=docs["ids"])
        print(f"Deleted {len(docs['ids'])} documents from vectorstore")
    else:
        print("Vectorstore is already empty")



def clean_record_files():
    # Delete the record files directory
    if os.path.exists(UPLOADS_DIR):
        shutil.rmtree(UPLOADS_DIR)


def delete_embeddings_by_source(source_path):
    try:
        # Delete documents where source_path matches
        vector_store._collection.delete(where={"source_path": source_path})
        print(f"Deleted embeddings for source: {source_path}")
        return f"Deleted embeddings for: {source_path}"
    except Exception as e:
        print(f"Error deleting embeddings: {str(e)}")
        return f"Error deleting embeddings: {str(e)}"


def delete_document(file_name):
    if not file_name:
        return "❌ Please select a file to delete from the dropdown"

    if not os.path.exists(upload_file_dir):
        return "No files to delete."

    # Remove from vector DB
    delete_embeddings_by_source(file_name)

    # Remove from record
    delete_file_record(file_name)

    return f"Deleted File: {file_name}\nDeleted Embeddings for: {file_name}"


def delete_url(url):
    if not url:
        return "❌ Please select a URL to delete from the dropdown"

    if not os.path.exists(upload_url_dir):
        return "No URLs to delete."

    # Remove from vector DB
    delete_embeddings_by_source(url)

    # Remove from record
    delete_url_record(url)

    return f"Deleted URL: {url}\nDeleted Embeddings for: {url}"


def clear_all_data():
    try:
        # Clear all documents from vector store
        clear_vector_store()
        print("Deleted all documents from vector store")

        # Clear uploaded files and URLs records
        clean_record_files()
        print("Deleted all uploaded files and URLs records")
    except Exception as e:
        print(f"Error clearing data: {str(e)}")
