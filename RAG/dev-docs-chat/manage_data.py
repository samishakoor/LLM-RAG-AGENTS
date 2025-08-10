import shutil
from vectorstore import (
    clear_vectorstore,
    delete_documents_by_source,
)
from shared_utils import UPLOADS_DIR, upload_file_dir, upload_url_dir
from handle_file_ingestion import (
    delete_file_record,
)
from handle_url_ingestion import (
    delete_url_record,
)
import os


def clean_record_files():
    """Delete the uploads directory and all record files."""
    # Delete the record files directory
    if os.path.exists(UPLOADS_DIR):
        shutil.rmtree(UPLOADS_DIR)

def delete_document(file_name):
    """Delete a specific file and its embeddings from the system."""
    if not file_name:
        return "❌ Please select a file to delete from the dropdown"

    if not os.path.exists(upload_file_dir):
        return "No files to delete."

    # Remove file documents from vector DB
    delete_documents_by_source(file_name)

    # Remove from record
    delete_file_record(file_name)

    print(f"\n\nDeleted File: {file_name}\nDeleted Embeddings for: {file_name}\n\n")
    return f"Deleted File: {file_name}\nDeleted Embeddings for: {file_name}"


def delete_url(url):
    """Delete a specific URL and its embeddings from the system."""
    if not url:
        return "❌ Please select a URL to delete from the dropdown"

    if not os.path.exists(upload_url_dir):
        return "No URLs to delete."

    # Remove url documents from vector DB
    delete_documents_by_source(url)

    # Remove from record
    delete_url_record(url)

    print(f"\n\nDeleted URL: {url}\nDeleted Embeddings for: {url}\n\n")
    return f"Deleted URL: {url}\nDeleted Embeddings for: {url}"


def clear_all_data():
    """Clear all data including vector store, uploaded files, and URL records."""
    try:
        # Clear all documents from vector store
        clear_vectorstore()
        print("\n\nDeleted all documents from vector store")

        # Clear uploaded files and URLs records
        clean_record_files()
        print("Deleted all uploaded files and URLs records")
    except Exception as e:
        print(f"Error clearing data: {str(e)}")
