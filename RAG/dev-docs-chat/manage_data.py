from shared_utils import vector_store
from handle_file_ingestion import (
    get_uploaded_files,
    delete_file_record,
    UPLOADED_FILE_RECORD,
)
from handle_url_ingestion import (
    delete_url_record,
    UPLOADED_URL_RECORD,
)
import os


def delete_files_from_gradio(file_name=None):
    uploaded_files = get_uploaded_files()
    for name, path in uploaded_files:
        if os.path.exists(path):
            if file_name:
                if name == file_name:
                    os.remove(path)
                    break
            else:
                os.remove(path)

def clear_vector_store():
    docs = vector_store.get()
    if docs.get("ids", None):
        vector_store.delete(ids=docs["ids"])


def clean_record_files():
    # Delete the file that stores uploaded file paths
    if os.path.exists(UPLOADED_FILE_RECORD):
        os.remove(UPLOADED_FILE_RECORD)
    # Delete the file that stores uploaded URL paths
    if os.path.exists(UPLOADED_URL_RECORD):
        os.remove(UPLOADED_URL_RECORD)


def delete_document(file_name):
    if not file_name:
        return "❌ Please select a file to delete from the dropdown"

    # Remove from vector DB
    vector_store.delete(where={"source": file_name})

    # Remove temp file from gradio
    delete_files_from_gradio(file_name)

    # Remove from record
    delete_file_record(file_name)

    return f"Deleted File: {file_name}\nDeleted Embeddings for: {file_name}"


def delete_url(url):
    if not url:
        return "❌ Please select a URL to delete from the dropdown"

    # Remove from vector DB
    vector_store.delete(where={"source": url})

    # Remove from record
    delete_url_record(url)

    return f"Deleted URL: {url}\nDeleted Embeddings for: {url}"


def clear_all_data():
    try:
        # Clear all documents from vector store
        clear_vector_store()
        print("Deleted all documents from vector store")

        # Delete all uploaded files
        delete_files_from_gradio()
        print("Deleted all uploaded files")

        # Clear uploaded files and URLs records
        clean_record_files()
        print("Deleted all uploaded files and URLs records")
    except Exception as e:
        print(f"Error clearing data: {str(e)}")

