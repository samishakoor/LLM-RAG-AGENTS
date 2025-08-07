import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from shared_utils import vector_store, chunk_documents, UPLOADED_FILE_RECORD


def save_uploaded_file_record(filename, full_path):
    with open(UPLOADED_FILE_RECORD, "a") as f:
        f.write(f"{filename}|{full_path}\n")


def get_uploaded_files():
    if not os.path.exists(UPLOADED_FILE_RECORD):
        return []
    with open(UPLOADED_FILE_RECORD, "r") as f:
        return [line.strip().split("|") for line in f.readlines()]


def delete_file_record(filename):
    files = get_uploaded_files()
    with open(UPLOADED_FILE_RECORD, "w") as f:
        for name, path in files:
            if name != filename:
                f.write(f"{name}|{path}\n")


def load_documents_from_file(file_path):
    # Get file extension in lowercase
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Select loader based on extension
    loader = None
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        print(f"Unsupported file type: {ext}")
        return None

    try:
        return loader.load()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def check_duplicate_file(file_name):
    for name, _ in get_uploaded_files():
        if name == file_name:
            return True
    return False


def file_upload_handler(input_file):
    if input_file is None:
        return "❌ Please select a file to upload"

    file_path = input_file.name if hasattr(input_file, "name") else str(input_file)

    if not os.path.exists(file_path):
        return "❌ File does not exist"

    file_name = os.path.basename(file_path)
    print(f"File name: {file_name}")

    # Check for duplicate file
    if check_duplicate_file(file_name):
        return f"❌ File '{file_name}' is already uploaded. Please select a different file or remove the existing file first."

    # Load documents from file
    print(f"Extracting docs from uploaded file: {file_path}")
    docs = load_documents_from_file(file_path)

    if not docs:
        return "❌ No docs found from the uploaded file"

    for doc in docs:
        doc.metadata["source"] = file_name

    # Chunk documents
    print("Chunking docs...")
    doc_splits = chunk_documents(docs)

    # Add documents to vector store
    print("Adding docs to vector store...")
    vector_store.add_documents(doc_splits)

    # Save record
    print(f"Saving record for file: {file_name}")
    save_uploaded_file_record(file_name, file_path)

    return f"✅ {file_name} processed and embedded successfully!"
