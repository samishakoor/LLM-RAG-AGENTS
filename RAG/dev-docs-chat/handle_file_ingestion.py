import os
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    TextLoader,
    PyPDFLoader,
)
from shared_utils import chunk_and_embed_docs, UPLOADS_DIR, upload_file_dir


def save_uploaded_file_record(filename):
    """Save file record to the uploaded files record file."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    with open(upload_file_dir, "a") as f:
        f.write(f"{filename.strip()}\n")


def get_uploaded_files():
    """Retrieve list of all uploaded files from the record file."""
    if not os.path.exists(upload_file_dir):
        return []
    with open(upload_file_dir, "r") as f:
        files = [line.strip() for line in f.readlines()]
        return files


def delete_file_record(filename):
    """Remove a specific file from the uploaded files record file."""
    files = get_uploaded_files()
    if not files:
        return
    with open(upload_file_dir, "w") as f:
        for name in files:
            if name != filename:
                f.write(f"{name}\n")


def load_documents_from_file(file_path):
    """Load and parse documents from a file based on its extension."""
    # Get file extension in lowercase
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Select loader based on extension
    loader = None
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".md", ".markdown"]:
        loader = UnstructuredMarkdownLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        print(f"Unsupported file type: {ext}")
        return None

    try:
        return loader.load()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def check_duplicate_file(file_name):
    """Check if a file has already been uploaded."""
    for name in get_uploaded_files():
        if name == file_name:
            return True
    return False


def file_upload_handler(input_file):
    """Process and embed documents from an uploaded file."""
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

    try:
        # Load documents from file
        print(f"Extracting docs from uploaded file: {file_path}")
        docs = load_documents_from_file(file_path)

        if not docs:
            return "❌ No docs found from the uploaded file"

        chunk_and_embed_docs(docs, source_type="file", source_path=file_path)

        # Save record
        print(f"Saving record for file: {file_name}")
        save_uploaded_file_record(file_name)

        return f"✅ {file_name} processed and embedded successfully!"
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return f"❌ Error processing file: {str(e)}"
