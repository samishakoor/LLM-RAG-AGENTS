from urllib.parse import urlparse
import gradio as gr
import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import shutil

import requests

load_dotenv()

COLLECTION_NAME = "dev_docs"

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_metadata={"hnsw:space": "cosine"},
)

UPLOADED_FILE_RECORD = "uploaded_files.txt"
UPLOADED_URL_RECORD = "uploaded_urls.txt"


def save_uploaded_file_record(filename, full_path):
    with open(UPLOADED_FILE_RECORD, "a") as f:
        f.write(f"{filename}|{full_path}\n")


def save_uploaded_url_record(url):
    with open(UPLOADED_URL_RECORD, "a") as f:
        f.write(f"{url}\n")


def get_uploaded_files():
    if not os.path.exists(UPLOADED_FILE_RECORD):
        return []
    with open(UPLOADED_FILE_RECORD, "r") as f:
        return [line.strip().split("|") for line in f.readlines()]


def get_uploaded_urls():
    if not os.path.exists(UPLOADED_URL_RECORD):
        return []
    with open(UPLOADED_URL_RECORD, "r") as f:
        return [line.strip() for line in f.readlines()]


def delete_file_record(filename):
    files = get_uploaded_files()
    with open(UPLOADED_FILE_RECORD, "w") as f:
        for name, path in files:
            if name != filename:
                f.write(f"{name}|{path}\n")


def delete_url_record(url):
    urls = get_uploaded_urls()
    with open(UPLOADED_URL_RECORD, "w") as f:
        for u in urls:
            if u != url:
                f.write(f"{u}\n")


def cleanup():
    print("Shutting down...")

    # Delete the file that stores uploaded file paths
    if os.path.exists(UPLOADED_FILE_RECORD):
        os.remove(UPLOADED_FILE_RECORD)


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


def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits


def delete_document(file_name):
    if not file_name:
        return "❌ Please select a file to delete from the dropdown"

    # Remove from vector DB
    vector_store.delete(where={"source": file_name})

    # Remove temp file
    uploaded_files = get_uploaded_files()
    for name, path in uploaded_files:
        if name == file_name and os.path.exists(path):
            os.remove(path)
            break

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


def file_upload_handler(input_file):
    if input_file is None:
        return "❌ Please select a file to upload"

    file_path = input_file.name if hasattr(input_file, "name") else str(input_file)

    if not os.path.exists(file_path):
        return "❌ File does not exist"

    file_name = os.path.basename(file_path)
    print(f"File name: {file_name}")

    # Check for duplicate file
    for name, _ in get_uploaded_files():
        if name == file_name:
            return f"❌ File '{file_name}' is already uploaded. Please select a different file or remove the existing file first."

    # Load documents from file
    print("Extracting docs from uploaded file...")
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
    save_uploaded_file_record(file_name, file_path)
    return f"✅ {file_name} processed and embedded successfully!"


def validate_and_check_url(url):
    # Step 1: Basic format validation
    if not url.startswith("https://"):
        print(f"URL must start with https://: {url}")
        return False

    # Step 2: Check if it's a well-formed URL
    parsed = urlparse(url)
    if not parsed.netloc:
        print(f"Not a valid URL: {url}")
        return False

    # Step 3: Check if the URL is reachable
    try:
        response = requests.head(url, timeout=15)
        if response.status_code < 400:
            print(f"URL is valid and reachable: {url}")
            return True
        else:
            print(f"URL responded with status: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"URL is not reachable: {e}")
        return False


def url_upload_handler(url):
    if not url or url.strip() == "":
        return "❌ Please enter a URL to ingest"

    # Check for duplicate URL
    if url in get_uploaded_urls():
        return f"❌ URL '{url}' is already ingested. Please enter a different URL or remove the existing URL first."

    # Validate URL
    validation_result = validate_and_check_url(url.strip())
    if not validation_result:
        return "❌ URL is not valid"

    try:
        # Load documents from URL
        print(f"Loading documents from URL: {url}")
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()

        if not docs:
            return "❌ No documents found from the URL"

        # Chunk documents
        print("Chunking documents...")
        doc_splits = chunk_documents(docs)

        # Add documents to vector store
        print("Adding docs to vector store...")
        vector_store.add_documents(doc_splits)

        # Save record
        save_uploaded_url_record(url)

        return f"✅ {url} processed and embedded successfully!"

    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return f"❌ Error processing URL: {str(e)}"


def clear_all_data():
    try:
        # Clear all documents from vector store
        all_docs = vector_store.get()
        if "ids" in all_docs and all_docs["ids"]:
            # Delete all documents by their IDs
            vector_store.delete(all_docs["ids"])
            print(f"Deleted {len(all_docs['ids'])} documents from vectorstore")
        else:
            print("Vectorstore is already empty")
        
        # Delete all uploaded files
        uploaded_files = get_uploaded_files()
        for _, path in uploaded_files:
            if os.path.exists(path):
                os.remove(path)
        print("Deleted all uploaded files")

        # Clear uploaded files record
        if os.path.exists(UPLOADED_FILE_RECORD):
            os.remove(UPLOADED_FILE_RECORD)
        print("Deleted uploaded files record")

        # Clear uploaded URLs record
        if os.path.exists(UPLOADED_URL_RECORD):
            os.remove(UPLOADED_URL_RECORD)
        print("Deleted uploaded URLs record")

        print("All data cleared successfully!")
    except Exception as e:
        print(f"Error clearing data: {str(e)}")


# --- Tab 1: Upload Documents UI ---
def upload_ui():
    with gr.Row():
        file_input = gr.File(label="Upload File", file_types=[".pdf", ".md", ".txt"])
    output = gr.Textbox(label="Status")

    upload_button = gr.Button("Ingest File")
    upload_button.click(fn=file_upload_handler, inputs=file_input, outputs=output)

    return file_input, upload_button, output


# --- Tab 2: Upload URLs UI ---
def url_upload_ui():
    with gr.Row():
        url_input = gr.Textbox(label="Enter URL", placeholder="https://example.com")
    output = gr.Textbox(label="Status")

    upload_button = gr.Button("Ingest URL")
    upload_button.click(fn=url_upload_handler, inputs=url_input, outputs=output)

    return url_input, upload_button, output


# --- Tab 3: Document Management UI ---
def manage_ui():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Files")
            file_dropdown = gr.Dropdown(choices=[], label="Uploaded Files")
            delete_file_btn = gr.Button("Delete Selected File")
            refresh_files_btn = gr.Button("Refresh File List")
            file_delete_output = gr.Textbox(
                label="File Delete Status", lines=2, visible=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### URLs")
            url_dropdown = gr.Dropdown(choices=[], label="Ingested URLs")
            delete_url_btn = gr.Button("Delete Selected URL")
            refresh_urls_btn = gr.Button("Refresh URL List")
            url_delete_output = gr.Textbox(
                label="URL Delete Status", lines=2, visible=False
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Nuclear Option - Clear All Data")
            gr.Markdown(
                "⚠️ **Warning:** This will delete ALL uploaded files, ingested URLs, and clear the entire vector database. This action cannot be undone."
            )
            clear_all_btn = gr.Button("Clear All Data", variant="stop")

    def update_file_dropdown_choices():
        return gr.update(choices=[f[0] for f in get_uploaded_files()])

    def update_url_dropdown_choices():
        return gr.update(choices=get_uploaded_urls())

    def show_file_status(result):
        return gr.update(value=result, visible=True)

    def show_url_status(result):
        return gr.update(value=result, visible=True)

    delete_file_btn.click(
        fn=delete_document, inputs=[file_dropdown], outputs=[file_delete_output]
    )
    delete_file_btn.click(
        fn=show_file_status, inputs=[file_delete_output], outputs=[file_delete_output]
    )
    delete_file_btn.click(
        fn=update_file_dropdown_choices, inputs=[], outputs=[file_dropdown]
    )

    delete_url_btn.click(
        fn=delete_url, inputs=[url_dropdown], outputs=[url_delete_output]
    )
    delete_url_btn.click(
        fn=show_url_status, inputs=[url_delete_output], outputs=[url_delete_output]
    )
    delete_url_btn.click(
        fn=update_url_dropdown_choices, inputs=[], outputs=[url_dropdown]
    )

    clear_all_btn.click(fn=clear_all_data, inputs=[], outputs=[])
    clear_all_btn.click(
        fn=update_file_dropdown_choices, inputs=[], outputs=[file_dropdown]
    )
    clear_all_btn.click(
        fn=update_url_dropdown_choices, inputs=[], outputs=[url_dropdown]
    )

    refresh_files_btn.click(
        fn=update_file_dropdown_choices, inputs=[], outputs=[file_dropdown]
    )
    refresh_urls_btn.click(
        fn=update_url_dropdown_choices, inputs=[], outputs=[url_dropdown]
    )

    return (
        file_dropdown,
        delete_file_btn,
        file_delete_output,
        update_file_dropdown_choices,
        url_dropdown,
        delete_url_btn,
        refresh_files_btn,
        refresh_urls_btn,
        update_url_dropdown_choices,
    )


# --- Main Interface ---
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Upload Document"):
            upload_ui()
        with gr.TabItem("Ingest from URL"):
            url_upload_ui()
        with gr.TabItem("Manage Data"):
            (
                file_dropdown,
                delete_file_btn,
                delete_output,
                update_file_choices_fn,
                url_dropdown,
                delete_url_btn,
                refresh_files_btn,
                refresh_urls_btn,
                update_url_choices_fn,
            ) = manage_ui()

    # Update dropdowns when app loads
    demo.load(fn=update_file_choices_fn, inputs=[], outputs=[file_dropdown])
    demo.load(fn=update_url_choices_fn, inputs=[], outputs=[url_dropdown])

demo.launch()

# Add a callback to run cleanup when the app shuts down
demo.close(fn=cleanup)
