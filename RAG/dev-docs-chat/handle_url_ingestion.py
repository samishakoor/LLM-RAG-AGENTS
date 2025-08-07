from urllib.parse import urlparse
import os
from langchain_community.document_loaders import UnstructuredURLLoader
import requests
from shared_utils import chunk_and_embed_docs, UPLOADS_DIR, upload_url_dir


def save_uploaded_url_record(url):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    with open(upload_url_dir, "a") as f:
        f.write(f"{url.strip()}\n")


def get_uploaded_urls():
    if not os.path.exists(upload_url_dir):
        return []
    with open(upload_url_dir, "r") as f:
        return [line.strip() for line in f.readlines()]

def delete_url_record(url):
    urls = get_uploaded_urls()
    if not urls:
        return
    with open(upload_url_dir, "w") as f:
        for u in urls:
            if u != url:
                f.write(f"{u}\n")


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


def check_duplicate_url(url):
    for u in get_uploaded_urls():
        if u == url:
            return True
    return False


def url_upload_handler(url):
    input_url = url.strip()
    if not input_url:
        return "❌ Please enter a URL to ingest"

    # Check for duplicate URL
    if check_duplicate_url(input_url):
        return f"❌ URL '{input_url}' is already ingested. Please enter a different URL or remove the existing URL first."

    # Validate URL
    validation_result = validate_and_check_url(input_url)
    if not validation_result:
        return "❌ URL is not valid"

    try:
        # Load documents from URL
        print(f"Extracting docs from URL: {input_url}")
        loader = UnstructuredURLLoader(urls=[input_url])

        docs = loader.load()

        if not docs:
            return "❌ No documents found from the URL"

        # Chunk and embed documents
        chunk_and_embed_docs(docs, source_type="url", source_path=input_url)

        # Save record
        print(f"Saving record for URL: {input_url}")
        save_uploaded_url_record(input_url)

        return f"✅ {input_url} processed and embedded successfully!"

    except Exception as e:
        print(f"Error processing URL {input_url}: {e}")
        return f"❌ Error processing URL: {str(e)}"
