from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from shared_utils import PINECONE_INDEX_NAME, chunk_documents

load_dotenv()

# Initialize pinecone vectorstore
pc = Pinecone()

# Create pinecone index (index is a collection of vectors) if it doesn't exist
if not pc.has_index(PINECONE_INDEX_NAME):
    print("CREATING NEW INDEX ...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # dimensions of text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(embedding=OpenAIEmbeddings(), index=index)


def chunk_and_embed_documents(docs, source_type, source_path):
    """Chunk and embed documents to the vector store"""
    # Add metadata to documents
    for doc in docs:
        doc.metadata["source_type"] = source_type
        doc.metadata["source_path"] = source_path

    # Chunk documents
    chunks = chunk_documents(docs)

    # Add documents to vector store
    print("Adding docs to vector store...")
    vectorstore.add_documents(chunks)


def delete_documents_by_source(source):
    """Delete embeddings from vector store based on source path."""
    index.delete(filter={"source_path": source})


def clear_vectorstore():
    """Clear all documents from the vector store"""
    index.delete(delete_all=True)
