# Run the app with:
# uv run streamlit run app.py 

import streamlit as st
import uuid
import tempfile
import os
from pathlib import Path
from typing import List

# Import existing RAG components
from rag_chain import RAGChain
from vector_service import VectorService
from tools.pdf_tools import PDFExtractionTool
from tools.docx_tools import DOCXExtractionTool
from tools.text_tools import TextExtractionTool
from tools.csv_tools import CSVExtractionTool
from tools.excel_tools import ExcelExtractionTool
from tools.chunking_tools import TextChunkingTool
from langchain_core.documents import Document


# Cache heavy resources so they persist across reruns
@st.cache_resource(show_spinner=False)
def get_vector_service():
    return VectorService()


@st.cache_resource(show_spinner=False)
def get_tools():
    return (
        PDFExtractionTool(),
        DOCXExtractionTool(),
        TextExtractionTool(),
        CSVExtractionTool(),
        ExcelExtractionTool(),
        TextChunkingTool(),
    )


@st.cache_resource(show_spinner=False)
def get_rag_chain(resource_id: str, _vector_service: VectorService):
    resource_uuid = uuid.UUID(resource_id)
    return RAGChain(resource_uuid, _vector_service)


class StreamlitRAGApp:
    """Streamlit-based RAG application with document upload and chat functionality."""

    def __init__(self):
        """Initialize the Streamlit RAG app."""
        self.vector_service = get_vector_service()
        (
            self.pdf_tool,
            self.docx_tool,
            self.text_tool,
            self.csv_tool,
            self.excel_tool,
            self.chunking_tool,
        ) = get_tools()

        self.supported_file_types = {
            "pdf": [".pdf"],
            "docx": [".docx", ".doc"],
            "excel": [".xlsx", ".xls"],
            "text": [".txt"],
            "csv": [".csv"],
            "markdown": [".md", ".markdown"],
        }

        # Use hardcoded resource ID
        self.resource_id = "123e4567-e89b-12d3-a456-426614174000"

        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Initialize RAG chain as instance variable
        print("Initializing RAG chain")
        self._initialize_rag_chain()

    def _initialize_rag_chain(self):
        """Initialize RAG chain for the hardcoded resource."""
        try:
            self.rag_chain = get_rag_chain(self.resource_id, self.vector_service)
            print(f"✅ RAG Chain initialized for resource: {self.resource_id}")
        except Exception as e:
            print(f"❌ Failed to initialize RAG Chain: {str(e)}")
            self.rag_chain = None

    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename."""
        return Path(filename).suffix.lower()

    def _extract_documents(self, uploaded_file, file_extension) -> List[Document]:
        """Extract documents from uploaded file based on file type."""

        # Store the original filename before converting to bytes
        original_filename = uploaded_file.name

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            documents = []

            if file_extension in self.supported_file_types["pdf"]:
                documents = self.pdf_tool.extract_text(tmp_file_path, original_filename)
            elif file_extension in self.supported_file_types["docx"]:
                documents = self.docx_tool.extract_text(
                    tmp_file_path, original_filename
                )
            elif file_extension in self.supported_file_types["text"]:
                documents = self.text_tool.extract_text(
                    tmp_file_path, original_filename
                )
            elif file_extension in self.supported_file_types["markdown"]:
                documents = self.text_tool.extract_markdown(
                    tmp_file_path, original_filename
                )
            elif file_extension in self.supported_file_types["csv"]:
                documents = self.csv_tool.extract_text(tmp_file_path, original_filename)
            elif file_extension in self.supported_file_types["excel"]:
                documents = self.excel_tool.extract_text(
                    tmp_file_path, original_filename
                )
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return []

            return documents

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def _get_file_type(self, file_extension: str) -> str:
        """Get file type from file extension."""
        for file_type, exts in self.supported_file_types.items():
            if file_extension in exts:
                return file_type
        return "text"

    def _chunk_and_store_documents(
        self, documents: List[Document], file_extension: str
    ):
        """Process documents and store them in the vector database."""
        if not documents:
            return

        try:
            # Split documents into chunks
            file_type = self._get_file_type(file_extension)
            chunks = self.chunking_tool.adaptive_chunk(documents, file_type)

            # Store in vector database
            resource_uuid = uuid.UUID(self.resource_id)
            self.vector_service.store_documents(chunks, resource_uuid)

        except Exception as e:
            st.error(f"❌ Failed to process and store documents: {str(e)}")

    def render_document_upload_tab(self):
        """Render the document upload tab."""
        st.header("📚 Document Upload")

        # Display hardcoded resource info
        st.info(f"**Resource ID:** `{self.resource_id}`")

        st.markdown("---")

        # File upload section
        st.subheader("Upload Documents")
        st.markdown(
            """
        Supported file types:
        - **PDF** (.pdf) - Text extraction with OCR support
        - **Word** (.docx, .doc) - Document processing
        - **Text** (.txt, .md, .markdown) - Plain text and markdown
        - **CSV** (.csv) - Tabular data
        - **Excel** (.xlsx, .xls) - Spreadsheet data
        """
        )

        uploaded_file = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "docx", "doc", "txt", "md", "csv", "xlsx", "xls", "markdown"],
            accept_multiple_files=False,
            help="Select a file (in a supported format) to upload and add to your knowledge base.",
        )

        if uploaded_file:
            # Check if processing is already happening
            if "processing_documents" not in st.session_state:
                st.session_state.processing_documents = False

            # Process directly on click; avoid reruns so spinner stays visible
            if st.button(
                "🚀 Ingest File",
                type="primary",
                disabled=st.session_state.processing_documents,
            ):
                st.session_state.processing_documents = True
                st.rerun()

            if st.session_state.processing_documents:
                with st.spinner("Processing documents..."):
                    st.write(f"📄 Processing: {uploaded_file.name}")
                    try:
                        file_extension = self._get_file_extension(uploaded_file.name)
                        documents = self._extract_documents(
                            uploaded_file, file_extension
                        )
                        if documents:
                            self._chunk_and_store_documents(documents, file_extension)
                    except Exception as e:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                    finally:
                        st.session_state.processing_documents = False
                    st.rerun()

    def render_chat_tab(self):
        """Render the chat interface tab."""
        st.header("💬 Chat Interface")

        # Check if RAG chain is initialized
        if self.rag_chain is None:
            st.warning(
                "⚠️ Please upload some documents first to initialize the RAG system."
            )
            return

        # Display current resource info
        st.info(f"**Chatting with knowledge base:** `{self.resource_id}`")

        # Chat input
        # Check if chat is already processing
        if "chat_processing" not in st.session_state:
            st.session_state.chat_processing = False

        # Disable chat input if processing
        user_input = st.chat_input(
            "Ask a question about your documents...",
            disabled=st.session_state.chat_processing,
        )

        if user_input and not st.session_state.chat_processing:
            # Set processing state
            st.session_state.chat_processing = True

            # Add user message to chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Get response from RAG chain
            with st.spinner("🤔 Thinking..."):
                try:
                    result = self.rag_chain.query_documents(user_input)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result["answer"]}
                    )

                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}. Please try again."
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )
                finally:
                    # Reset processing state
                    st.session_state.chat_processing = False
                    st.rerun()

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

        # Clear chat button
        if st.session_state.chat_history:
            # Check if clearing is already happening
            if "clearing_chat" not in st.session_state:
                st.session_state.clearing_chat = False

            if st.button(
                "🗑️ Clear Chat History", disabled=st.session_state.clearing_chat
            ):
                st.session_state.clearing_chat = True
                st.session_state.chat_history = []
                st.rerun()

    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="RAG System - Document Chat",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🤖 RAG System - Document Chat")
        st.markdown("**Resource-based Retrieval Augmented Generation System**")

        # Main content tabs
        tab1, tab2 = st.tabs(["📚 Document Upload", "💬 Chat Interface"])

        with tab1:
            self.render_document_upload_tab()

        with tab2:
            self.render_chat_tab()


def main():
    """Main application entry point."""
    try:
        if "_app_instance" not in st.session_state:
            st.session_state._app_instance = StreamlitRAGApp()
        app = st.session_state._app_instance
        app.run()
    except Exception as e:
        st.error(f"❌ Application failed to start: {str(e)}")


if __name__ == "__main__":
    main()
