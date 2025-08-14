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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class StreamlitRAGApp:
    """Streamlit-based RAG application with document upload and chat functionality."""

    def __init__(self):
        """Initialize the Streamlit RAG app."""
        self.vector_service = VectorService()
        self.pdf_tool = PDFExtractionTool()
        self.docx_tool = DOCXExtractionTool()
        self.text_tool = TextExtractionTool()
        self.csv_tool = CSVExtractionTool()
        self.excel_tool = ExcelExtractionTool()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Initialize session state
        if "current_resource_id" not in st.session_state:
            st.session_state.current_resource_id = str(uuid.uuid4())
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = None

        # Initialize RAG chain for current resource
        self._initialize_rag_chain()

    def _initialize_rag_chain(self):
        """Initialize RAG chain for the current resource."""
        try:
            resource_id = uuid.UUID(st.session_state.current_resource_id)
            st.session_state.rag_chain = RAGChain(resource_id, self.vector_service)
            st.success(
                f"✅ RAG Chain initialized for resource: {st.session_state.current_resource_id[:8]}..."
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG Chain: {str(e)}")

    def _create_new_resource(self):
        """Create a new resource ID and initialize RAG chain."""
        new_resource_id = str(uuid.uuid4())
        st.session_state.current_resource_id = new_resource_id
        st.session_state.chat_history = []
        self._initialize_rag_chain()
        st.rerun()

    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename."""
        return Path(filename).suffix.lower()

    def _extract_documents(self, uploaded_file) -> List[Document]:
        """Extract documents from uploaded file based on file type."""
        file_extension = self._get_file_extension(uploaded_file.name)

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            documents = []

            if file_extension == ".pdf":
                documents = self.pdf_tool.extract_text(
                    tmp_file_path, uploaded_file.name
                )
            elif file_extension == ".docx":
                documents = self.docx_tool.extract_text(
                    tmp_file_path, uploaded_file.name
                )
            elif file_extension in [".txt", ".md"]:
                documents = self.text_tool.extract_text(
                    tmp_file_path, uploaded_file.name
                )
            elif file_extension == ".csv":
                documents = self.csv_tool.extract_text(
                    tmp_file_path, uploaded_file.name
                )
            elif file_extension in [".xlsx", ".xls"]:
                documents = self.excel_tool.extract_text(
                    tmp_file_path, uploaded_file.name
                )
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return []

            return documents

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def _process_and_store_documents(self, documents: List[Document]):
        """Process documents and store them in the vector database."""
        if not documents:
            return

        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            st.info(f"📄 Split {len(documents)} documents into {len(chunks)} chunks")

            # Store in vector database
            resource_id = uuid.UUID(st.session_state.current_resource_id)
            result = self.vector_service.store_documents(chunks, resource_id)

            st.success(
                f"✅ Successfully stored {result['document_count']} chunks in collection: {result['collection_name']}"
            )

        except Exception as e:
            st.error(f"❌ Failed to process and store documents: {str(e)}")

    def render_document_upload_tab(self):
        """Render the document upload tab."""
        st.header("📚 Document Upload & Knowledge Base")

        # Resource management section
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(
                f"**Current Resource ID:** `{st.session_state.current_resource_id[:8]}...`"
            )
        with col2:
            if st.button("🆕 New Resource", type="primary"):
                self._create_new_resource()

        st.markdown("---")

        # File upload section
        st.subheader("Upload Documents")
        st.markdown(
            """
        Supported file types:
        - **PDF** (.pdf) - Text extraction with OCR support
        - **Word** (.docx) - Document processing
        - **Text** (.txt, .md) - Plain text and markdown
        - **CSV** (.csv) - Tabular data
        - **Excel** (.xlsx, .xls) - Spreadsheet data
        """
        )

        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "docx", "txt", "md", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Select one or more files to add to your knowledge base",
        )

        if uploaded_files:
            if st.button("🚀 Process & Store Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    total_documents = 0

                    for uploaded_file in uploaded_files:
                        st.write(f"📄 Processing: {uploaded_file.name}")

                        try:
                            documents = self._extract_documents(uploaded_file)
                            if documents:
                                self._process_and_store_documents(documents)
                                total_documents += len(documents)
                        except Exception as e:
                            st.error(
                                f"❌ Failed to process {uploaded_file.name}: {str(e)}"
                            )

                    if total_documents > 0:
                        st.success(
                            f"🎉 Successfully processed {total_documents} documents!"
                        )

        # Collection information
        st.markdown("---")
        st.subheader("📊 Collection Information")

        try:
            resource_id = uuid.UUID(st.session_state.current_resource_id)
            collection_name = self.vector_service.get_collection_name(resource_id)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Collection Name", collection_name)
            with col2:
                st.metric(
                    "Resource ID", st.session_state.current_resource_id[:8] + "..."
                )

            # Show collection stats if available
            try:
                # This would require additional methods in VectorService to get collection stats
                st.info("ℹ️ Collection statistics would be displayed here")
            except:
                pass

        except Exception as e:
            st.error(f"❌ Error retrieving collection information: {str(e)}")

    def render_chat_tab(self):
        """Render the chat interface tab."""
        st.header("💬 Chat Interface")

        # Check if RAG chain is initialized
        if st.session_state.rag_chain is None:
            st.warning(
                "⚠️ Please upload some documents first to initialize the RAG system."
            )
            return

        # Display current resource info
        st.info(
            f"**Chatting with knowledge base:** `{st.session_state.current_resource_id[:8]}...`"
        )

        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")

        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Get response from RAG chain
            with st.spinner("🤔 Thinking..."):
                try:
                    answer = st.session_state.rag_chain.query_documents(user_input)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}. Please try again."
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )

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
            if st.button("🗑️ Clear Chat History"):
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

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Settings")

            # Resource management
            st.subheader("Resource Management")
            st.info(f"Current: `{st.session_state.current_resource_id[:8]}...`")

            if st.button("🆕 New Resource", key="sidebar_new"):
                self._create_new_resource()

            st.markdown("---")

            # System info
            st.subheader("ℹ️ System Info")
            st.markdown(
                """
            - **Vector Database:** PostgreSQL + pgvector
            - **Embedding Model:** text-embedding-3-small
            - **LLM:** OpenAI GPT (configurable)
            - **Chunking:** Recursive character splitter
            """
            )

        # Main content tabs
        tab1, tab2 = st.tabs(["📚 Document Upload", "💬 Chat Interface"])

        with tab1:
            self.render_document_upload_tab()

        with tab2:
            self.render_chat_tab()


def main():
    """Main application entry point."""
    try:
        app = StreamlitRAGApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application failed to start: {str(e)}")
        st.error("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
