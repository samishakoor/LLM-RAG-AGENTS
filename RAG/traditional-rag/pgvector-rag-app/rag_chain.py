from uuid import UUID
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from vector_service import VectorService
from config import settings


class RAGChain:
    """Complete RAG system implementation using LangChain, pgvector, and OpenAI."""

    def __init__(self, resource_id: UUID, vector_service: VectorService):
        """
        Initialize RAG chain for a specific resource.

        Args:
            project_id: Project UUID for namespace isolation
            vector_service: Existing VectorService instance
        """
        self.resource_id = resource_id
        self.vector_service = vector_service

        # Initialize LLM with streaming support
        self.llm = ChatOpenAI(
            model=settings.MODEL, streaming=True, openai_api_key=settings.OPENAI_API_KEY
        )

        # Create project-specific retriever using existing VectorService
        self.retriever = self.vector_service.create_retriever(
            resource_id=resource_id,
            search_kwargs={"k": 10},  # Retrieve top 10 relevant chunks
        )

        # set up the conversation memory for the chat with explicit output key
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

        # Build the RAG chain
        self.chain = self._build_rag_chain()

    def _build_rag_chain(self):
        """Build the RAG chain following LangChain patterns."""

        # Create a system message that includes the context
        system_message = """
        You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer
    

        Use the following pieces of context to answer the user's question. 
        ----------------
        {context}
        """

        # Create the prompt template using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "{question}"),
            ]
        )

        # set up the conversation chain with memory handling
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

        return rag_chain

    def query_documents(self, message: str) -> str:
        """Query the documents using this RAG chain instance."""
        try:
            response = self.chain.invoke({"question": message})
            answer = response.get("answer", "")
            return answer
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."
