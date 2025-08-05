import os
import glob
from typing import TypedDict, Annotated, Sequence
import uuid
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class DrugTreatmentRAGAgent:
    def __init__(self):
        self.db_name = "drug_treatment_db"
        self.collection_name = "drug_treatment_docs"
        self.folder_path = "drug-knowledge-base"
        self.llm_with_tools = None
        self.max_retrieval_docs = 5
        self.retriever = None
        self.memory = MemorySaver()
        self.thread_id = str(uuid.uuid4())
        self.tools = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.rag_agent_graph = None
        
        
    def load_documents(self):
        """
        Load PDF documents from the specified directory using PyPDFDirectoryLoader.

        Returns:
        List of Document objects: Loaded PDF documents represented as Langchain Document objects.
        """
        
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Knowledge base folder not found: {self.folder_path}")

        documents = []
        pdf_files = glob.glob(os.path.join(self.folder_path, "**", "*.pdf"), recursive=True)
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)        # PyPDFLoader only extracts text from pdf file
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDF {pdf_file}: {e}")
                raise

        print(f"Found {len(documents)} pages in {len(pdf_files)} PDF documents")
        return documents

    def split_documents(self, documents):
        """
        Split the text content of the given list of Document objects into smaller chunks.

        Args:
            documents: List of Document objects containing text content to split.

        Returns:
            List of Document objects representing the split text chunks.
        """
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"{len(chunks)} Chunks created")
        return chunks
    
    def save_to_vector_store(self, chunks):
        """
        Save the split text chunks to a vector store.

        Args:
            chunks: List of Document objects representing the split text chunks.
        """
        
        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, embedding_function=self.embeddings).delete_collection()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_name,
            collection_name=self.collection_name,
        )
        print(f"Vectorstore created with {vectorstore._collection.count()} documents")
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.max_retrieval_docs})

    def load_existing_vector_store(self):
        """Load existing vector store without recreating"""
        
        vectorstore = Chroma(
            persist_directory=self.db_name,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        print(f"Loaded existing vectorstore with {vectorstore._collection.count()} documents")
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.max_retrieval_docs})
        
    def setup_knowledge_base(self):
        """Smart knowledge base setup - only recreate if needed"""
        
        if os.path.exists(self.db_name):
            print("Loading existing knowledge base...")
            self.load_existing_vector_store()
        else:
            print("Creating new knowledge base...")
            documents = self.load_documents()
            chunks = self.split_documents(documents)
            self.save_to_vector_store(chunks)

    def setup_llm_and_tools(self):
        """Setup the LLM and tools"""
        @tool
        def retriever_tool(query: str) -> str:
            """
            Useful for retrieving authoritative medical and clinical information about how to treat or cure substance use disorders related to drugs like cocaine, methamphetamine, opioids, and other stimulants or depressants.
            Do not use this tool for general non-medical questions.
            """
            docs = self.retriever.invoke(query)
            if not docs:
                return "No relevant information was found in the available drug treatment and recovery documents for your query."

            results = [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
            return "\n\n".join(results)

        self.tools = [retriever_tool]
        rag_llm = ChatOpenAI(model="gpt-4o-mini")
        self.llm_with_tools = rag_llm.bind_tools(self.tools)

    def build_graph(self):
        """Build the graph"""
        def tool_router(state: AgentState):
            last_message = state["messages"][-1]
            return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

        system_prompt = """
        You are a knowledgeable AI assistant trained to answer questions about the treatment and recovery of substance use disorders, including drugs like cocaine, methamphetamine, opioids, and other addictive substances.
        Your answers should be based strictly on the trusted drug treatment and recovery documents loaded into your knowledge base — including clinical guidelines, research papers, and public health manuals from sources such as SAMHSA, ASAM, NIDA, and the FDA.
        When responding, use only evidence and guidance retrieved from those documents. You have access to a retriever tool that allows you to search relevant sections of these documents to support your responses.
        Do not make assumptions or fabricate information. If the answer is not covered by the documents, state that no relevant information was found.
        """

        def call_rag_llm(state: AgentState) -> AgentState:
            messages = list(state["messages"])
            messages = [SystemMessage(content=system_prompt)] + messages
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("rag_llm", call_rag_llm)
        graph_builder.add_node("tools", ToolNode(self.tools))

        graph_builder.add_conditional_edges("rag_llm", tool_router, {True: "tools", False: END})
        graph_builder.add_edge("tools", "rag_llm")
        graph_builder.set_entry_point("rag_llm")
       
        self.rag_agent_graph = graph_builder.compile(checkpointer=self.memory)

    def run(self):
        """Run the agent"""
        config = {"configurable": {"thread_id": self.thread_id}}
        
        print("\n=== Drug Treatment RAG Agent ===")
        while True:
            user_input = input("\nWhat is your question? ")
            if user_input.lower() in ["exit", "quit"]:
                break

            messages = [HumanMessage(content=user_input)]
            result = self.rag_agent_graph.invoke({"messages": messages}, config=config)

            print("\n=== ANSWER ===")
            print(result["messages"][-1].content)

    def start(self):
        self.setup_knowledge_base()
        self.setup_llm_and_tools()
        self.build_graph()
        self.run()


if __name__ == "__main__":
    agent = DrugTreatmentRAGAgent()
    agent.start()
