import os
from typing import List
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Step 1: Load Environment Variables
load_dotenv()

# Step 2: Load and Split Documents
urls = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://realpython.com/python-basics/",
    "https://www.learnpython.org/"
]

loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

# Step 3: Create Vector Store
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="python_docs",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# Step 4: Define Retrieval and Answer Generation Functions
def retrieve(question: str) -> List[str]:
    documents = retriever.invoke(question)
    return [doc.page_content for doc in documents]

def generate_answer(question: str, context: List[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini")
    context_text = "\n".join(context)
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\n"
    response = llm.invoke(prompt)
    return response

# Step 5: Execute the Workflow
def run_pipeline(question: str) -> str:
    print("Retrieving documents...")
    documents = retrieve(question)
    
    print("Generating answer...")
    answer = generate_answer(question, documents)
    return answer

# Step 6: Chatbot Interface
if __name__ == "__main__":
    print("Welcome to the Python Docs Chatbot! (Type 'exit' to quit)\n")
    
    while True:
        question = input("You: ")
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        
        try:
            answer = run_pipeline(question)
            print(f"AI: {answer.content}\n")
        except Exception as e:
            print(f"Error: {e}\n")