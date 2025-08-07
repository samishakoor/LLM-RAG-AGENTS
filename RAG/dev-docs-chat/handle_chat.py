from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from shared_utils import vector_store

# set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# set up the conversation memory for the chat with explicit output key
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

# set up the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# set up the conversation chain with memory handling
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True,
)


def handle_chat(message, history):
    """Handle the chat conversation"""
    try:
        # Try to get response from knowledge base
        response = conversation_chain.invoke({"question": message})
        return response["answer"]
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try again."
