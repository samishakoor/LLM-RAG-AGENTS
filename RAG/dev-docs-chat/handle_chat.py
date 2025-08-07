from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from shared_utils import vector_store

# set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# set up the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# set up the conversation chain with the LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory
)


# function to handle the chat
def handle_chat(message, history):
    response = conversation_chain.invoke({"question": message})
    return response["answer"]
