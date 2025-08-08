from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from shared_utils import vector_store

# set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# set up the conversation memory for the chat with explicit output key
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

# set up the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create a system message that includes the context
system_message = """
You are a helpful AI assistant for developer documentation. 
Guidelines:
1. If you have relevant information from the knowledge base, provide detailed, accurate answers.
2. If you don't have specific information in current documents, check the chat history for relevant previous conversations.
3. If you find relevant information in chat history, acknowledge that the original documents were removed but share the information from our previous conversation.
4. If asked about topics not in your knowledge base or chat history, suggest what documents might be useful to upload.
5. If the user asks about a topic that is not related to the documents, politely explain that you are not able to answer that question.
6. Always be helpful and professional.

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
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
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
