from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from shared_utils import vector_store


# set up the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# set up the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create a system message
system_template = """You are a helpful AI assistant for developer documentation. 
When users ask questions:
- If you have relevant information from the knowledge base, provide detailed, accurate answers
- If you don't have specific information, politely explain what you can help with based on available documents
- Always be helpful and professional
- If asked about topics not in your knowledge base, suggest what documents might be useful to upload

Here are some relevant documents:
{context}

Current conversation: {chat_history}
Human: {question}
Assistant:"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)
# set up the conversation chain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
)


def handle_chat(message, history):
    try:
        # Try to get response from knowledge base
        response = conversation_chain.invoke({"question": message})
        answer = response["answer"]

        # If answer is too generic, provide helpful response
        if any(phrase in answer.lower() for phrase in ["don't know", "no information"]):
            return "I don't have specific information about that in my current knowledge base. However, I can help you with questions about the documents you've uploaded."

        return answer
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try again."
