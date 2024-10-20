import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
#from dotenv import load_dotenv
#import os

#load_dotenv()

#api_key = os.getenv("TOGETHER_API_KEY")
#if not api_key:
#    raise ValueError("TOGETHER_API_KEY is not set in the environment")

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=5, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you with cooking today?"}
    ]

# Initialize ChatOpenAI and ConversationChain
llm = ChatOpenAI(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    openai_api_key=st.secrets["TOGETHER_API_KEY"],
    openai_api_base="https://api.together.xyz/v1"
)

conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

# Create user interface
st.title("🤖 CookTasty: Your Culinary AI Assistant")
st.subheader("Ask me anything about cooking!")

# User input
if prompt := st.chat_input("What cooking question do you have?"):
    if prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt})

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Define system message for assistant's behavior
system_message = "You are CookTasty, a helpful AI assistant specializing in cooking. " \
                 "You answer questions in a funny and engaging way with unusual culinary analogies. " \
                 "You only answer questions related to cooking. " \
                 "For non-cooking questions, respond with 'I'm sorry, but I can only help with cooking-related questions. " \
                 "Would you like to know how to make a delicious meal instead?'"

# Generate assistant response if the last message is from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Cooking up a response..."):
            # Prepare the conversation history
            conversation_history = [SystemMessage(content=system_message)]
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    conversation_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    conversation_history.append(AIMessage(content=msg["content"]))
            
            # Generate response
            response = llm.invoke(conversation_history).content
            
            st.write(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.buffer_memory.save_context({"input": prompt}, {"output": response})