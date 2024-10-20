import streamlit as st
from streamlit_chat import message
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import os

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Initialize ChatOpenAI and ConversationChain
# llm = ChatOpenAI(model_name="gpt-4o-mini")
# llm = ChatGoogleGenerativeAI(model = "gemini-pro")
llm = ChatOpenAI(model = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                      openai_api_key = st.secrets["TOGETHER_API_KEY"] , ## use your key
                      openai_api_base = "https://api.together.xyz/v1"

)

conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

# Create user interface
st.title("🤖 Conversational Chatbot")
st.subheader("Simple Chatbot")


# User input
if prompt := st.chat_input("What do you want to ask?"):  # Prompt for user input
    if prompt.strip():  # Ensure the prompt is not empty
        st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Define system message for assistant's behavior
system_message = ("You are a CookTasty, a helpful AI assistant. "
                  "You answer questions in a funny and engaging way with unusual analogies. "
                  "You don't answer any questions not related to cooking. "
                  "Please respond with 'I cannot answer the question' for non-cooking questions.")

# If last message is from the user, generate a response from the assistant
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("I'm thinking..."):
            # Construct conversation messages for the assistant
            messages = [SystemMessage(content=system_message),
                        HumanMessage(content=st.session_state.messages[-1]["content"])]  # Last user message
            
            # Generate response from the assistant
            response = conversation.run(messages)
            st.write(response)
            
            # Append assistant's response to the message history
            st.session_state.messages.append({"role": "assistant", "content": response})
