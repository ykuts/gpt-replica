from langchain_core.messages import SystemMessage
from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from dotenv import load_dotenv
import os

# Загружаем переменные окружения
load_dotenv()

from langchain_openai import ChatOpenAI

# Проверяем наличие ключа
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY is not set in the environment")

llama_model = ChatOpenAI(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    openai_api_key=api_key,
    openai_api_base="https://api.together.xyz/v1"
)

system_message = """You are a BearBot, a helpful AI assistant created by Build Fast with AI.
You answer questions in a funny and engaging way with unusual analogies.
You don't answer any questions not related to AI. Please respond with 'I cannot answer the question' for non-AI questions.
"""

memory = ConversationBufferMemory(k = 3)

conversation = ConversationChain(
    llm=llama_model,
    memory = memory
)


# Add the system message to the conversation's memory
conversation.memory.chat_memory.add_message(SystemMessage(content=system_message))

# Now run the conversation with just the human message
prompt = "Who are you?"
#response = conversation.invoke(input=prompt)
response = conversation.run(input=prompt)
print(response)

conversation.memory

chat_history = conversation.memory.chat_memory.messages

chat_history

def display_chat_history(chat_history):
    for message in chat_history:
        role = "Human" if message.__class__.__name__ == "HumanMessage" else "AI"
        print(f"{role}: {message.content}")
        print("-" * 50)  # Separator between messages

# Assuming chat_history is your variable containing the messages
display_chat_history(chat_history)