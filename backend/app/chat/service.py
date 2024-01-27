import os
import json
from llamaapi import LlamaAPI
from langchain.llms import HuggingFaceHub
from langchain_experimental.llms import ChatLlamaAPI 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_sql_agent
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.memory import ConversationBufferMemory
# from langchain.agents import AgentExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
load_dotenv()

conversation=None
    
class ChatService:
     def get_answer(self, message):
          return chatMessage(message)



def chatMessage(message):
    # LLM
    llm = load_llama_model()
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human."
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    global conversation
    if conversation is None:
        conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
  
    obj= conversation({"question": message})
    key_to_convert = "chat_history"

    if key_to_convert in obj:
        obj[key_to_convert] = str(obj[key_to_convert])
    print(json.dumps(obj))
    return json.dumps(obj)
     

def load_google_ai():
    print(os.getenv('GOOGLE_API_KEY'))
    if "GOOGLE_API_KEY" not in os.environ:
        print("key not found")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True)

    return llm


def load_llama_model():

    llama = LlamaAPI(os.environ.get("LLAMA_API_TOKEN"))

    llm = ChatLlamaAPI(client=llama, verbose=True)

    return llm

def load_mistral_model():
    repo_id = "mistralai/Mistral-7B-v0.1"
    huggingfacehubKey=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehubKey, 
    repo_id=repo_id, model_kwargs={"temperature":0.1})
    return llm

def load_flant5_model():
    repo_id = "google/flan-t5-xxl"
    huggingfacehubKey=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehubKey, 
    repo_id=repo_id, model_kwargs={"temperature":0.1, "max_new_tokens":50})
    return llm

