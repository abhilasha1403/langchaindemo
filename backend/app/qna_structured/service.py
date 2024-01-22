import os

from llamaapi import LlamaAPI
from langchain.llms import HuggingFaceHub
from langchain_experimental.llms import ChatLlamaAPI 
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_experimental.sql.base import SQLDatabaseSequentialChain
from langchain import hub
from langchain.agents import create_sql_agent

# from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
load_dotenv()


class QnAStructuredService:
     def get_answer(self, query):
          return querydb(query)

def querydb(query):
    db=fetchdb()
    llm=load_google_ai()
    agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    #chain = create_sql_query_chain(llm, db )
    #response = chain.invoke({"question":query})
    #print(response)
    #db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db,verbose=True)
    answer=agent_executor.run(query)
    return answer
         


def fetchdb(
    
):
    try:
      db = SQLDatabase.from_uri(f"postgresql+psycopg2://postgres:admin@localhost:5432/dvdrental",
                                include_tables=["actor", "film","film_actor","film_category","category"])
    except Exception as e:
        raise Exception(
            f"Failed to connect postgres db: {str(e)}"
        )

   
    return db


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
    repo_id=repo_id, model_kwargs={"temperature":0.1, "max_new_tokens":50})
    return llm

def load_flant5_model():
    repo_id = "google/flan-t5-xxl"
    huggingfacehubKey=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehubKey, 
    repo_id=repo_id, model_kwargs={"temperature":0.1, "max_new_tokens":50})
    return llm