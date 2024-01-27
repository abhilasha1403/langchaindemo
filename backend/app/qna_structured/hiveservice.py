import os

from llamaapi import LlamaAPI
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
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
from sqlalchemy import create_engine
# from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Chroma
load_dotenv()




class QnAHiveService:
     def get_answer(self, query):
          return querydb(query)



def create_retrieval_qa_bot(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    persist_dir="C:\\Users\\abhil\\DeepLearning\\Repo\\langchaindemo\\backend\\hivedb",
    device="cpu",
):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name : {str(e)}"
        )

    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print(db.get())

    return db.as_retriever()


def querydb(query):
   
    db=fetchhivedb()
    prompt = """ 
Given an input question, first create a syntactically correct hql query to run
then look at the results of the query and return the answer. 
The question: {question}
"""
    #retriever =create_retrieval_qa_bot()
    # docs = retriever.get_relevant_documents(query)
    # print(docs)
    # relevant_tables=""
    # for document in docs:
    #      relevant_tables =relevant_tables+"/n"+ str(document.page_content) 

    # prompt=prompt.format(question=query,table_info=relevant_tables)
    prompt=prompt.format(question=query)
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
    print(prompt)
    answer=agent_executor.run(prompt)
    return answer
         
def fetchhivedb():
    engine = create_engine(f'hive://localhost:10000/dvdrental')
    db = SQLDatabase(engine)
    return db;




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

def load_vicuna_model():
    repo_id = "lmsys/vicuna-33b-v1.3"
    huggingfacehubKey=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehubKey, 
    repo_id=repo_id, model_kwargs={"temperature":0.1})
    return llm

def load_flant5_model():
    repo_id = "google/flan-t5-xxl"
    huggingfacehubKey=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehubKey, 
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 256})
    return llm