
import os
import json
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
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import BaseTool
from typing import Optional
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

class MetadataSearchTool(BaseTool):
    question=""
    name = "sql_db_list_tables"
    description = "Input to this tool is an input question. Output is a json with relevant table names and their schema information on which query should be executed"
    def __init__(self,query):
        # Call the __init__ method of the parent class using super()
        super().__init__()
        global question 
        question=query

    def _run(
        self, user_question: str
    ) -> str:
        relevant_tables=[]
        retriever=create_retrieval_qa_bot()
        global question
        relevant_doc = retriever.get_relevant_documents(question)
        if len(relevant_doc) == 0 or len(question) == 0:
            return "There are no meta data syntax examples to be used in this scenario."
        else:           
            for document in relevant_doc:
             tablename=find_nested_value(json.loads(document.page_content),"tablename")
             relevant_tables.append(tablename) 
        
        result = ', '.join(relevant_tables)
        return result

def find_nested_value(data, target_key):
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for key, value in data.items():
            nested_result = find_nested_value(value, target_key)
            if nested_result is not None:
                return nested_result
    elif isinstance(data, list):
        for item in data:
            nested_result = find_nested_value(item, target_key)
            if nested_result is not None:
                return nested_result
    return None

    async def _arun(
        self, question: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")



def get_answer2():
        user_question = input("What would you like to know from your data?: ")
        metadata_search_tool = MetadataSearchTool(user_question)
        db = fetchhivedb()
        llm = load_google_ai()

        
        prompt = """You are an agent designed to interact with a hive database.
Given an input question, create a syntactically correct hive query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You should first get the table info you know from the sql_db_list_tables by passing question to it.
If the info is enough to construct the query, you can build it.
Otherwise,you can then look at the tables in the database to see what you can query.
Then you should query the schema of the most relevant tables
If the question does not seem related to the database, just return "I don't know" as the answer. Question is {question}""" 

        prompt=prompt.format(question=user_question)
        agent_executor = create_sql_agent(
    llm=llm,
    extra_tools=[metadata_search_tool],
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
        
        
        answer=agent_executor.run(prompt)
        print(answer)
        return ""



def fetchhivedb():
    engine = create_engine(f'hive://localhost:10000/dvdrental')  #change hive connection
    engine.execution_options(set={ "auto.convert.join": "false"})

    db = SQLDatabase(engine)
    return db

def create_retrieval_qa_bot(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    persist_dir="C:\\Users\\abhil\\DeepLearning\\Repo\\langchaindemo\\backend\\hivedb",
    device="cpu",
):
    ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
    DB_DIR: str = os.path.join(ABS_PATH, "hivedb")
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

    return db.as_retriever()

def load_google_ai():
    GOOGLE_API_KEY='AIzaSyBT43TG_ZGK2gGXEH1rq48QXB8EivyLuwY'
    os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY
    llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True,temperature=0)

    return llm

if __name__ == '__main__':
    get_answer2()
