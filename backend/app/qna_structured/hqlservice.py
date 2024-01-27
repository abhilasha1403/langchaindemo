
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
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_retriever_tool



def get_answer():
        user_question = input("What would you like to know from your data?: ")
        db = fetchhivedb()
        llm = load_google_ai()
        retriever=create_retrieval_qa_bot()

        description = """Use to look up relevant table names and there columns to query upon. Input is a question, output is \
relvant tables and there column definitions in json"""
        retriever_tool = create_retriever_tool(
            retriever,
            name="table_info_tool",
            description=description,
        )    
        system = """You are an agent designed to interact with a hive database.
Given an input question, create a syntactically correct hive query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If you need to get relevant tables and schema information, you must ALWAYS first look up the filter value using the "table_info_tool" tool! 

If the question does not seem related to the database, just return "I don't know" as the answer.""" 
        agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    extra_tools=[retriever_tool],
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    prompt=system,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
        answer=agent_executor.invoke(user_question)
        print(answer)
        return ""



def fetchhivedb():
    engine = create_engine(f'hive://localhost:10000/dvdrental')  #change hive connection
    db = SQLDatabase(engine)
    return db

def create_retrieval_qa_bot(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
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

    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    return db.as_retriever()

def load_google_ai():
    GOOGLE_API_KEY='AIzaSyBT43TG_ZGK2gGXEH1rq48QXB8EivyLuwY'
    os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY
    llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True)

    return llm

if __name__ == '__main__':
    get_answer()
