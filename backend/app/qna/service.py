import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
)
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
class QnAService:

    def get_answer(self, query):
      
        return retrieve_bot_answer(query)
    
def retrieve_bot_answer(query):
       
        retriever=create_retrieval_qa_bot()
        template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        model = load_google_ai()

        chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser())

        return chain.invoke("How can i run the app using docker?")
        
def create_retrieval_qa_bot(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    persist_dir="C:\\Users\\abhil\\DeepLearning\\Repo\\langchaindemo\\backend\\db",
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

        return db.as_retriever()
    
# prompt_template = """Use the following pieces of context to answer the users question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# ALWAYS return a "SOURCES" part in your answer.
# The "SOURCES" part should be a reference to the source of the document from which you got your answer.
# The example of your response should be:

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """


# def set_custom_prompt():
#     """
#     Prompt template for QA retrieval for each vectorstore
#     """
#     prompt = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )
#     return prompt


# def create_retrieval_qa_chain(llm, prompt, db):
#     """
#     Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

#     This function initializes a RetrievalQA object with a specific chain type and configurations,
#     and returns this QA chain. The retriever is set up to return the top 3 results (k=3).

#     Args:
#         llm (any): The language model to be used in the RetrievalQA.
#         prompt (str): The prompt to be used in the chain type.
#         db (any): The database to be used as the retriever.

#     Returns:
#         RetrievalQA: The initialized QA chain.
#     """
#     rag_chain = (
#     {"context": db.as_retriever(search_kwargs={"k": 3}) | "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
#     )
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=db.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt},
#     )
#     return qa_chain

def load_google_ai():
    print(os.getenv('GOOGLE_API_KEY'))
    if "GOOGLE_API_KEY" not in os.environ:
        print("key not found")
    llm = ChatGoogleGenerativeAI(model="gemini-pro")


    return llm 

def load_model(
   
):
    """
    Load a locally downloaded model.

    Parameters:
        model_path (str): The path to the model to be loaded.
        model_type (str): The type of the model.
        max_new_tokens (int): The maximum number of new tokens for the model.
        temperature (float): The temperature parameter for the model.

    Returns:
        CTransformers: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        SomeOtherException: If the model file is corrupt.
    """
    llama = LlamaAPI(os.environ.get("LLAMA_API_TOKEN"))

    # Additional error handling could be added here for corrupt files, etc.

    llm = ChatLlamaAPI(client=llama)

    return llm


# def create_retrieval_qa_bot(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     persist_dir="C:\\Users\\abhil\\DeepLearning\\Repo\\langchaindemo\\backend\\db",
#     device="cpu",
# ):
#     """
#     This function creates a retrieval-based question-answering bot.

#     Parameters:
#         model_name (str): The name of the model to be used for embeddings.
#         persist_dir (str): The directory to persist the database.
#         device (str): The device to run the model on (e.g., 'cpu', 'cuda').

#     Returns:
#         RetrievalQA: The retrieval-based question-answering bot.

#     Raises:
#         FileNotFoundError: If the persist directory does not exist.
#         SomeOtherException: If there is an issue with loading the embeddings or the model.
#     """
    
#     if not os.path.exists(persist_dir):
#         raise FileNotFoundError(f"No directory found at {persist_dir}")

#     try:
#         embeddings = HuggingFaceEmbeddings(
#             model_name=model_name,
#             model_kwargs={"device": device},
#         )
#     except Exception as e:
#         raise Exception(
#             f"Failed to load embeddings with model name {model_name}: {str(e)}"
#         )

#     db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

#     try:
#         llm = load_model()  # Assuming this function exists and works as expected
#     except Exception as e:
#         raise Exception(f"Failed to load model: {str(e)}")

#     qa_prompt = (
#         set_custom_prompt()
#     )  # Assuming this function exists and works as expected

#     try:
#         rag_chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
#         qa = create_retrieval_qa_chain(
#             llm=llm, prompt=qa_prompt, db=db
#         )  # Assuming this function exists and works as expected
#     except Exception as e:
#         raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

#     return qa


# def retrieve_bot_answer(query):
#     """
#     Retrieves the answer to a given query using a QA bot.

#     This function creates an instance of a QA bot, passes the query to it,
#     and returns the bot's response.

#     Args:
#         query (str): The question to be answered by the QA bot.

#     Returns:
#         dict: The QA bot's response, typically a dictionary with response details.
#     """
#     qa_bot_instance = create_retrieval_qa_bot()
#     bot_response = qa_bot_instance({"query": query})
#     return bot_response




