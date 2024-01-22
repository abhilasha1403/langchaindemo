import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
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

    retriever = create_retrieval_qa_bot()
    template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)

    # model = load_google_ai()
    model = load_llama_model()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser())

    return chain.invoke(query)


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


def load_google_ai():
    print(os.getenv('GOOGLE_API_KEY'))
    if "GOOGLE_API_KEY" not in os.environ:
        print("key not found")
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    return llm


def load_llama_model():

    llama = LlamaAPI(os.environ.get("LLAMA_API_TOKEN"))

    llm = ChatLlamaAPI(client=llama)

    return llm


