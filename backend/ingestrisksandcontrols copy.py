from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import os
from langchain_community.document_loaders import (
     DirectoryLoader,
    TextLoader
)
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI 
def load_llama_model():

    os.environ["LLAMA_API_TOKEN"]="LL-dlB0BwX20TLvjPR9u3cwyyB3zdhqj1DrCXzHmIIjn5EDCVOkvkxiL8xuDg2KBZTN"
    llama = LlamaAPI(os.environ.get("LLAMA_API_TOKEN"))
    llm = ChatLlamaAPI(client=llama)
    return llm

def load_google_ai():
    os.environ["GOOGLE_API_KEY"]='AIzaSyBT43TG_ZGK2gGXEH1rq48QXB8EivyLuwY'
    if "GOOGLE_API_KEY" not in os.environ:
        print("key not found")
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    return llm

llm = load_llama_model()

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please create a summary by keeping all the important points
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of all important points. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=400,
)

map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)


def createSummary():
    directoryList = "C:\\Users\\abhil\DeepLearning\\Repo\\langchaindemo\\backend\\process\\"
    for directory_name in os.listdir(directoryList):
        directory_path = os.path.join(directoryList, directory_name)
        pattern = "./*.txt"  # Adjust pattern for your file types

    # Create the DirectoryLoader
        loader = DirectoryLoader(directory_path, glob=pattern)

        document = loader.load()
        split_docs = text_splitter.split_documents(document)
        print(map_reduce_chain.run(split_docs))

createSummary()