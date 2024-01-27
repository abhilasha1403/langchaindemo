from sqlalchemy import create_engine, inspect
from pyhive import hive
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import os
engine = create_engine(f'hive://localhost:10000/dvdrental') #hive db connection string
# Create an Inspector
inspector = inspect(engine)
table_names = inspector.get_table_names()
def split_json(json_string):
    # Parse the JSON string into a Python dictionary
    json_dict = json.loads(json_string)

    # Initialize an empty list to store the split dictionaries
    split_dicts = []

    # Iterate over the key-value pairs in the dictionary
    for key, value in json_dict.items():
        # Create a new dictionary with only one key-value pair
        split_dict = {key: value}
        # Append the new dictionary to the list
        split_dicts.append(split_dict)

    return split_dicts

def convert_to_documents(split_dicts):
    documents = []

    for split_dict in split_dicts:
        document = {"page_content": split_dict}
        documents.append(Document(page_content=json.dumps(split_dict)))
    return documents


result_json = {}
for table_name in table_names:
    try:
        # Get table description
        description = inspector.get_table_comment(table_name)
        if description is None:
            description = ""  # Set to an empty string if description is None
    except Exception as e:
        print(f"Error fetching description for table {table_name}: {e}")
        description = ""  # Set to an empty string in case of an error


    # Get table schema
    columns = inspector.get_columns(table_name)
    schema_info = [{"name": column['name'], "type": str(column['type'])} for column in columns]

    # Add information to the JSON structure
    result_json[table_name] = {
        "tablename":table_name,
        "description": description,
        "schema": schema_info
    }

# Print the resulting JSON
print(json.dumps(result_json, indent=2))
result_schema = json.dumps(result_json)
 # Split loaded schema into chunks
result = split_json(result_schema)
documents = convert_to_documents(result)
#print(documents[0].page_content)
print(documents)
    # Initialize HuggingFace embeddings
huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "hivedb")

    # Create and persist a Chroma vector database from the chunked documents
vector_database = Chroma.from_documents(
        documents=documents,
        embedding=huggingface_embeddings,
        persist_directory=DB_DIR,
    )

vector_database.persist()

