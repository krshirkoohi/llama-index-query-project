from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

app = FastAPI()

PERSIST_DIR = "./storage"
index = None

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = input('Enter your OpenAI API key: ')

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

class QueryRequest(BaseModel):
    query: str

@app.get("/query")
async def query_index(
    query: str = Query(..., description="The query to be processed")
):
    if not index:
        raise HTTPException(status_code=500, detail="Index not loaded")
    
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    
    return {"response": response}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI index query service. Use POST /query with a JSON payload to query the index."}