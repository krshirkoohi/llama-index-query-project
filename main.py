from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from llama_index.llms import openai
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    ServiceContext,
    load_index_from_storage,
    Settings
)
import os
import re
from typing import List
import shutil

app = FastAPI()

# Setup directories
DATA_DIR = './data'
INDEX_DIR = './storage'
#ROOT_DIR = './static'
for dir in [DATA_DIR, INDEX_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

# Global variable to store the index and chat engine
index = None
chat_engine = None

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text)

# web page
app.mount("/static", StaticFiles(directory="static"), name="static")
def clear_directory(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove the directory and all its contents
        os.makedirs(directory)  # Recreate the empty directory
@app.get("/")
async def root():
    # Clear the contents of the data and storage directories
    clear_directory(DATA_DIR)
    clear_directory(INDEX_DIR)
    
    # Redirect to the static HTML page
    return RedirectResponse(url="/static/main.html")

# openai key
@app.post("/set_openai_key")
async def set_openai_key(api_key: dict):
    if 'api_key' not in api_key or not api_key['api_key']:
        raise HTTPException(status_code=400, detail="API key is required.")
    
    os.environ["OPENAI_API_KEY"] = api_key['api_key']
    return {"message": "OpenAI API key set successfully."}

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    global index, chat_engine
    documents = []

    for file in files:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        reader = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True)
        reader_data = reader.iter_data()

        for docs in reader_data:
            for doc in docs:
                cleaned_doc = Document(text=clean_text(doc.text), metadata={"source": doc.metadata["file_name"]})
                documents.append(cleaned_doc)

    # Index the documents
    try:
        service_context = ServiceContext.from_defaults(embed_model="local:BAAI/bge-small-en-v1.5")
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    index.storage_context.persist()
    chat_engine = index.as_chat_engine()

    return {"message": "Files uploaded and indexed successfully."}

from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    if chat_engine is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Please upload files first.")
    
    # Generate the response from the chat engine
    response = chat_engine.chat(request.query)
    
    # Extract the response text and sources
    response_text = response.response  # Assuming response object has 'response' attribute
    sources = response.source_nodes if hasattr(response, 'source_nodes') else []

    # Use a set to remove duplicate sources
    unique_sources = set()
    for source in sources:
        document_name = source.node.metadata.get("source", "Unknown Document")
        unique_sources.add(document_name)
    
    # Format the sources to include in the response
    if unique_sources:
        source_documents = ', '.join(unique_sources)
        full_response = {"response": response_text, "sources": f"{source_documents}"}
    else:
        full_response = {"response": response_text}
    
    return full_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)