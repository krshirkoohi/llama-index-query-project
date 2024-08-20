import os
import sys
import logging
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    ServiceContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.storage.docstore import SimpleDocumentStore

# Set up logging
log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log.info("Logger initialized successfully")

# Load environment variables
load_dotenv()
default_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Directory to store uploaded files
UPLOAD_DIR = "./data"
INDEX_DIR = "./storage"
for dir in [UPLOAD_DIR, INDEX_DIR]:
    if os.path.exists(dir):
        shutil.rmtree(dir)  
    os.makedirs(dir, exist_ok=True)

# Global variables
index = None

class LangModel(BaseModel):
    model: Optional[str] = "openai"
    key: Optional[str] = default_key  # Default to the environment variable

class Message(BaseModel):
    sender: str
    message: str

class Conversation(BaseModel):
    conversation: List[Message]

@app.on_event("startup")
async def startup_event():
    log.info("Startup complete: Ready to accept requests.")

@app.get("/", response_class=RedirectResponse)
async def redirect_to_main_page():
    return RedirectResponse(url="/static/interactive_pdf_chat.html")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), model: str = Form(...), openai_key: Optional[str] = Form(None)):
    global index
    try:
        # Check if the file already exists
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        if os.path.exists(file_path):
            log.info(f"File {file.filename} already exists.")
            return JSONResponse(status_code=400, content={"message": "File already uploaded!"})

        # Set the selected model
        if model == 'openai':
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            elif default_key:
                os.environ["OPENAI_API_KEY"] = default_key
            else:
                return JSONResponse(status_code=400, content={"message": "OpenAI API key is required. Please enter a valid key."})
            Settings.embed_model = OpenAI()
            log.info(f"OpenAI model set with provided key.")
        else:
            Settings.embed_model = HuggingFaceEmbedding(model_name=model)
            log.info(f"Model set to {model}")

        # Save the uploaded PDF file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        log.info(f"File {file.filename} uploaded successfully.")

        # Process the PDF to create documents
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        # Add metadata to documents and log the content
        for doc in documents:
            doc.metadata["source"] = file.filename
            log.info(f"Document created with text: {doc.text[:100]}... and metadata: {doc.metadata}")

        # Load existing documents from storage
        all_documents = []
        if os.path.exists(INDEX_DIR):
            for doc_file in os.listdir(INDEX_DIR):
                if doc_file.endswith(".txt"):
                    with open(os.path.join(INDEX_DIR, doc_file), 'r') as f:
                        text = f.read()
                        doc_name = os.path.splitext(doc_file)[0]  # remove '.txt' from filename
                        all_documents.append(Document(text=text, metadata={"source": doc_name}))
                        log.info(f"Loaded existing document: {doc_name} with content: {text[:100]}...")

        # Add new documents to the list
        all_documents.extend(documents)

        # Rebuild the index with all documents
        index = VectorStoreIndex.from_documents(all_documents)
        log.info(f"Index rebuilt with documents: {[doc.metadata['source'] for doc in all_documents]}")

        # Save new documents for future use
        for doc in documents:
            doc_name = f"{doc.metadata['source']}.txt"
            with open(os.path.join(INDEX_DIR, doc_name), 'w') as f:
                f.write(doc.text)

        return JSONResponse(content={"message": f"File {file.filename} uploaded and indexed successfully."})

    except Exception as e:
        log.error(f"Failed to upload and index file: {file.filename}. Error: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"Failed to upload and index file: {file.filename}. Error: {str(e)}"})

@app.post("/chat")
async def chat(conversation: Conversation):
    global index
    try:
        if not index:
            raise HTTPException(status_code=400, detail="No PDF has been uploaded and processed yet.")

        # Extract the last user message to generate a response
        last_user_message = conversation.conversation[-1].message
        log.info(f"Received message: {last_user_message}")

        # Use the chat engine to query the index
        chat_engine = index.as_chat_engine()
        response = chat_engine.query(last_user_message)
        log.info(f"Generated response: {response.response}")

        # Initialize a set for sources
        sources = set()

        # Heuristic to determine if the response genuinely uses indexed documents
        relevant_sources_found = False

        for source in response.source_nodes:
            log.info(f"Source node metadata: {source.node.metadata}")
            source_text = source.node.text.strip()

            # Check if the response contains a significant match with the source text
            if source_text in response.response:
                sources.add(source.node.metadata["source"])
                relevant_sources_found = True
            elif any(keyword in response.response for keyword in source_text.split()[:5]):
                sources.add(source.node.metadata["source"])
                relevant_sources_found = True

        # Prepare the response message
        response_message = {"response": response.response}

        if relevant_sources_found:
            # Only include sources if they were actually relevant to the response
            response_message["sources"] = f"Sources: {', '.join(sources)}"
        else:
            log.info("No relevant sources found in response; omitting sources from display.")

        return JSONResponse(content=response_message)

    except Exception as e:
        log.error(f"Error querying index: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"Failed to query PDF content. Error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)