import os
import sys
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Header, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
from pydantic import BaseModel
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.storage.docstore import SimpleDocumentStore
url_domain = "127.0.0.1"

# Initialize logging
log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log.info("Logger initialized successfully")

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Load the OpenAI API key from environment variables
#os.environ["OPENAI_API_KEY"] ='' # input('Enter your OpenAI API key: ')
load_dotenv()  # Load environment variables from .env file
key = os.getenv("OPENAI_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not key:
    raise ValueError("No API key found. Please set the OPENAI_API_KEY environment variable.")
else:
    os.environ["OPENAI_API_KEY"] = key

# classes
class LangModel(BaseModel):
    model: Optional[str] = "openai"
    key: Optional[str] = os.environ["OPENAI_API_KEY"]  # Default to the environment variable

class QueryRequest(BaseModel):
    model: LangModel
    query: str

class Message(BaseModel): # pydantic model for each message
    sender: str
    message: str

class Conversation(BaseModel): # pydantic model for the conversation history
    conversation: List[Message]

# directories
FILE_DIR = "./file_storage"  # save uploads
INDEX_DIR = "./index_storage"  # save indexing
for dir in [FILE_DIR, INDEX_DIR]:
    os.makedirs(dir, exist_ok=True)

index = None  # This will hold the index for the uploaded PDF

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global index  # Allow modification of the global index
    try:
        # Save the uploaded PDF file
        file_path = os.path.join(FILE_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Process the PDF to create an index
        try:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            index = VectorStoreIndex(documents)
            log.info(f"Index created successfully for file: {file.filename}")
            return JSONResponse(content={"message": f"File {file.filename} uploaded and indexed successfully."})
        except Exception as e:
            log.error(f"Error processing PDF file {file.filename}: {str(e)}")
            return JSONResponse(status_code=500, content={"message": f"Failed to process PDF file. Error: {str(e)}"})

    except Exception as e:
        log.error(f"Failed to upload file: {file.filename}, error: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"Failed to upload file: {file.filename}. Error: {str(e)}"})

@app.post("/chat")
async def chat(conversation: Conversation):
    global index  # Access the global index created during file upload
    try:
        if not index:
            raise HTTPException(status_code=400, detail="No PDF has been uploaded and processed yet.")

        # Extract the last user message to generate a response
        last_user_message = conversation.conversation[-1].message
        log.info(f"Received message: {last_user_message}")

        # Use the query engine to query the index
        query_engine = index.as_query_engine()  # Assuming the method exists
        response = query_engine.query(last_user_message)
        log.info(f"Generated response: {response.response}")
        
        return JSONResponse(content={"response": response.response})
    except Exception as e:
        log.error(f"Error querying index: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"Failed to query PDF content. Error: {str(e)}"})

@app.get("/chat")
async def redirect_to_chat_page():
    return RedirectResponse(url="/static/chat.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)