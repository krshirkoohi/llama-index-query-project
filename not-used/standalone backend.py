# imports


import os, re
from llama_index.llms import openai
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    Settings
)


# setup
DATA_DIR = './data'
INDEX_DIR = './storage'
for dir in [DATA_DIR, INDEX_DIR]:
  if not os.path.exists(dir):
    os.mkdir(dir)


# select model
models = [ # RAG LLMs are preferred
    'BAAI/bge-small-en',
    'BAAI/bge-small-en-v1.5',
    'BAAI/bge-base-en-v1.5',
    'multi-qa-MiniLM-L6-cos-v1',
    'openai'
]
print('List of supported models:')
for m in range(0, len(models)):
   print(f'type {m} for {models[m]}')
try:
    model_choice = input('Please type your model choice: ')
except ValueError:
    print("Must be an integer number!")
model = models[int(model_choice)]

if model == 'openai':
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
else:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model
    )


# read documents
documents = []
reader = SimpleDirectoryReader(
    input_dir=DATA_DIR,
    recursive=True # catch subdirectories
)
reader_data = reader.iter_data()
for docs in reader_data:
   #print((docs))

   for doc in docs:
      documents.append(Document(text=doc.text, metadata={"source": doc.metadata["file_name"]}))
def cleanText(text):
   cleaned_text = re.sub(r'\s+', ' ', text)
   return cleaned_text
cleaned_documents = [
    Document(text=cleanText(doc.text), metadata=doc.metadata) 
    for doc in documents
]
cleaned_documents


# index documents
service_context = ServiceContext.from_defaults(
    embed_model="local:BAAI/bge-small-en-v1.5"
)
try:
    index = VectorStoreIndex.from_documents(cleaned_documents, service_context=service_context)
except NameError:
   index = load_index_from_storage()
index.storage_context.persist()


# main query loop
chat_engine = index.as_chat_engine()
user_input = ""
print("Type 'exit' to quit")
while True:
    user_input = input("\nEnter your query: ")
    if user_input != "exit":
        response = chat_engine.query(user_input)
        print("\n" + response.response)  # This prints the main response text
        
        sources = set()
        for source in response.source_nodes:
            sources.add(source.node.metadata["source"])
        sources = list(sources) # make subscriptable

        if len(sources) > 0:
            print("\nSources: ")
            for s in range(0,len(sources)):
                print(f'[{s+1}]: {sources[s]}')
    else:
        print("Goodbye!")
        break