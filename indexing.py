import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load your API key from .env file
load_dotenv()

# Step 1 — LOAD
# This reads all PDFs from your data folder
print("Loading PDFs...")
loader = PyPDFDirectoryLoader("data/")
documents = loader.load()
print(f"Loaded {len(documents)} pages from your PDFs")

# Step 2 — CHUNK
# This breaks the documents into smaller pieces
print("Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # each chunk is 1000 characters
    chunk_overlap=200     # 200 characters overlap between chunks
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3 — EMBED AND STORE
# This converts chunks into vectors and saves them to FAISS
print("Embedding and storing chunks... (this may take a minute)")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 4 — SAVE
# This saves the vector database to your project folder
vector_store.save_local("faiss_index")
print("Done! Vector database saved to faiss_index folder")
print(f"Your knowledge base contains {len(chunks)} chunks ready for search")