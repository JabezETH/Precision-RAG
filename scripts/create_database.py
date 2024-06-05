from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings

import os
import shutil
import fitz  # PyMuPDF
from dotenv import load_dotenv

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Insert the path to your data_processing module
sys.path.insert(1, '/home/jabez/Documents/week_7/Precision-RAG/notebooks/data_processing')
import data_processing 

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "/home/jabez/Documents/week_7/Precision-RAG/data"

# Use the cleaned paragraphs from your data_processing module
cleaned_paragraphs = data_processing.cleaned_paragraphs

# Convert paragraphs to Document objects with metadata
documents = [
    Document(page_content=para, metadata={"source": f"document_{i}", "article": f"Article_{idx}"})
    for i, doc in enumerate(cleaned_paragraphs) for idx, para in enumerate(doc.split("\n\n"))
]

def split_paragraphs(documents: list[Document], chunk_size=1000, overlap=200):
    chunks = []
    current_chunk = []
    current_metadata = {}
    current_chunk_size = 0

    for doc in documents:
        words = doc.page_content.split()
        if current_chunk_size + len(words) > chunk_size:
            chunk_content = ' '.join(current_chunk)
            chunks.append(Document(page_content=chunk_content, metadata=current_metadata))
            current_chunk = words[-overlap:]  # Start new chunk with overlap
            current_metadata = doc.metadata  # Inherit metadata
            current_chunk_size = len(current_chunk)
        else:
            if not current_chunk:
                current_metadata = doc.metadata
            current_chunk.extend(words)
            current_chunk_size += len(words)

    if current_chunk:
        chunk_content = ' '.join(current_chunk)
        chunks.append(Document(page_content=chunk_content, metadata=current_metadata))

    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    logging.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    try:
        chunks = split_paragraphs(documents)
        save_to_chroma(chunks)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
