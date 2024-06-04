# Updated imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

import os
import shutil
import fitz  # PyMuPDF
from dotenv import load_dotenv

CHROMA_PATH = "chroma"
DATA_PATH = "/home/jabez/Documents/week_7/Precision-RAG/data"

# Load environment variables from .env file
load_dotenv()

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    documents = []

    # Load Markdown files
    md_loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents.extend(md_loader.load())

    # Load PDF files
    pdf_paths = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    documents.extend(load_pdf_documents(pdf_paths))

    return documents

def load_pdf_documents(pdf_paths: list[str]) -> list[Document]:
    documents = []
    for path in pdf_paths:
        with fitz.open(path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
            documents.append(Document(page_content=text, metadata={"source": path}))
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)
    else:
        print("Not enough chunks to access the 11th one.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
