import docx
import re
import openai
import numpy as np
import faiss

def read_word_document(file_path):
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs if para.text.strip()]
    return full_text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'\d+\s+', '', text)  # Remove page numbers or unwanted numbers
    return text.strip()

paragraphs = read_word_document('/home/jabez/Documents/week_7/Precision-RAG/data/ET_Criminal_Code.docx')
cleaned_paragraphs = [clean_text(para) for para in paragraphs]

# print(cleaned_paragraphs)