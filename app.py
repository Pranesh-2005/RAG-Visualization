import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import faiss
import PyPDF2

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def chunk_text(text, chunk_size):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]