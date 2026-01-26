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

def embed_chunks(chunks):
    if len(chunks) == 0:
        return None
    return model.encode(chunks)

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def search_faiss(index, query_vec, k):
    distances, indices = index.search(query_vec, k)
    return indices[0], distances[0]

def visualize_3d(vectors, query_vec):
    if vectors is None or query_vec is None:
        return go.Figure()

    all_vecs = np.vstack([vectors, query_vec])
    reduced = PCA(n_components=3).fit_transform(all_vecs)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=reduced[:-1, 0],
        y=reduced[:-1, 1],
        z=reduced[:-1, 2],
        mode="markers",
        marker=dict(size=4, color="skyblue"),
        name="Chunks"
    ))

    fig.add_trace(go.Scatter3d(
        x=[reduced[-1, 0]],
        y=[reduced[-1, 1]],
        z=[reduced[-1, 2]],
        mode="markers+text",
        text=["Query"],
        marker=dict(size=8, color="red"),
        name="Query"
    ))

    fig.update_layout(
        title="📐 RAG Vector Space (3D)",
        height=500
    )

    return fig


TAB_LABELS = {
    0: "Next ➡ Chunking",
    1: "Next ➡ Embeddings",
    2: "Next ➡ FAISS Search",
    3: "Done ✅"
}

def go_next(tab): return min(tab + 1, 3)
def go_prev(tab): return max(tab - 1, 0)

def update_next_label(tab):
    return gr.update(value=TAB_LABELS[tab])

def update_prev_visibility(tab):
    return gr.update(visible=tab > 0)
