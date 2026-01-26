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



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄➡️🧠 RAG Visualizer (PDF + FAISS)")
    gr.Markdown("### See how PDFs turn into vectors and answers")

    current_tab = gr.State(0)
    chunks_state = gr.State([])
    embeddings_state = gr.State(None)
    faiss_index_state = gr.State(None)
    query_vec_state = gr.State(None)

    with gr.Row():
        nav_prev = gr.Button("⬅ Back", visible=False)
        nav_next = gr.Button("Next ➡ Chunking")

    tabs = gr.Tabs(selected=0)

    with tabs:
        # ---------------- TAB 0 ----------------
        with gr.Tab("📄 Upload PDF", id=0):
            pdf_file = gr.File(file_types=[".pdf"])
            pdf_text = gr.Textbox(lines=8, label="Extracted Text")

            def load_pdf(file):
                return read_pdf(file)

            gr.Button("📖 Read PDF").click(
                load_pdf,
                inputs=pdf_file,
                outputs=pdf_text
            )

        # ---------------- TAB 1 ----------------
        with gr.Tab("✂️ Chunking", id=1):
            chunk_size = gr.Slider(50, 200, 100, step=10)
            chunk_table = gr.Dataframe()

            def run_chunking(text, size):
                chunks = chunk_text(text, size)
                df = pd.DataFrame({
                    "Chunk ID": range(len(chunks)),
                    "Text": chunks
                })
                return df, chunks

            gr.Button("✂️ Create Chunks").click(
                run_chunking,
                inputs=[pdf_text, chunk_size],
                outputs=[chunk_table, chunks_state]
            )

        # ---------------- TAB 2 ----------------
        with gr.Tab("🧠 Embeddings + FAISS", id=2):
            embed_info = gr.Markdown()

            def build_embeddings(chunks):
                vecs = embed_chunks(chunks)
                index = build_faiss_index(vecs)
                return f"""
                ### ✅ FAISS Index Ready
                - Chunks: **{len(chunks)}**
                - Vector Dim: **{vecs.shape[1]}**
                """, vecs, index

            gr.Button("🧠 Build FAISS Index").click(
                build_embeddings,
                inputs=chunks_state,
                outputs=[embed_info, embeddings_state, faiss_index_state]
            )

        # ---------------- TAB 3 ----------------
        with gr.Tab("🔍 Retrieval + 3D View", id=3):
            query = gr.Textbox(label="Ask a question")
            k = gr.Slider(1, 10, 3, step=1)
            results = gr.Dataframe()
            plot = gr.Plot()

            def retrieve(query, chunks, vectors, index, k):
                q_vec = model.encode([query])
                idx, dist = search_faiss(index, q_vec, k)

                df = pd.DataFrame({
                    "Chunk ID": idx,
                    "Distance": dist,
                    "Text": [chunks[i][:200] + "..." for i in idx]
                })

                fig = visualize_3d(vectors, q_vec)
                return df, fig

            gr.Button("🔍 Search").click(
                retrieve,
                inputs=[query, chunks_state, embeddings_state, faiss_index_state, k],
                outputs=[results, plot]
            )

    nav_next.click(
    fn=go_next,
    inputs=current_tab,
    outputs=current_tab
).then(
    fn=lambda tab: gr.update(selected=tab),
    inputs=current_tab,
    outputs=tabs
).then(
    fn=update_next_label,
    inputs=current_tab,
    outputs=nav_next
).then(
    fn=update_prev_visibility,
    inputs=current_tab,
    outputs=nav_prev
    )
    
    nav_prev.click(
    fn=go_prev,
    inputs=current_tab,
    outputs=current_tab
).then(
    fn=lambda tab: gr.update(selected=tab),
    inputs=current_tab,
    outputs=tabs
).then(
    fn=update_next_label,
    inputs=current_tab,
    outputs=nav_next
).then(
    fn=update_prev_visibility,
    inputs=current_tab,
    outputs=nav_prev
)