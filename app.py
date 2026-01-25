import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import faiss
import PyPDF2

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA