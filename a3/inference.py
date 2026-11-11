import time
import streamlit as st
from config import ROOT
from study import load_data, model_status
from vendor.a2_predictor import Predictor


@st.cache_data
def data():
    return load_data()


@st.cache_data
def checkpoint_status():
    return model_status()


@st.cache_resource
def model():
    if not checkpoint_status()['matches']:
        raise ValueError('Checkpoint SHA-256 does not match A2')
    return Predictor(ROOT/'models/emotion-ferplus-8.onnx')


def predict(image_id):
    """Live A2 inference API: top1, top2, probabilities, elapsed milliseconds."""
    rows = {r['image_id']: r for r in data()[0]}
    start = time.perf_counter()
    top1, top2, probs = model().predict(ROOT/rows[image_id]['image_path'])
    return top1, top2, probs, (time.perf_counter()-start)*1000


def advice(row):
    try:
        first, second, probs, elapsed = predict(row['image_id'])
        source = 'live'
    except Exception:
        first, second, probs = row['model_prediction'], row['model_second_choice'], row['probs']
        elapsed, source = None, 'cached_fallback'
    return dict(ai_top1=first, ai_top2=second, ai_probs=probs, ai_inference_ms=elapsed, ai_source=source)
