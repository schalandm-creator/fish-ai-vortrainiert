import streamlit as st
from PIL import Image
import numpy as np
from transformers import pipeline
import json
import time

st.set_page_config(
    page_title="FischID • Next-Gen",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === EXTREM FANCY CSS ===
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #0a0f1c 0%, #1a2338 100%); color: #e0f2fe;}
    h1 {font-size: 3.8rem; background: linear-gradient(90deg, #22d3ee, #a855f7, #ec4899);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin: 0;}
    .subtitle {text-align: center; color: #94a3b8; font-size: 1.4rem; margin-bottom: 2rem;}
    .card {background: rgba(255,255,255,0.06); backdrop-filter: blur(12px); border-radius: 20px; 
           padding: 1.8rem; border: 1px solid rgba(255,255,255,0.1);}
    .result {background: linear-gradient(90deg, #10b981, #34d399); color: white; border-radius: 18px; padding: 1.5rem;}
    .stButton>button {background: linear-gradient(90deg, #6366f1, #a855f7); color: white; 
                      border-radius: 14px; height: 3.2rem; font-size: 1.1rem; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.title("FischID")
st.markdown('<p class="subtitle">Next-Gen KI Fisch-Erkennung • Made for Excellence</p>', unsafe_allow_html=True)

# Modell laden
@st.cache_resource(show_spinner="Lade hochmodernes KI-Modell...")
def load_model():
    return pipeline("image-classification", 
                    model="google/vit-base-patch16-224", 
                    top_k=8)

classifier = load_model()

# Daten
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# Mapping
fish_mapping = {
    "Pike": "Hecht", "Zander": "Zander", "Perch": "Flussbarsch", "Bass": "Flussbarsch",
    "Carp": "Karpfen", "Trout": "Meerforelle", "Bream": "Brassen", "Catfish": "Wels",
    "Eel": "Aal", "Flatfish": "Scholle", "Roach": "Rotauge"
}

st.subheader("📸 Mach ein Foto oder lade eines hoch")

c1, c2 = st.columns([1, 1])
with c1:
    camera = st.camera_input("Kamera", key="cam")
with c2:
    upload = st.file_uploader("Datei hochladen", type=["jpg", "jpeg", "png"])

image = None
if camera:
    image = Image.open(camera).convert("RGB")
elif upload:
    image = Image.open(upload).convert("RGB")

if image:
    st.image(image, use_column_width=True, caption="Dein Foto")

    with st.spinner("🚀 KI analysiert mit modernster Vision-Technologie..."):
        start = time.time()
        results = classifier(image)
        duration = time.time() - start

    top = results[0]
    raw_label = top['label'].replace("_", " ").title()
    confidence = top['score'] * 100

    fish_name = fish_mapping.get(raw_label, raw_label)

    if confidence >= 78:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.success(f"**{fish_name}** erkannt!")
        st.success(f"Sicherheit: **{confidence:.1f}%**")
        st.caption(f"Analyse in {duration:.2f} Sekunden")
        st.markdown('</div>', unsafe_allow_html=True)

        bundesland = st.selectbox("🌍 Bundesland", options=list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        if info:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
            with col2:
                st.metric("Schonzeit", info.get('schonzeit', "Keine"))
    else:
        st.error(f"Die KI ist sich nur zu {confidence:.1f}% sicher. Versuche ein besseres Foto (gutes Licht, ganzer Fisch von der Seite).")

st.markdown("---")
st.caption("FischID • Powered by Google ViT + Hugging Face • Futuristisches Design für Top-Projekte")
