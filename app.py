import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import DetrImageProcessor, DetrForSegmentation
import numpy as np
import json
import time

st.set_page_config(page_title="FischID • DETR Pro", page_icon="🐟", layout="centered")

# === KRASS MODERNES DESIGN ===
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #020617 0%, #1e2937 100%); color: #e0f2fe;}
    h1 {font-size: 4rem; background: linear-gradient(90deg, #22d3ee, #c026d3, #f472b6);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .card {background: rgba(255,255,255,0.07); backdrop-filter: blur(16px); border-radius: 24px; 
           padding: 2rem; border: 1px solid rgba(148,163,184,0.2);}
    .result-box {background: linear-gradient(90deg, #10b981, #34d399); color: white; border-radius: 20px; padding: 1.5rem;}
</style>
""", unsafe_allow_html=True)

st.title("FischID • DETR")
st.markdown("**DETR Fish Segmentation • Detection + Masken • Next Level KI**")

# Modell laden
@st.cache_resource(show_spinner="Lade DETR Fish Segmentation Modell...")
def load_detr_model():
    processor = DetrImageProcessor.from_pretrained("FriedParrot/fish-segmentation-simple")
    model = DetrForSegmentation.from_pretrained("FriedParrot/fish-segmentation-simple")
    return processor, model

processor, model = load_detr_model()

# Daten
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# Kamera / Upload
st.subheader("📸 Foto machen oder hochladen")
col1, col2 = st.columns(2)
with col1:
    camera = st.camera_input("Kamera")
with col2:
    uploaded = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if camera or uploaded:
    image = Image.open(camera if camera else uploaded).convert("RGB")
    st.image(image, caption="Eingabebild", use_column_width=True)

    with st.spinner("🧠 DETR analysiert Fisch (Detection + Segmentation)..."):
        start = time.time()
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-processing
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_segmentation(outputs, target_sizes=target_sizes)[0]

        duration = time.time() - start

    # Einfache Anzeige der besten Erkennung
    if len(results["labels"]) > 0:
        scores = results["scores"].detach().numpy()
        best_idx = scores.argmax()
        score = scores[best_idx] * 100
        label_id = results["labels"][best_idx].item()
        # Label Mapping (DETR verwendet oft COCO-ähnliche Labels)
        label_map = {0: "Fisch", 1: "Hecht", 2: "Zander", 3: "Karpfen"}  # erweiterbar
        fish_name = label_map.get(label_id, "Unbekannter Fisch")

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"**{fish_name}** erkannt")
        st.success(f"**Konfidenz:** {score:.1f}% • Zeit: {duration:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)

        # Bundesland & Regeln
        bundesland = st.selectbox("🌍 Bundesland", list(fish_data["bundeslaender"].keys()))
        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        
        if info:
            c1, c2 = st.columns(2)
            with c1: st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
            with c2: st.metric("Schonzeit", info.get('schonzeit', "Keine"))
    else:
        st.warning("Kein Fisch erkannt. Versuche ein klareres Foto.")

st.markdown("---")
st.caption("FischID DETR Pro • Fish Segmentation + Detection • Futuristisches Schulprojekt-Design")
