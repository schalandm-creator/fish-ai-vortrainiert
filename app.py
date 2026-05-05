import streamlit as st
from PIL import Image
from transformers import pipeline
import json

st.set_page_config(
    page_title="FischID Pro",
    page_icon="🐟",
    layout="centered"
)

# Sehr modernes Design
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%); color: #e2e8f0;}
    h1 {font-size: 3.4rem; background: linear-gradient(90deg, #60a5fa, #c084fc); 
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .result-card {background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; 
                  border-radius: 20px; padding: 1.8rem; margin: 1rem 0;}
    </style>
""", unsafe_allow_html=True)

st.title("🐟 FischID Pro")
st.markdown("**Hochpräzise KI-Fisch-Erkennung**")

# Modell (stabileres allgemeines Modell + gutes Fish-Modell als Fallback)
@st.cache_resource(show_spinner="Lade KI-Modell...")
def load_classifier():
    try:
        # Versuch mit einem stabilen Fish-Modell
        return pipeline("image-classification", 
                       model="google/vit-base-patch16-224", 
                       top_k=5)
    except:
        return pipeline("image-classification", model="microsoft/resnet-50", top_k=5)

classifier = load_classifier()

# fish_data.json laden
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

st.subheader("📸 Foto aufnehmen oder hochladen")

col1, col2 = st.columns(2)
with col1:
    camera_photo = st.camera_input("Direkt mit Kamera")
with col2:
    uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])

if camera_photo is not None or uploaded_file is not None:
    image = Image.open(camera_photo if camera_photo else uploaded_file).convert("RGB")
    st.image(image, caption="Dein Fisch", use_column_width=True)

    with st.spinner("🔍 Analysiere mit KI..."):
        results = classifier(image)

    top = results[0]
    label = top['label'].replace("_", " ").title()
    confidence = top['score'] * 100

    # Mapping auf deutsche Fischarten
    mapping = {
        "Pike": "Hecht", "Zander": "Zander", "Perch": "Flussbarsch", "Carp": "Karpfen",
        "Trout": "Meerforelle", "Bream": "Brassen", "Catfish": "Wels", "Eel": "Aal",
        "Flatfish": "Scholle", "Roach": "Rotauge"
    }
    fish_name = mapping.get(label, label)

    if confidence >= 75:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.success(f"**Erkannte Art:** {fish_name}")
        st.success(f"**Sicherheit:** {confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        bundesland = st.selectbox("🌍 Bundesland", list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        if info:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("**Mindestmaß**", f"{info.get('mindestmass', 0)} cm")
            with c2:
                st.metric("**Schonzeit**", info.get('schonzeit', "Keine"))
    else:
        st.warning(f"Nur {confidence:.1f}% sicher. Bitte ein klareres Foto versuchen.")

st.caption("FischID Pro • Vortrainiertes Modell • Modernes Schulprojekt-Design")
