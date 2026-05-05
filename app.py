import streamlit as st
from PIL import Image
from transformers import pipeline
import json

st.set_page_config(
    page_title="FischID Pro • KI Fisch-Erkennung",
    page_icon="🐟",
    layout="centered"
)

# Modernes Design
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #0f172a, #1e2937);}
    h1 {font-size: 3.2rem; background: linear-gradient(90deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .result-card {background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; border-radius: 16px; padding: 1.5rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🐟 FischID Pro")
st.markdown("**Professionelle KI-Fisch-Erkennung mit Hugging Face**")

# Modell laden (vortrainiert)
@st.cache_resource(show_spinner="Lade fortschrittliches Fisch-Modell...")
def load_classifier():
    return pipeline("image-classification", 
                    model="jeemsterri/fish_classification", 
                    top_k=5)

classifier = load_classifier()

# Daten laden
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# Foto aufnehmen oder hochladen
st.subheader("📸 Mach ein Foto oder lade eines hoch")

col1, col2 = st.columns(2)
with col1:
    camera_photo = st.camera_input("Kamera")
with col2:
    uploaded_file = st.file_uploader("Datei hochladen", type=["jpg", "jpeg", "png"])

if camera_photo is not None or uploaded_file is not None:
    image = Image.open(camera_photo if camera_photo else uploaded_file).convert("RGB")
    st.image(image, caption="Dein Fisch-Foto", use_column_width=True)

    with st.spinner("🔍 KI analysiert..."):
        results = classifier(image)

    top = results[0]
    label = top['label'].replace("_", " ").title()
    confidence = top['score'] * 100

    # Mapping auf deutsche Namen
    mapping = {
        "Pike": "Hecht", "Zander": "Zander", "Perch": "Flussbarsch", 
        "Carp": "Karpfen", "Trout": "Meerforelle", "Bream": "Brassen",
        "Catfish": "Wels", "Eel": "Aal"
    }
    fish_name = mapping.get(label, label)

    if confidence >= 75:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.success(f"**Erkannte Art:** {fish_name}")
        st.success(f"**Genauigkeit:** {confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        bundesland = st.selectbox("🌍 Bundesland", 
                                  list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        if info:
            c1, c2 = st.columns(2)
            with c1: st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
            with c2: st.metric("Schonzeit", info.get('schonzeit', "Keine"))
    else:
        st.error(f"❌ Nur {confidence:.1f}% sicher. Bitte besseres Foto machen.")

st.caption("FischID Pro • Vortrainiertes Modell • Modernes Design")
