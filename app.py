import streamlit as st
from PIL import Image
from transformers import pipeline
import json
import time

st.set_page_config(
    page_title="FischID Pro",
    page_icon="🐟",
    layout="centered"
)

# Sehr modernes & fancy Design
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%); color: #e2e8f0;}
    h1 {font-size: 3.6rem; background: linear-gradient(90deg, #67e8f9, #c084fc, #f472b6);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .card {background: rgba(255,255,255,0.08); backdrop-filter: blur(12px); border-radius: 20px; 
           padding: 1.8rem; border: 1px solid rgba(148,163,184,0.15);}
    .result {background: linear-gradient(90deg, #10b981, #34d399); color: white; border-radius: 18px; padding: 1.6rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🐟 FischID Pro")
st.markdown("**Hochwertige KI-Fisch-Erkennung**")

# Modell (stabil & gut für Fische)
@st.cache_resource(show_spinner="Lade modernes Fisch-Erkennungs-Modell...")
def load_model():
    return pipeline("image-classification", 
                    model="google/vit-base-patch16-224", 
                    top_k=6)

classifier = load_model()

# Daten laden
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

st.subheader("📸 Foto aufnehmen oder hochladen")

c1, c2 = st.columns(2)
with c1:
    camera = st.camera_input("Kamera")
with c2:
    uploaded = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if camera or uploaded:
    image = Image.open(camera if camera else uploaded).convert("RGB")
    st.image(image, caption="Dein Foto", use_column_width=True)

    with st.spinner("🔍 KI analysiert den Fisch..."):
        start = time.time()
        results = classifier(image)
        duration = time.time() - start

    top = results[0]
    label = top['label'].replace("_", " ").title()
    confidence = top['score'] * 100

    # Mapping auf deine deutschen Arten
    mapping = {
        "Pike": "Hecht", "Zander": "Zander", "Perch": "Flussbarsch",
        "Carp": "Karpfen", "Trout": "Meerforelle", "Bream": "Brassen",
        "Catfish": "Wels", "Eel": "Aal", "Flatfish": "Scholle", "Roach": "Rotauge"
    }
    fish_name = mapping.get(label, label)

    if confidence >= 75:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.success(f"**Erkannte Art:** {fish_name}")
        st.success(f"**Sicherheit:** {confidence:.1f}%")
        st.caption(f"Analyse-Dauer: {duration:.2f} Sekunden")
        st.markdown('</div>', unsafe_allow_html=True)

        bundesland = st.selectbox("🌍 Aus welchem Bundesland kommst du?", 
                                  list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        if info:
            col1, col2 = st.columns(2)
            with col1: st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
            with col2: st.metric("Schonzeit", info.get('schonzeit', "Keine"))
    else:
        st.warning(f"Nur {confidence:.1f}% sicher. Bitte ein klareres Foto versuchen.")

st.markdown("---")
st.caption("FischID Pro • Google ViT + Hugging Face • Futuristisches Design")
