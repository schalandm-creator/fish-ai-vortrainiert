import streamlit as st
from PIL import Image, ImageEnhance
from transformers import pipeline
import json
import time

st.set_page_config(
    page_title="FischID",
    page_icon="🐟",
    layout="centered"
)

# Sehr sauberes, modernes Design
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%);
        color: #e0f2fe;
    }
    h1 {
        font-size: 3.7rem;
        background: linear-gradient(90deg, #67e8f9, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.4rem;
        margin-bottom: 2rem;
    }
    .result {
        background: linear-gradient(90deg, #10b981, #34d399);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("FischID")
st.markdown('<p class="subtitle">Intelligente Fisch-Erkennung</p>', unsafe_allow_html=True)

# ====================== MODELL ======================
@st.cache_resource(show_spinner="Lade KI-Modell...")
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224",
        top_k=6
    )

classifier = load_model()

# fish_data.json
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# ====================== FOTO ======================
st.subheader("📸 Foto aufnehmen oder hochladen")

col1, col2 = st.columns(2)
with col1:
    camera = st.camera_input("Kamera")
with col2:
    uploaded = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if camera or uploaded:
    image = Image.open(camera if camera else uploaded).convert("RGB")
    
    # Bild verbessern
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.25)
    
    st.image(image, caption="Dein Foto", use_column_width=True)

    # Schöner Ladebalken
    progress_bar = st.progress(0)
    status = st.empty()
    start = time.time()

    for i in range(100):
        time.sleep(0.015)
        progress_bar.progress(i + 1)
        status.text(f"Analyse läuft... {time.time() - start:.2f} s")

    # KI-Analyse
    with st.spinner("KI wertet aus..."):
        results = classifier(image)

    top = results[0]
    label = top['label'].replace("_", " ").title()
    confidence = top['score'] * 100

    mapping = {
        "Pike": "Hecht", "Zander": "Zander", "Perch": "Flussbarsch",
        "Carp": "Karpfen", "Trout": "Meerforelle", "Bream": "Brassen",
        "Catfish": "Wels", "Eel": "Aal"
    }
    fish_name = mapping.get(label, label)

    if confidence >= 75:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.success(f"**Erkannte Art:** {fish_name}")
        st.success(f"**Sicherheit:** {confidence:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        bundesland = st.selectbox("🌍 Bundesland", list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        if info:
            c1, c2 = st.columns(2)
            with c1: st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
            with c2: st.metric("Schonzeit", info.get('schonzeit', "Keine"))
    else:
        st.warning(f"Nur {confidence:.1f}% Sicherheit. Bitte neues Foto versuchen.")

st.caption("FischID • Stabil & Modern")
