import streamlit as st
from PIL import Image, ImageEnhance
from transformers import pipeline
import json
import time

# ====================== SEHR ELEGANTE CONFIG ======================
st.set_page_config(
    page_title="FischID",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Elegantes, modernes Design (nicht wie typisches Streamlit)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%);
        color: #f1f5f9;
    }
    h1 {
        font-size: 3.6rem;
        background: linear-gradient(90deg, #67e8f9, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.35rem;
        margin-bottom: 2.5rem;
    }
    .container {
        max-width: 800px;
        margin: 0 auto;
    }
    .upload-area {
        border: 2px dashed #64748b;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.03);
    }
    .result {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("FischID")
st.markdown('<p class="subtitle">Intelligente Fisch-Erkennung mit modernster KI</p>', unsafe_allow_html=True)

# ====================== MODELL ======================
@st.cache_resource(show_spinner="Lade KI-Modell...")
def load_model():
    return pipeline(
        "image-classification", 
        model="google/vit-base-patch16-224", 
        top_k=7
    )

classifier = load_model()

# Daten
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# ====================== UPLOAD ======================
st.markdown('<div class="container">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    camera = st.camera_input("📷 Mit Kamera aufnehmen")
with col2:
    uploaded = st.file_uploader("📁 Bild hochladen", type=["jpg", "jpeg", "png"])

image = None
if camera is not None:
    image = Image.open(camera).convert("RGB")
elif uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

if image is not None:
    # Bild leicht optimieren
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.25)
    
    st.image(image, caption="Dein Foto", use_column_width=True)

    # Fortschritt
    progress_bar = st.progress(0)
    status = st.empty()
    start = time.time()

    for i in range(100):
        time.sleep(0.012)
        progress_bar.progress(i+1)
        status.text(f"Analyse läuft... {time.time()-start:.2f} s")

    # Vorhersage
    with st.spinner("KI wertet das Bild aus..."):
        results = classifier(image)

    top = results[0]
    label = top['label'].replace("_", " ").title()
    confidence = top['score'] * 100

    # Mapping
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
        st.markdown('</div>', unsafe_allow_html=True)

        bundesland = st.selectbox("🌍 Bundesland auswählen", 
                                  options=list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        if info:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
            with c2:
                st.metric("Schonzeit", info.get('schonzeit', "Keine"))
    else:
        st.warning(f"Die KI ist sich nur zu {confidence:.1f}% sicher. Bitte versuche ein klareres Foto.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")
st.caption("FischID • Präzise • Modern • Praktisch")
