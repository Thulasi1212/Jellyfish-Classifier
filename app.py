import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Jellyfish Classifier 🪼",
    page_icon="🪼",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Deep ocean background */
.stApp {
    background: linear-gradient(160deg, #020b18 0%, #041e3a 40%, #062d55 100%);
    min-height: 100vh;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(135deg, #7fffd4, #00bfff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.hero-sub {
    text-align: center;
    color: #7ecfea;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

/* Upload area */
.upload-box {
    border: 2px dashed #2a6fa8;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(255,255,255,0.03);
    transition: all 0.3s ease;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(0,191,255,0.08), rgba(127,255,212,0.05));
    border: 1px solid rgba(0,191,255,0.25);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    backdrop-filter: blur(10px);
}

.result-name {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #7fffd4;
    margin: 0;
}

.result-confidence {
    font-size: 1rem;
    color: #7ecfea;
    margin-top: 0.2rem;
    font-weight: 300;
    letter-spacing: 1px;
}

/* Info card */
.info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}

.info-card p {
    color: #a8c8e8;
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 0;
}

.info-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00bfff;
    margin-bottom: 0.4rem;
}

/* Progress bar override */
.stProgress > div > div {
    background: linear-gradient(90deg, #00bfff, #7fffd4) !important;
    border-radius: 99px !important;
}

/* Divider */
.ocean-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a6fa8, transparent);
    margin: 2rem 0;
}

/* Emoji badge */
.species-badge {
    display: inline-block;
    background: rgba(0,191,255,0.12);
    border: 1px solid rgba(0,191,255,0.3);
    border-radius: 99px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00bfff;
    margin-bottom: 0.6rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #2a6fa8;
    font-size: 0.78rem;
    margin-top: 3rem;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ── Jellyfish info database ──────────────────────────────────
JELLYFISH_INFO = {
    "Moon_jellyfish": {
        "emoji": "🌙",
        "scientific": "Aurelia aurita",
        "habitat": "Worldwide oceans",
        "size": "Up to 40cm bell diameter",
        "fun_fact": "The most common jellyfish worldwide. The four pink/purple rings visible through their translucent bell are their reproductive organs.",
        "danger": "Harmless ✅"
    },
    "barrel_jellyfish": {
        "emoji": "🪼",
        "scientific": "Rhizostoma pulmo",
        "habitat": "Atlantic Ocean, Mediterranean Sea",
        "size": "Up to 90cm bell diameter",
        "fun_fact": "One of the largest jellyfish in UK waters, they are harmless to humans and are actually a food source for leatherback sea turtles.",
        "danger": "Low ✅"
    },
    "blue_jellyfish": {
        "emoji": "💙",
        "scientific": "Cyanea lamarckii",
        "habitat": "North Atlantic, North Sea",
        "size": "Up to 30cm bell diameter",
        "fun_fact": "Their vivid blue or yellow colour fades as they age. They are most commonly spotted in summer months near UK coasts.",
        "danger": "Mild sting ⚠️"
    },
    "compass_jellyfish": {
        "emoji": "🧭",
        "scientific": "Chrysaora hysoscella",
        "habitat": "Eastern Atlantic, Mediterranean",
        "size": "Up to 30cm bell diameter",
        "fun_fact": "Named after the brown compass-like markings on their bell. They are an active predator, catching small fish and crustaceans.",
        "danger": "Moderate sting ⚠️"
    },
    "lions_mane_jellyfish": {
        "emoji": "🦁",
        "scientific": "Cyanea capillata",
        "habitat": "Arctic, North Atlantic, North Pacific",
        "size": "Up to 2m bell — world's largest jellyfish!",
        "fun_fact": "The world's largest known jellyfish species. Their tentacles can extend over 30 meters — longer than a blue whale!",
        "danger": "Strong sting 🔴"
    },
    "mauve_stinger_jellyfish": {
        "emoji": "💜",
        "scientific": "Pelagia noctiluca",
        "habitat": "Mediterranean, Atlantic, Indo-Pacific",
        "size": "Up to 10cm bell diameter",
        "fun_fact": "They are bioluminescent — they glow blue-green at night when disturbed. Despite being small, their sting is surprisingly painful.",
        "danger": "Painful sting 🔴"
    }
}

# ── Model loading ────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("best_jellyfish_model.keras")
        return model
    except Exception as e:
        st.error(f"⚠️ Model file not found. Please upload `best_jellyfish_model.keras` to your repo. Error: {e}")
        return None

# ── Preprocessing ────────────────────────────────────────────
def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── UI ───────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Jellyfish Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Deep Learning · MobileNetV2 · 6 Species</div>', unsafe_allow_html=True)

model = load_model()

# Class names — must match your training order!
CLASS_NAMES = [
    "Moon_jellyfish",           # capital M — matches train_ds.class_names
    "barrel_jellyfish",
    "blue_jellyfish",
    "compass_jellyfish",
    "lions_mane_jellyfish",
    "mauve_stinger_jellyfish",  # includes _jellyfish suffix
]

uploaded_file = st.file_uploader(
    "Upload a jellyfish image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported: JPG, JPEG, PNG, WEBP"
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        if model is not None:
            with st.spinner("Analyzing..."):
                input_arr = preprocess_image(image)
                preds = model.predict(input_arr, verbose=0)[0]
                top_idx = int(np.argmax(preds))
                top_class = CLASS_NAMES[top_idx]
                confidence = float(preds[top_idx])
                info = JELLYFISH_INFO.get(top_class, {})

            # Result card
            st.markdown(f"""
            <div class="result-card">
                <div class="species-badge">{info.get('emoji','🪼')} Identified</div>
                <div class="result-name">{top_class.replace('_', ' ').title()}</div>
                <div class="result-confidence">Confidence: {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(confidence)

            # Top 3 predictions
            st.markdown("**Top predictions:**")
            top3_idx = np.argsort(preds)[::-1][:3]
            for i in top3_idx:
                name = CLASS_NAMES[i].replace('_', ' ').title()
                prob = preds[i]
                bar_col, label_col = st.columns([3, 1])
                with bar_col:
                    st.progress(float(prob), text=name)
                with label_col:
                    st.markdown(f"<p style='color:#7fffd4;font-size:0.85rem;margin-top:6px'>{prob*100:.1f}%</p>", unsafe_allow_html=True)

    # Species info
    st.markdown('<hr class="ocean-divider">', unsafe_allow_html=True)

    if info:
        st.markdown("### 🔬 Species Info")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Scientific Name</div>
                <p><i>{info['scientific']}</i></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Habitat</div>
                <p>{info['habitat']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Size</div>
                <p>{info['size']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Sting Danger</div>
                <p>{info['danger']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card" style="margin-top:1rem">
            <div class="info-label">🌊 Did You Know?</div>
            <p>{info['fun_fact']}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-box">
        <p style="color:#2a6fa8;font-size:2rem;margin:0">🪼</p>
        <p style="color:#7ecfea;margin:0.5rem 0 0">Drop a jellyfish image above to identify it</p>
        <p style="color:#2a6fa8;font-size:0.8rem;margin-top:0.3rem">Supports: barrel · blue · compass · lion's mane · mauve stinger · moon</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">Built with TensorFlow · MobileNetV2 · Streamlit 🪼</div>', unsafe_allow_html=True)
