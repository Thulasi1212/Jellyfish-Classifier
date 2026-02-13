import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Jellyfish Classifier 🪼",
    page_icon="🪼",
    layout="wide"
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

# ── Sidebar Navigation ───────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0;">
    <p style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
      background: linear-gradient(135deg, #7fffd4, #00bfff);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
      🪼 Jellyfish Classifier
    </p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Classifier", "📊 Model Performance"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div style="margin-top:2rem; color:#2a6fa8; font-size:0.75rem; text-align:center; letter-spacing:1px;">
    Built with TensorFlow · MobileNetV2 · Streamlit
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="hero-title">Jellyfish Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Deep Learning · MobileNetV2 · 6 Species</div>', unsafe_allow_html=True)

model = load_model()

# Class names — must match your training order!
CLASS_NAMES = [
    "Moon_jellyfish",
    "barrel_jellyfish",
    "blue_jellyfish",
    "compass_jellyfish",
    "lions_mane_jellyfish",
    "mauve_stinger_jellyfish",
]

# ═══════════════════════════════════════════════
# PAGE 1 — CLASSIFIER
# ═══════════════════════════════════════════════
if page == "🔍 Classifier":

    # ── Upload bar at top (compact) ──
    uploaded_files = st.file_uploader(
        "Upload jellyfish images",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported: JPG, JPEG, PNG, WEBP",
        label_visibility="collapsed",
        accept_multiple_files=True
    )

    st.markdown("""
    <div style="text-align:center; margin-top:-0.5rem; margin-bottom:0.3rem;">
        <p style="color:#7fffd4; font-size:0.82rem; letter-spacing:1px;">
            💡 <b>Tip:</b> Hold <kbd style="background:#0a2a4a; border:1px solid #2a6fa8; border-radius:4px; padding:1px 6px; font-size:0.78rem;">Ctrl</kbd> (Windows) or <kbd style="background:#0a2a4a; border:1px solid #2a6fa8; border-radius:4px; padding:1px 6px; font-size:0.78rem;">⌘ Cmd</kbd> (Mac) to select multiple images at once!
        </p>
        <p style="color:#e67e22; font-size:0.82rem; letter-spacing:1px; margin-top:-0.3rem;">
            ⚠️ Only classifies these 6 species: Barrel · Blue · Compass · Lion's Mane · Mauve Stinger · Moon Jellyfish
        </p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)

            st.markdown(f"""
            <div style="margin-top:1.5rem; margin-bottom:0.3rem;">
                <span class="species-badge">📁 {uploaded_file.name}</span>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1.2, 1.2, 1.6], gap="medium")

            with col1:
                st.markdown('<div class="info-label">📷 Uploaded Image</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)

            with col2:
                if model is not None:
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        input_arr = preprocess_image(image)
                        preds = model.predict(input_arr, verbose=0)[0]
                        top_idx = int(np.argmax(preds))
                        top_class = CLASS_NAMES[top_idx]
                        confidence = float(preds[top_idx])
                        info = JELLYFISH_INFO.get(top_class, {})

                    st.markdown('<div class="info-label">🔍 Prediction</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="species-badge">{info.get('emoji','🪼')} Identified</div>
                        <div class="result-name">{top_class.replace('_', ' ').title()}</div>
                        <div class="result-confidence">Confidence: {confidence*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="info-label">Top Predictions</div>', unsafe_allow_html=True)
                    top3_idx = np.argsort(preds)[::-1][:3]
                    for i in top3_idx:
                        name = CLASS_NAMES[i].replace('_', ' ').title()
                        prob = float(preds[i])
                        st.progress(prob, text=f"{name}  {prob*100:.1f}%")

            with col3:
                if model is not None and info:
                    st.markdown('<div class="info-label">🔬 Species Info</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.4rem">
                        <div class="info-card">
                            <div class="info-label">Scientific Name</div>
                            <p><i>{info['scientific']}</i></p>
                        </div>
                        <div class="info-card">
                            <div class="info-label">Habitat</div>
                            <p>{info['habitat']}</p>
                        </div>
                        <div class="info-card">
                            <div class="info-label">Size</div>
                            <p>{info['size']}</p>
                        </div>
                        <div class="info-card">
                            <div class="info-label">Sting Danger</div>
                            <p>{info['danger']}</p>
                        </div>
                    </div>
                    <div class="info-card" style="margin-top:0.6rem">
                        <div class="info-label">🌊 Did You Know?</div>
                        <p>{info['fun_fact']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<hr class="ocean-divider">', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 0;">
            <p style="font-size:3.5rem; margin:0;">🪼</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700;
                      background: linear-gradient(135deg, #7fffd4, #00bfff);
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                      margin: 0.5rem 0 0.3rem;">
                Upload one or more jellyfish images to identify them
            </p>
            <p style="color:#2a6fa8; font-size:0.82rem; letter-spacing:2px; text-transform:uppercase; margin:0;">
                barrel · blue · compass · lion's mane · mauve stinger · moon
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Confusion Matrix · Classification Report · Test Accuracy</div>', unsafe_allow_html=True)

    # ── Real values from your test set ──
    CM = np.array([
        [6, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0],
        [1, 0, 5, 0, 1, 0],
        [0, 0, 0, 6, 1, 0],
        [0, 0, 0, 0, 8, 0],
        [0, 0, 1, 0, 0, 6],
    ])

    DISPLAY_NAMES = ["Moon", "Barrel", "Blue", "Compass", "Lion's Mane", "Mauve Stinger"]
    EMOJIS = ["🌙", "🪼", "💙", "🧭", "🦁", "💜"]

    METRICS = [
        {"name": "Moon Jellyfish",      "emoji": "🌙", "precision": 0.86, "recall": 1.00, "f1": 0.92, "support": 6},
        {"name": "Barrel Jellyfish",    "emoji": "🪼", "precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 5},
        {"name": "Blue Jellyfish",      "emoji": "💙", "precision": 0.83, "recall": 0.71, "f1": 0.77, "support": 7},
        {"name": "Compass Jellyfish",   "emoji": "🧭", "precision": 1.00, "recall": 0.86, "f1": 0.92, "support": 7},
        {"name": "Lion's Mane",         "emoji": "🦁", "precision": 0.80, "recall": 1.00, "f1": 0.89, "support": 8},
        {"name": "Mauve Stinger",       "emoji": "💜", "precision": 1.00, "recall": 0.86, "f1": 0.92, "support": 7},
    ]

    # ── Summary stats ──
    total = CM.sum()
    correct = CM.diagonal().sum()
    acc = correct / total

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    for col, label, value in zip(
        [col_s1, col_s2, col_s3, col_s4],
        ["Test Accuracy", "Total Images", "Correct", "Classes"],
        [f"{acc*100:.1f}%", int(total), int(correct), 6]
    ):
        with col:
            st.markdown(f"""
            <div class="result-card" style="text-align:center; padding:1rem;">
                <div class="result-name" style="font-size:1.8rem;">{value}</div>
                <div class="result-confidence">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ── Confusion Matrix Plot ──
    with col_left:
        st.markdown('<div class="info-label">🔢 Confusion Matrix</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#041e3a")
        ax.set_facecolor("#041e3a")

        sns.heatmap(
            CM, annot=True, fmt="d", ax=ax,
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.5, linecolor="#062d55",
            xticklabels=DISPLAY_NAMES,
            yticklabels=DISPLAY_NAMES,
            cbar_kws={"shrink": 0.8}
        )

        ax.set_xlabel("Predicted", color="#7ecfea", fontsize=10, labelpad=10)
        ax.set_ylabel("Actual", color="#7ecfea", fontsize=10, labelpad=10)
        ax.tick_params(colors="#7ecfea", labelsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)

        # Color the colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color="#7ecfea")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#7ecfea")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Classification Report ──
    with col_right:
        st.markdown('<div class="info-label">📊 Classification Report</div>', unsafe_allow_html=True)

        # Header
        st.markdown("""
        <div style="display:grid; grid-template-columns:1.4fr 0.8fr 0.8fr 0.8fr 0.7fr;
             gap:0.3rem; padding:0.4rem 0.8rem;
             font-size:0.65rem; color:#2a6fa8; letter-spacing:1px; text-transform:uppercase;">
            <span>Species</span>
            <span style="text-align:center">Precision</span>
            <span style="text-align:center">Recall</span>
            <span style="text-align:center">F1</span>
            <span style="text-align:center">Support</span>
        </div>
        """, unsafe_allow_html=True)

        for m in METRICS:
            def bar(val):
                color = "#7fffd4" if val >= 0.90 else "#00bfff" if val >= 0.80 else "#e67e22"
                return f"""
                <div style="text-align:center">
                    <div style="height:3px; border-radius:99px; margin-bottom:3px;
                        background:linear-gradient(90deg, {color} {val*100}%,
                        rgba(255,255,255,0.05) {val*100}%)"></div>
                    <span style="color:{color}; font-size:0.82rem; font-weight:600">{val*100:.0f}%</span>
                </div>"""

            st.markdown(f"""
            <div class="info-card" style="display:grid;
                 grid-template-columns:1.4fr 0.8fr 0.8fr 0.8fr 0.7fr;
                 gap:0.3rem; align-items:center; padding:0.7rem 0.8rem; margin-top:0.4rem;">
                <span style="font-size:0.85rem">{m['emoji']} {m['name']}</span>
                {bar(m['precision'])}
                {bar(m['recall'])}
                {bar(m['f1'])}
                <span style="text-align:center; color:#7ecfea; font-size:0.82rem">{m['support']}</span>
            </div>
            """, unsafe_allow_html=True)

        # Macro avg
        st.markdown("""
        <div style="margin-top:0.8rem; padding:0.6rem 0.8rem;
             background:rgba(0,191,255,0.05); border:1px solid rgba(0,191,255,0.2);
             border-radius:10px; display:grid;
             grid-template-columns:1.4fr 0.8fr 0.8fr 0.8fr 0.7fr; gap:0.3rem;">
            <span style="color:#7ecfea; font-size:0.82rem; font-weight:600">Macro Avg</span>
            <span style="text-align:center; color:#7fffd4; font-size:0.82rem; font-weight:600">92%</span>
            <span style="text-align:center; color:#7fffd4; font-size:0.82rem; font-weight:600">90%</span>
            <span style="text-align:center; color:#7fffd4; font-size:0.82rem; font-weight:600">90%</span>
            <span style="text-align:center; color:#7ecfea; font-size:0.82rem">40</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">Built with TensorFlow · MobileNetV2 · Streamlit 🪼</div>', unsafe_allow_html=True)
