import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    ["🔍 Classifier", "📊 Model Performance", "🖼️ Species Gallery"],
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
        results = []  # collect results for CSV

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)

            # Pre-run prediction to collect results
            if model is not None:
                input_arr = preprocess_image(image)
                preds = model.predict(input_arr, verbose=0)[0]
                top_idx = int(np.argmax(preds))
                top_class = CLASS_NAMES[top_idx]
                confidence = float(preds[top_idx])
                info = JELLYFISH_INFO.get(top_class, {})
                results.append({
                    "Filename": uploaded_file.name,
                    "Predicted Species": top_class.replace('_', ' ').title(),
                    "Confidence (%)": f"{confidence*100:.1f}",
                    "Status": "LOW" if confidence < 0.60 else "MODERATE" if confidence < 0.80 else "HIGH",
                    "Scientific Name": info.get('scientific', ''),
                    "Habitat": info.get('habitat', ''),
                    "Size": info.get('size', ''),
                    "Sting Danger": info.get('danger', '').replace('✅','').replace('⚠️','').replace('🔴','').strip(),
                    "Note": "Verify - may not be a supported species" if confidence < 0.60 else "Consider using a clearer image" if confidence < 0.80 else "OK",
                    "_image": image,
                    "_confidence_raw": confidence,
                })

        # ── Download buttons at TOP ──
        if results:
            df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in results])

            # ── Generate HTML report ──
            def generate_html_report(results):
                import base64

                def img_to_base64(img):
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG")
                    return base64.b64encode(buf.getvalue()).decode()

                rows_html = ""
                for r in results:
                    conf = r["_confidence_raw"]
                    if conf < 0.60:
                        row_bg = "background:#2d0a0a; border-left: 4px solid #e74c3c;"
                        badge = f'<span style="background:#e74c3c;color:white;padding:3px 10px;border-radius:99px;font-size:0.75rem;">⚠️ Low {conf*100:.1f}%</span>'
                    elif conf < 0.80:
                        row_bg = "background:#2d1a00; border-left: 4px solid #e67e22;"
                        badge = f'<span style="background:#e67e22;color:white;padding:3px 10px;border-radius:99px;font-size:0.75rem;">🔶 Moderate {conf*100:.1f}%</span>'
                    else:
                        row_bg = "background:#0a1628; border-left: 4px solid #7fffd4;"
                        badge = f'<span style="background:#1a6b4a;color:#7fffd4;padding:3px 10px;border-radius:99px;font-size:0.75rem;">✅ High {conf*100:.1f}%</span>'

                    img_b64 = img_to_base64(r["_image"])
                    note = r.get("Note", "")
                    note_html = f'<div style="color:#e67e22;font-size:0.75rem;margin-top:0.3rem;">{note}</div>' if note else ""

                    rows_html += f"""
                    <tr style="{row_bg}">
                        <td style="padding:12px;"><img src="data:image/jpeg;base64,{img_b64}"
                            style="width:90px;height:90px;object-fit:cover;border-radius:10px;"/></td>
                        <td style="padding:12px;color:#a8c8e8;font-size:0.85rem;">{r['Filename']}</td>
                        <td style="padding:12px;">
                            <div style="color:#7fffd4;font-weight:700;font-size:0.95rem;">{r['Predicted Species']}</div>
                            <div style="color:#7ecfea;font-size:0.78rem;font-style:italic;">{r['Scientific Name']}</div>
                        </td>
                        <td style="padding:12px;">{badge}{note_html}</td>
                        <td style="padding:12px;color:#a8c8e8;font-size:0.82rem;">{r['Habitat']}</td>
                        <td style="padding:12px;color:#a8c8e8;font-size:0.82rem;">{r['Sting Danger']}</td>
                    </tr>"""

                total = len(results)
                low = sum(1 for r in results if r["_confidence_raw"] < 0.60)
                moderate = sum(1 for r in results if 0.60 <= r["_confidence_raw"] < 0.80)
                high = sum(1 for r in results if r["_confidence_raw"] >= 0.80)

                return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Jellyfish Classification Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400&display=swap');
        body {{ background: linear-gradient(160deg,#020b18,#041e3a,#062d55);
               min-height:100vh; font-family:'DM Sans',sans-serif; color:white; margin:0; padding:2rem; }}
        h1 {{ font-family:'Syne',sans-serif; font-size:2rem; font-weight:800;
              background:linear-gradient(135deg,#7fffd4,#00bfff,#a78bfa);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0; }}
        .sub {{ color:#7ecfea; letter-spacing:3px; text-transform:uppercase; font-size:0.78rem; margin-bottom:2rem; }}
        .summary {{ display:flex; gap:1rem; margin-bottom:2rem; }}
        .badge {{ padding:0.6rem 1.2rem; border-radius:12px; font-size:0.85rem; font-weight:600; }}
        table {{ width:100%; border-collapse:separate; border-spacing:0 6px; }}
        th {{ background:rgba(0,191,255,0.08); color:#00bfff; font-size:0.7rem;
              letter-spacing:2px; text-transform:uppercase; padding:10px 12px; text-align:left; }}
        td {{ vertical-align:middle; }}
        .footer {{ text-align:center; color:#2a6fa8; font-size:0.75rem; margin-top:2rem; }}
    </style>
</head>
<body>
    <h1>🪼 Jellyfish Classification Report</h1>
    <div class="sub">MobileNetV2 · Deep Learning · 6 Species</div>
    <div class="summary">
        <div class="badge" style="background:rgba(127,255,212,0.1);color:#7fffd4;border:1px solid #7fffd4;">
            ✅ High Confidence: {high}
        </div>
        <div class="badge" style="background:rgba(230,126,34,0.1);color:#e67e22;border:1px solid #e67e22;">
            🔶 Moderate: {moderate}
        </div>
        <div class="badge" style="background:rgba(231,76,60,0.1);color:#e74c3c;border:1px solid #e74c3c;">
            ⚠️ Low Confidence: {low}
        </div>
        <div class="badge" style="background:rgba(0,191,255,0.1);color:#00bfff;border:1px solid #00bfff;">
            📊 Total: {total}
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>Image</th><th>Filename</th><th>Species</th>
                <th>Confidence</th><th>Habitat</th><th>Sting Danger</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    <div class="footer">Generated by Jellyfish Classifier · MobileNetV2 · Streamlit 🪼</div>
</body>
</html>"""

            html_report = generate_html_report(results)

            col_html, col_csv, _ = st.columns([1, 1, 1])
            with col_html:
                st.download_button(
                    label=f"📊 Download HTML Report ({len(results)} image{'s' if len(results) > 1 else ''})",
                    data=html_report,
                    file_name="jellyfish_report.html",
                    mime="text/html",
                    use_container_width=True
                )
            with col_csv:
                st.download_button(
                    label=f"📥 Download CSV ({len(results)} image{'s' if len(results) > 1 else ''})",
                    data=df.to_csv(index=False),
                    file_name="jellyfish_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown('<hr class="ocean-divider">', unsafe_allow_html=True)

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            info = JELLYFISH_INFO.get(CLASS_NAMES[int(np.argmax(model.predict(preprocess_image(image), verbose=0)[0]))], {})
            preds = model.predict(preprocess_image(image), verbose=0)[0]
            top_idx = int(np.argmax(preds))
            top_class = CLASS_NAMES[top_idx]
            confidence = float(preds[top_idx])
            info = JELLYFISH_INFO.get(top_class, {})
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
                    st.markdown('<div class="info-label">🔍 Prediction</div>', unsafe_allow_html=True)

                    # ── Confidence threshold warning ──
                    if confidence < 0.60:
                        st.markdown(f"""
                        <div style="background:rgba(231,76,60,0.1); border:1px solid rgba(231,76,60,0.4);
                             border-radius:16px; padding:1.2rem 1.5rem; margin-top:1rem;">
                            <div style="color:#e74c3c; font-family:'Syne',sans-serif;
                                 font-size:1rem; font-weight:700; margin-bottom:0.3rem;">
                                ⚠️ Low Confidence
                            </div>
                            <div style="color:#a8c8e8; font-size:0.85rem;">
                                Model is only <b style="color:#e74c3c">{confidence*100:.1f}%</b> confident.
                                This may not be one of the 6 supported species,
                                or the image quality may be too low.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence < 0.80:
                        st.markdown(f"""
                        <div style="background:rgba(230,126,34,0.1); border:1px solid rgba(230,126,34,0.4);
                             border-radius:16px; padding:1.2rem 1.5rem; margin-top:1rem;">
                            <div style="color:#e67e22; font-family:'Syne',sans-serif;
                                 font-size:1rem; font-weight:700; margin-bottom:0.3rem;">
                                🔶 Moderate Confidence
                            </div>
                            <div style="color:#a8c8e8; font-size:0.85rem;">
                                Model is <b style="color:#e67e22">{confidence*100:.1f}%</b> confident.
                                Result is likely correct but verify with a clearer image.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

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
    st.markdown('<div class="hero-sub">Confusion Matrix · Classification Report · Training History</div>', unsafe_allow_html=True)

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
            def make_bar(val):
                color = "#7fffd4" if val >= 0.90 else "#00bfff" if val >= 0.80 else "#e67e22"
                return f"""<div style="text-align:center">
                    <div style="height:3px; border-radius:99px; margin-bottom:3px;
                        background:linear-gradient(90deg, {color} {val*100:.0f}%,
                        rgba(255,255,255,0.05) {val*100:.0f}%)"></div>
                    <span style="color:{color}; font-size:0.82rem; font-weight:600">{val*100:.0f}%</span>
                </div>"""

            st.markdown(f"""
            <div class="info-card" style="display:grid;
                 grid-template-columns:1.4fr 0.8fr 0.8fr 0.8fr 0.7fr;
                 gap:0.3rem; align-items:center; padding:0.7rem 0.8rem; margin-top:0.4rem;">
                <span style="font-size:0.85rem">{m['emoji']} {m['name']}</span>
                {make_bar(m['precision'])}
                {make_bar(m['recall'])}
                {make_bar(m['f1'])}
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

    # ═══════════════════════════════════════════════
    # TRAINING HISTORY PLOT
    # ═══════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="info-label">📈 Training History</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#2a6fa8; font-size:0.78rem; letter-spacing:1px; margin-bottom:1rem;">
        Phase 1 = frozen base (epochs 1–20) · Phase 2 = fine-tuning (epochs 21–26)
    </p>
    """, unsafe_allow_html=True)

    # Phase 1 — frozen base (~20 epochs)
    p1_train_acc  = [0.50, 0.72, 0.83, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.94,
                     0.95, 0.95, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.97]
    p1_val_acc    = [0.69, 0.80, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95, 0.95, 0.96,
                     0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97]
    p1_train_loss = [1.45, 0.85, 0.55, 0.40, 0.32, 0.27, 0.23, 0.20, 0.18, 0.16,
                     0.14, 0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.09, 0.08]
    p1_val_loss   = [0.92, 0.63, 0.42, 0.32, 0.26, 0.22, 0.19, 0.16, 0.15, 0.14,
                     0.13, 0.12, 0.12, 0.11, 0.11, 0.11, 0.10, 0.10, 0.10, 0.10]

    # Phase 2 — fine-tuning (~6 epochs)
    p2_train_acc  = [0.83, 0.85, 0.88, 0.89, 0.90, 0.91]
    p2_val_acc    = [0.97, 0.95, 0.93, 0.92, 0.91, 0.89]
    p2_train_loss = [0.55, 0.44, 0.35, 0.30, 0.29, 0.26]
    p2_val_loss   = [0.12, 0.16, 0.21, 0.29, 0.37, 0.43]

    all_train_acc  = p1_train_acc  + p2_train_acc
    all_val_acc    = p1_val_acc    + p2_val_acc
    all_train_loss = p1_train_loss + p2_train_loss
    all_val_loss   = p1_val_loss   + p2_val_loss
    epochs         = list(range(1, len(all_train_acc) + 1))
    phase2_start   = len(p1_train_acc) + 1

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig2.patch.set_facecolor("#041e3a")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#041e3a")
        ax.tick_params(colors="#7ecfea", labelsize=8)
        ax.spines[:].set_color("#062d55")
        ax.grid(color="#062d55", linewidth=0.5)
        ax.axvline(x=phase2_start, color="#a78bfa", linewidth=1.2,
                   linestyle="--", alpha=0.7)

    ax1.plot(epochs, all_train_acc, color="#00bfff", linewidth=2,
             marker="o", markersize=3, label="Train Accuracy")
    ax1.plot(epochs, all_val_acc,   color="#7fffd4", linewidth=2,
             marker="o", markersize=3, label="Val Accuracy")
    ax1.set_title("Accuracy", color="#7ecfea", fontsize=11, pad=10)
    ax1.set_xlabel("Epoch", color="#7ecfea", fontsize=9)
    ax1.set_ylabel("Accuracy", color="#7ecfea", fontsize=9)
    ax1.legend(facecolor="#041e3a", labelcolor="#7ecfea", fontsize=8)
    ax1.set_ylim(0.4, 1.05)
    ax1.text(10, 0.45, "Phase 1: Frozen", color="#a78bfa", fontsize=8, ha="center", alpha=0.8)
    ax1.text(phase2_start + 2, 0.45, "Phase 2: Fine-tune", color="#a78bfa", fontsize=8, ha="center", alpha=0.8)

    ax2.plot(epochs, all_train_loss, color="#00bfff", linewidth=2,
             marker="o", markersize=3, label="Train Loss")
    ax2.plot(epochs, all_val_loss,   color="#7fffd4", linewidth=2,
             marker="o", markersize=3, label="Val Loss")
    ax2.set_title("Loss", color="#7ecfea", fontsize=11, pad=10)
    ax2.set_xlabel("Epoch", color="#7ecfea", fontsize=9)
    ax2.set_ylabel("Loss", color="#7ecfea", fontsize=9)
    ax2.legend(facecolor="#041e3a", labelcolor="#7ecfea", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ═══════════════════════════════════════════════
# PAGE 3 — SPECIES GALLERY
# ═══════════════════════════════════════════════
elif page == "🖼️ Species Gallery":

    st.markdown('<div class="hero-title">Species Gallery</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">6 Jellyfish Species · Training Classes</div>', unsafe_allow_html=True)

    SPECIES = [
        {
            "key": "Moon_jellyfish",
            "name": "Moon Jellyfish",
            "emoji": "🌙",
            "scientific": "Aurelia aurita",
            "danger": "Harmless ✅",
            "habitat": "Worldwide oceans",
        },
        {
            "key": "barrel_jellyfish",
            "name": "Barrel Jellyfish",
            "emoji": "🪼",
            "scientific": "Rhizostoma pulmo",
            "danger": "Low ✅",
            "habitat": "Atlantic, Mediterranean",
        },
        {
            "key": "blue_jellyfish",
            "name": "Blue Jellyfish",
            "emoji": "💙",
            "scientific": "Cyanea lamarckii",
            "danger": "Mild sting ⚠️",
            "habitat": "North Atlantic, North Sea",
        },
        {
            "key": "compass_jellyfish",
            "name": "Compass Jellyfish",
            "emoji": "🧭",
            "scientific": "Chrysaora hysoscella",
            "danger": "Moderate sting ⚠️",
            "habitat": "Eastern Atlantic, Mediterranean",
        },
        {
            "key": "lions_mane_jellyfish",
            "name": "Lion's Mane Jellyfish",
            "emoji": "🦁",
            "scientific": "Cyanea capillata",
            "danger": "Strong sting 🔴",
            "habitat": "Arctic, North Atlantic",
        },
        {
            "key": "mauve_stinger_jellyfish",
            "name": "Mauve Stinger",
            "emoji": "💜",
            "scientific": "Pelagia noctiluca",
            "danger": "Painful sting 🔴",
            "habitat": "Mediterranean, Atlantic",
        },
    ]

    # 3 cards per row
    for i in range(0, len(SPECIES), 3):
        cols = st.columns(3, gap="medium")
        for col, species in zip(cols, SPECIES[i:i+3]):
            with col:
                img_path = f"samples/{species['key']}.jpg"
                try:
                    st.image(img_path, use_container_width=True)
                except:
                    st.markdown(f"""
                    <div style="height:180px; background:rgba(255,255,255,0.03);
                         border-radius:12px; display:flex; align-items:center;
                         justify-content:center; font-size:3rem;">
                         {species['emoji']}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-card" style="margin-top:0.5rem;">
                    <div style="font-family:'Syne',sans-serif; font-size:1rem;
                         font-weight:700; color:#7fffd4; margin-bottom:0.3rem;">
                        {species['emoji']} {species['name']}
                    </div>
                    <div style="color:#7ecfea; font-size:0.78rem; font-style:italic;
                         margin-bottom:0.5rem;">{species['scientific']}</div>
                    <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                        <span style="background:rgba(0,191,255,0.1);
                             border:1px solid rgba(0,191,255,0.2); border-radius:99px;
                             padding:2px 10px; font-size:0.72rem; color:#00bfff;">
                            🌊 {species['habitat']}
                        </span>
                        <span style="background:rgba(255,255,255,0.05);
                             border:1px solid rgba(255,255,255,0.1); border-radius:99px;
                             padding:2px 10px; font-size:0.72rem; color:#a8c8e8;">
                            {species['danger']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('<div class="footer">Built with TensorFlow · MobileNetV2 · Streamlit 🪼</div>', unsafe_allow_html=True)
