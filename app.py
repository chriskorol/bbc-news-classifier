import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="BBC News Klassifikator",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Anthropic-Inspired Theme ───────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,500;8..60,600;8..60,700&family=Inter:wght@300;400;500;600&display=swap');

    /* ── Root / Global ── */
    :root {
        --bg-primary: #E8E6DC;
        --bg-light: #F5F4ED;
        --bg-card: #FAF9F0;
        --text-primary: #141413;
        --text-secondary: #30302E;
        --text-muted: #87867F;
        --accent-terracotta: #D97757;
        --accent-amber: #EDA100;
        --border-subtle: rgba(20, 20, 19, 0.12);
        --border-medium: rgba(194, 192, 182, 0.4);
        --radius: 4px;
    }

    /* ── Main background ── */
    .stApp, [data-testid="stAppViewContainer"], .main .block-container {
        background-color: var(--bg-primary) !important;
    }
    header[data-testid="stHeader"] {
        background-color: var(--bg-primary) !important;
    }

    /* ── Typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }
    h1, h2, h3 {
        font-family: 'Source Serif 4', 'Georgia', serif !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
    }
    p, li, span, label, .stMarkdown {
        color: var(--text-secondary) !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

    /* ── Container width ── */
    .block-container {
        max-width: 720px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    /* ── Text area ── */
    .stTextArea textarea {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 16px !important;
        transition: border-color 200ms ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-terracotta) !important;
        box-shadow: 0 0 0 1px var(--accent-terracotta) !important;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
        font-style: italic;
    }
    .stTextArea label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--text-muted) !important;
    }

    /* ── Primary button (Anthropic terracotta) ── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: var(--accent-terracotta) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 2rem !important;
        letter-spacing: 0.02em !important;
        transition: all 200ms cubic-bezier(.77, 0, .175, 1) !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #C4613E !important;
        transform: translateY(-1px);
    }

    /* ── Secondary button ── */
    .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        font-size: 0.85rem !important;
    }

    /* ── Result card ── */
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        padding: 28px 32px;
        margin: 24px 0;
    }
    .result-category {
        font-family: 'Source Serif 4', serif;
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.02em;
    }
    .result-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    /* ── Confidence bars ── */
    .confidence-row {
        display: flex;
        align-items: center;
        margin: 8px 0;
        gap: 12px;
    }
    .confidence-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 400;
        color: var(--text-secondary);
        min-width: 120px;
        text-transform: capitalize;
    }
    .confidence-bar-bg {
        flex: 1;
        height: 6px;
        background: var(--border-subtle);
        border-radius: 3px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 600ms cubic-bezier(0.16, 1, 0.3, 1);
    }
    .confidence-value {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--text-primary);
        min-width: 48px;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    /* ── Divider ── */
    .anthropic-divider {
        border: none;
        border-top: 1px solid var(--border-subtle);
        margin: 32px 0;
    }

    /* ── Expander (examples) ── */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
        background-color: transparent !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius) !important;
        background-color: var(--bg-card) !important;
    }

    /* ── Success/warning boxes ── */
    .stAlert {
        border-radius: var(--radius) !important;
    }

    /* ── Subtle logo mark ── */
    .anthropic-mark {
        font-size: 0.85rem;
        color: var(--text-muted);
        text-align: center;
        margin-top: 48px;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.1em;
    }
    .anthropic-mark a {
        color: var(--text-muted);
        text-decoration: none;
        transition: color 200ms ease;
    }
    .anthropic-mark a:hover {
        color: var(--accent-terracotta);
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ─────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 8px;">
    <span style="font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 500;
                 text-transform: uppercase; letter-spacing: 0.1em; color: #87867F;">
        Machine Learning · Text Classification
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("# BBC News Klassifikator")

st.markdown("""
<p style="font-family: 'Source Serif 4', serif; font-size: 1.15rem; color: #30302E;
          line-height: 1.6; margin-bottom: 32px; font-weight: 300;">
    Geben Sie einen englischen Nachrichtenartikel ein.<br>
    Das Modell erkennt automatisch die Kategorie — mit über 99% Genauigkeit.
</p>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    categories = joblib.load(os.path.join(MODEL_DIR, "categories.pkl"))
    return model, tfidf_vectorizer, categories

try:
    model, tfidf, categories = load_model()
except FileNotFoundError:
    st.error("Modell nicht gefunden. Bitte zuerst das Notebook ausführen.")
    st.stop()


# ─── Category Config ────────────────────────────────────────────────────────

CATEGORY_CONFIG = {
    "business":      {"icon": "◆", "color": "#D97757"},
    "entertainment": {"icon": "◆", "color": "#EDA100"},
    "politics":      {"icon": "◆", "color": "#7A8B6F"},
    "sport":         {"icon": "◆", "color": "#5B8FA8"},
    "tech":          {"icon": "◆", "color": "#8B7BA8"},
}


# ─── Input ──────────────────────────────────────────────────────────────────

text_input = st.text_area(
    "Nachrichtentext",
    height=180,
    placeholder="Paste a BBC news article here...",
)

col1, col2 = st.columns([2, 5])
with col1:
    classify_btn = st.button("Klassifizieren", type="primary")


# ─── Prediction ─────────────────────────────────────────────────────────────

if classify_btn:
    if not text_input.strip():
        st.warning("Bitte geben Sie einen Text ein.")
    else:
        text_tfidf = tfidf.transform([text_input])
        prediction = model.predict(text_tfidf)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_tfidf)[0]
        else:
            scores = model.decision_function(text_tfidf)[0]
            exp_scores = np.exp(scores - scores.max())
            probabilities = exp_scores / exp_scores.sum()

        config = CATEGORY_CONFIG.get(prediction, {"icon": "◆", "color": "#D97757"})

        # Result card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Vorhergesagte Kategorie</div>
            <div class="result-category">
                <span style="color: {config['color']};">✦</span> {prediction.capitalize()}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars
        st.markdown("""
        <div style="margin-bottom: 8px;">
            <span style="font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 500;
                         text-transform: uppercase; letter-spacing: 0.08em; color: #87867F;">
                Konfidenz
            </span>
        </div>
        """, unsafe_allow_html=True)

        sorted_indices = probabilities.argsort()[::-1]
        for idx in sorted_indices:
            cat = categories[idx] if isinstance(categories, list) else model.classes_[idx]
            prob = probabilities[idx]
            cfg = CATEGORY_CONFIG.get(cat, {"icon": "◆", "color": "#D97757"})

            bar_color = cfg["color"] if prob > 0.1 else "rgba(20, 20, 19, 0.15)"

            st.markdown(f"""
            <div class="confidence-row">
                <span class="confidence-label">{cat.capitalize()}</span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill"
                         style="width: {prob * 100:.1f}%; background-color: {bar_color};"></div>
                </div>
                <span class="confidence-value">{prob:.1%}</span>
            </div>
            """, unsafe_allow_html=True)


# ─── Divider ────────────────────────────────────────────────────────────────

st.markdown('<hr class="anthropic-divider">', unsafe_allow_html=True)


# ─── Example Texts ──────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 16px;">
    <span style="font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 500;
                 text-transform: uppercase; letter-spacing: 0.1em; color: #87867F;">
        Beispieltexte zum Testen
    </span>
</div>
""", unsafe_allow_html=True)

examples = {
    "Business": "The stock market surged today as investors reacted positively to the latest quarterly earnings reports from major tech companies. Wall Street analysts predict continued growth in the financial sector.",
    "Sport": "Manchester United secured a dramatic 3-2 victory over Liverpool at Old Trafford. The winning goal came in the 89th minute from a spectacular free kick by the young midfielder.",
    "Tech": "Apple announced its latest iPhone model featuring an advanced AI chip and improved camera system. The new device supports satellite connectivity and enhanced augmented reality capabilities.",
    "Politics": "The Prime Minister addressed Parliament today regarding new legislation on immigration reform. Opposition leaders criticized the proposed changes as insufficient to address current challenges.",
    "Entertainment": "The new blockbuster film broke box office records in its opening weekend, earning over $200 million worldwide. Critics praised the director's innovative storytelling approach.",
}

for cat, example in examples.items():
    cfg = CATEGORY_CONFIG.get(cat.lower(), {"icon": "◆", "color": "#D97757"})
    with st.expander(f"✦ {cat}"):
        st.code(example, language=None)


# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="anthropic-mark">
    <span>THWS · Datenbanken-Projekt · 2026</span>
</div>
""", unsafe_allow_html=True)
