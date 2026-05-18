import streamlit as st
import numpy as np
import time
import tensorflow as tf
import pickle
import re

st.set_page_config(
    page_title="Log Anomaly Detector",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    /* Dark terminal theme */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stTextArea textarea {
        background-color: #161b22 !important;
        color: #7ee787 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 13px !important;
        border: 1px solid #30363d !important;
    }
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem;
    }
    .stButton > button:hover { background-color: #2ea043; }
 
    /* metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; margin: 6px 0 2px; }
    .metric-lbl { font-size: 12px; color: #8b949e; }
 
    /* token highlight */
    .token-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
    .token-chip {
        padding: 4px 10px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 13px;
        font-weight: 500;
    }
 
    /* verdict banner */
    .verdict-normal {
        background: #0f2d1a; border: 1px solid #238636;
        border-radius: 10px; padding: 16px 22px;
        font-size: 1.1rem; font-weight: 600; color: #7ee787;
    }
    .verdict-anomaly {
        background: #2d0f0f; border: 1px solid #f85149;
        border-radius: 10px; padding: 16px 22px;
        font-size: 1.1rem; font-weight: 600; color: #f85149;
    }
    .section-title {
        font-size: 13px; font-weight: 600; color: #8b949e;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 18px 0 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras",compile=False)

@st.cache_resource
def load_vectorizer():
    with open("vectorizer_vocab.pkl","rb") as f:
        vocab_config=pickle.load(f)
        
    vectorizer=tf.keras.layers.TextVectorization(
        max_tokens=10000,
        output_sequence_length=50,
        standardize='lower_and_strip_punctuation',
    )
    vectorizer.set_vocabulary(vocab_config['vocabulary'])
    return vectorizer

@st.cache_data
def load_threshold():
    try:
        with open("anomaly_threshold.txt","rb") as f:
            return float(f.read().strip())
    except Exception as e:
       return 1e-9
   
def templatise(message: str) -> str:
    """
    Cleans raw log messages using Regular Expressions (Regex). 
    It replaces specific data (like IPs or timestamps) with generic tags (<IP>, <LAT>).
    This prevents the model from memorizing specific user IPs and forces it to learn the *structure* of the log.
    """
    msg = str(message)
    msg = re.sub(r'\b\d{1,3}(\.\d{1,3}){3}\b', '<IP>',  msg) # Replaces 192.168.1.1 with <IP>
    msg = re.sub(r'\b[A-Z0-9]{6,}\b',           '<ID>',  msg) # Replaces long hex codes with <ID>
    msg = re.sub(r'\d+ms',                      '<LAT>', msg) # Replaces '45ms' with <LAT>
    msg = re.sub(r'\b\d{5,}\b',                 '<NUM>', msg) # Replaces long numbers with <NUM>
    
    msg = msg.replace(',', ' ').replace('[', ' ').replace(']', ' ')
    return msg.strip()


def score_sequence(model, vectorizer, text):
    processed = templatise(text)

    X = vectorizer([processed])
    X_np = X.numpy()

    probs = model.predict(X, verbose=0)

    vocab = vectorizer.get_vocabulary()

    token_ids = X_np[0]
    token_probs = probs[0]

    true_log_p = np.log(
        token_probs[np.arange(50), token_ids] + 1e-9
    )

    losses = -true_log_p

    token_loss_pairs = []

    for tid, loss in zip(token_ids, losses):
        if tid == 0:
            continue

        tok_str = vocab[tid] if tid < len(vocab) else '[UNK]'

        token_loss_pairs.append(
            (tok_str, float(loss))
        )

    seq_score = float(losses.max())

    return seq_score, token_loss_pairs

def loss_to_color(loss, max_loss):
    ratio = 0 if max_loss == 0 else min(loss / max_loss, 1.0)

    if ratio < 0.3:
        r, g, b = 15, 45, 26
        border = "#238636"
        text = "#7ee787"

    elif ratio < 0.65:
        r, g, b = 45, 35, 10
        border = "#d29922"
        text = "#e3b341"

    else:
        r, g, b = 45, 15, 15
        border = "#f85149"
        text = "#ff7b72"

    return f"background:rgb({r},{g},{b});border:1px solid {border};color:{text}"

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("---")

    st.markdown("**Model**")
    st.code("LogTransformerAE v1.0", language=None)

    st.markdown("**Architecture**")
    st.markdown("- Params: 71,071")
    st.markdown("- Layers: 2 × Transformer encoder")
    st.markdown("- Embedding dim: 64")
    st.markdown("- Heads: 4")
    st.markdown("- Vocab size: 31 tokens")

    st.markdown("---")

    st.markdown("**Scoring method**")
    st.markdown("Max-token cross-entropy loss")

    st.markdown("**Threshold**")

    try:
        thr = load_threshold()
        st.code(f"{thr:.2e}", language=None)
    except Exception:
        st.warning("threshold file not found")

    st.markdown("---")

    st.markdown("**Results on HDFS dataset**")
    st.markdown("- Precision : 1.00")
    st.markdown("- Recall    : 0.53")
    st.markdown("- ROC-AUC   : 0.77")

    st.markdown("---")

    st.markdown("### 📋 Quick examples")

    example_normal = "e5 e22 e5 e5 e11 e9 e11 e9 e11 e9 e26 e26 e26 e2 e2 e2 e23 e23 e23 e21 e21 e21"
    example_anomaly = "e5 e22 e5 e7 e11 e10 e14 e7"

    if st.button("Load normal example"):
        st.session_state['log_input'] = example_normal

    if st.button("Load anomaly example"):
        st.session_state['log_input'] = example_anomaly

st.markdown("# 🔍 Log Anomaly Detector")

st.markdown(
    "Paste a log event sequence below. The model scores each token and flags sequences that deviate from patterns learned during normal system operation."
)

try:
    model = load_model()
    vectorizer = load_vectorizer()
    threshold = load_threshold()
    model_loaded = True

except Exception as e:
    st.error(f"Could not load model: {e}")

    st.info(
        "Make sure `best_model.keras` and `vectorizer_vocab.pkl` are in the same folder as this script."
    )

    model_loaded = False

default_text = st.session_state.get('log_input', '')

log_input = st.text_area(
    "Log sequence input",
    value=default_text,
    height=120,
    placeholder="e.g. e5 e22 e5 e5 e11 e9 e11 e9 e11 e9",
    label_visibility="collapsed",
)

analyze_btn = st.button(
    "Analyze sequence",
    disabled=not model_loaded
)

if analyze_btn and log_input.strip():

    with st.spinner("Scoring sequence ..."):
        time.sleep(0.3)

        seq_score, token_loss_pairs = score_sequence(
            model,
            vectorizer,
            log_input
        )

    is_anomaly = seq_score > threshold

    st.markdown("---")

    if is_anomaly:
        st.markdown(
            f'''
            <div class="verdict-anomaly">
            ⚠️ ANOMALY DETECTED —
            score {seq_score:.6f} exceeds threshold {threshold:.2e}
            </div>
            ''',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'''
            <div class="verdict-normal">
            ✅ NORMAL —
            score {seq_score:.6f} is below threshold {threshold:.2e}
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        colour = "#f85149" if is_anomaly else "#7ee787"

        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-lbl">Anomaly Score</div>
                <div class="metric-val" style="color:{colour}">
                    {seq_score:.6f}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-lbl">Threshold</div>
                <div class="metric-val" style="color:#e3b341">
                    {threshold:.2e}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-lbl">Tokens analyzed</div>
                <div class="metric-val" style="color:#79c0ff">
                    {len(token_loss_pairs)}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col4:
        verdict = "ANOMALY" if is_anomaly else "NORMAL"
        colour = "#f85149" if is_anomaly else "#7ee787"

        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-lbl">Verdict</div>
                <div class="metric-val"
                     style="color:{colour};font-size:1.3rem">
                    {verdict}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown("")

    st.markdown(
        '''
        <div class="section-title">
        Token-level loss attribution
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Each token is coloured by its reconstruction loss.
        <span style='color:#7ee787'>Green</span> = expected token.
        <span style='color:#e3b341'>Amber</span> = suspicious token.
        <span style='color:#f85149'>Red</span> = highly anomalous token.
        """,
        unsafe_allow_html=True,
    )

    if token_loss_pairs:

        max_loss = max(l for _, l in token_loss_pairs)

        chips = ""

        for tok, loss in token_loss_pairs:

            style = loss_to_color(loss, max_loss)
            chips += f'<span class="token-chip" style="{style}" title="loss: {loss:.4f}">{tok}</span>'

        st.markdown(
            f'<div class="token-row">{chips}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '''
            <div class="section-title">
            Most suspicious tokens
            </div>
            ''',
            unsafe_allow_html=True
        )

        sorted_pairs = sorted(
            token_loss_pairs,
            key=lambda x: x[1],
            reverse=True
        )

        top_n = min(5, len(sorted_pairs))

        cols = st.columns(top_n)

        for i, (tok, loss) in enumerate(sorted_pairs[:top_n]):

            with cols[i]:

                ratio = loss / max_loss if max_loss > 0 else 0

                bar = (
                    "█" * int(ratio * 10)
                    + "░" * (10 - int(ratio * 10))
                )

                colour = (
                    "#f85149"
                    if ratio > 0.65
                    else "#e3b341"
                    if ratio > 0.3
                    else "#7ee787"
                )

                st.markdown(
                    f'''
                    <div class="metric-card">
                        <div class="metric-val"
                             style="color:{colour};font-size:1.1rem">
                            {tok}
                        </div>

                        <div style="
                            font-family:monospace;
                            font-size:11px;
                            color:{colour}">
                            {bar}
                        </div>

                        <div class="metric-lbl">
                            loss: {loss:.4f}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

    else:
        st.warning(
            "No tokens found after preprocessing."
        )

elif analyze_btn and not log_input.strip():
    st.warning(
        "Please enter a log sequence before analyzing."
    )


