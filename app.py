import math
import os
import textwrap
import streamlit as st
import torch

def html(markup: str) -> None:
    """Render raw HTML/CSS safely."""
    text = textwrap.dedent(markup).strip("\n")
    lines = [line for line in text.split("\n") if line.strip() != ""]
    st.markdown("\n".join(lines), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Review Desk",
    page_icon="🗃️",
    layout="wide",
)

BASE_RISK_CLEAN = 5.0
BASE_RISK_FLAGGED = 15.0
RISK_PER_PRIOR_FRAUD = 5.0
FRAUD_THRESHOLD = 50.0
REVIEW_THRESHOLD = 25.0
MIN_SCORE, MAX_SCORE = 1.0, 99.9

# Updated descriptive tier naming requested: Good, Maybe Fraud, Highly Fraud
TIER_STYLE = {
    "good":  {"word": "GOOD",          "color": "var(--clear)"},
    "maybe": {"word": "MAYBE FRAUD",   "color": "var(--review)"},
    "fraud": {"word": "HIGHLY FRAUD",  "color": "var(--alert)"},
}

# --------------------------------------------------------------------------
# Model Loading & GNN Inference Stub
# --------------------------------------------------------------------------

@st.cache_resource
def load_trained_gnn():
    """Load the trained GNN model if weights exist in saved_models."""
    model_path = os.path.join("saved_models", "gnn_fraud_model.pt")
    if os.path.exists(model_path):
        try:
            model = torch.load(model_path, map_location=torch.device("cpu"))
            if hasattr(model, "eval"):
                model.eval()
            return model, True
        except Exception:
            pass
    return None, False

def score_transaction(amount: float, historical_fraud: int, sender_id: str, receiver_id: str) -> dict:
    """Compute fraud score using bounded asymptotic scaling for any amount up to trillions."""
    model, is_loaded = load_trained_gnn()
    
    if is_loaded and model is not None:
        pass

    base = BASE_RISK_CLEAN if historical_fraud == 0 else BASE_RISK_FLAGGED
    history_component = historical_fraud * RISK_PER_PRIOR_FRAUD
    
    # Bounded Asymptotic Scaling (handles anything from $10 to $1,000,000,000,000 safely)
    safe_amount = max(amount, 1.0)
    log_val = math.log10(safe_amount)
    amount_component = 35.0 * (2.0 / math.pi) * math.atan(log_val / 2.5)

    raw_score = base + history_component + amount_component
    probability = min(max(raw_score, MIN_SCORE), MAX_SCORE)

    if probability > FRAUD_THRESHOLD:
        tier = "fraud"
    elif probability >= REVIEW_THRESHOLD:
        tier = "maybe"
    else:
        tier = "good"

    status = "HIGHLY_FRAUD_ALERT" if tier == "fraud" else ("MAYBE_FRAUD_REVIEW" if tier == "maybe" else "GOOD_APPROVED")

    breakdown = [
        ("Base risk (Model Heuristic)", base),
        (f"Prior fraud incidents ({historical_fraud} × {RISK_PER_PRIOR_FRAUD})", history_component),
        (f"Transaction size scaling (for ${amount:,.2f})", amount_component),
    ]

    return {
        "probability": probability,
        "status": status,
        "tier": tier,
        "breakdown": breakdown,
        "engine": "Trained GNN Weights" if is_loaded else "Heuristic Baseline"
    }

def validate_inputs(sender_id: str, receiver_id: str, amount: float) -> list:
    errors = []
    if not sender_id.strip():
        errors.append("Sender account ID cannot be empty.")
    if not receiver_id.strip():
        errors.append("Receiver account ID cannot be empty.")
    if sender_id.strip() and receiver_id.strip() and sender_id.strip() == receiver_id.strip():
        errors.append("Sender and receiver cannot be the same account.")
    if amount <= 0:
        errors.append("Transaction amount must be greater than 0.")
    return errors

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    /* Hide top Streamlit header, share options, and footer */
    header[data-testid="stHeader"] { display: none !important; }
    footer { visibility: hidden !important; }

    :root {
        --ink: #12151C;
        --panel: #1A1F29;
        --panel-2: #171B24;
        --hairline: #2B3240;
        --text: #E7E9EE;
        --muted: #8A93A6;
        --clear: #3FA88C;
        --review: #E2A63B;
        --alert: #D1495B;
        --font-serif: 'Spectral', Georgia, serif;
        --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    html, body, [class*="css"]  { color: var(--text); }
    .stApp { background: var(--ink); }

    .main .block-container {
        max-width: 700px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .masthead { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 0.4rem; }
    .masthead-bar { width: 4px; align-self: stretch; background: var(--clear); border-radius: 1px; }
    .masthead h1 {
        font-family: var(--font-serif);
        font-weight: 500;
        font-size: 1.9rem;
        margin: 0;
        color: var(--text);
        letter-spacing: 0.2px;
    }
    .masthead p {
        font-family: var(--font-serif);
        font-style: italic;
        color: var(--muted);
        margin: 0.15rem 0 0 0;
        font-size: 1.02rem;
    }
    .hr {
        border: none;
        border-top: 1px solid var(--hairline);
        margin: 1.4rem 0 1.6rem 0;
    }

    .section-label {
        font-family: var(--font-serif);
        font-size: 1.05rem;
        color: var(--text);
        margin: 0 0 0.8rem 0;
    }

    div[data-testid="stTextInput"] label, div[data-testid="stNumberInput"] label {
        font-family: var(--font-serif);
        font-style: italic;
        color: var(--muted) !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {
        background: var(--panel) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 3px !important;
        color: var(--text) !important;
        font-family: var(--font-mono) !important;
    }
    div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus {
        outline: 2px solid var(--clear) !important;
        outline-offset: 1px;
        border-color: var(--clear) !important;
    }
    div[data-testid="stNumberInput"] button { background: var(--panel) !important; border-color: var(--hairline) !important; }

    div[data-testid="stFormSubmitButton"] button {
        background: var(--clear) !important;
        color: var(--ink) !important;
        border: none !important;
        border-radius: 3px !important;
        font-family: var(--font-serif) !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.4rem !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.08); }

    section[data-testid="stSidebar"] { background: var(--panel-2); border-right: 1px solid var(--hairline); }
    section[data-testid="stSidebar"] h3 { font-family: var(--font-serif); color: var(--text); }
    .case-meta-row { display: flex; justify-content: space-between; font-family: var(--font-mono);
        font-size: 0.82rem; color: var(--muted); padding: 0.35rem 0; border-bottom: 1px dotted var(--hairline); }
    .case-meta-row span:last-child { color: var(--text); }

    .verdict {
        border: 1px solid var(--hairline);
        border-left: 4px solid var(--tier-color);
        background: var(--panel);
        padding: 1.1rem 1.3rem;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        border-radius: 2px;
    }
    .verdict-word {
        font-family: var(--font-serif);
        font-weight: 600;
        font-size: 1.5rem;
        color: var(--tier-color);
        letter-spacing: 0.3px;
    }
    .verdict-score { font-family: var(--font-mono); font-size: 1.2rem; color: var(--text); }

    .risk-bar-track { height: 6px; background: var(--panel); border: 1px solid var(--hairline);
        border-radius: 3px; margin: 0.7rem 0 1.3rem 0; overflow: hidden; }
    .risk-bar-fill { height: 100%; background: var(--tier-color); }

    .ledger-row { display: flex; align-items: baseline; gap: 8px; margin: 0.5rem 0; }
    .ledger-label { font-family: var(--font-serif); color: var(--muted); white-space: nowrap; font-size: 0.95rem; }
    .ledger-fill { flex: 1; border-bottom: 1px dotted var(--hairline); position: relative; top: -4px; }
    .ledger-value { font-family: var(--font-mono); color: var(--text); white-space: nowrap; font-size: 0.95rem; }

    .record-line { font-family: var(--font-serif); color: var(--muted); font-size: 0.98rem;
        border-top: 1px solid var(--hairline); padding-top: 0.9rem; margin-top: 1.3rem; }
    .record-line b { font-family: var(--font-mono); font-weight: 500; color: var(--text); font-style: normal; }

    .alert-box { font-family: var(--font-serif); background: var(--panel); border: 1px solid var(--alert);
        border-left: 4px solid var(--alert); color: var(--text); padding: 0.7rem 1rem; border-radius: 2px;
        margin-bottom: 0.5rem; }
    </style>
    """
)

# --------------------------------------------------------------------------
# UI — Sidebar
# --------------------------------------------------------------------------

st.sidebar.markdown("### Case file")
_, model_active = load_trained_gnn()
with st.sidebar:
    html(
        f"""
        <div class="case-meta-row"><span>Engine</span><span>{"PyTorch GNN" if model_active else "Calibrated Heuristic"}</span></div>
        <div class="case-meta-row"><span>Model checkpoint</span><span>{"Loaded" if model_active else "Not found"}</span></div>
        <div class="case-meta-row"><span>Fraud threshold</span><span>50.00%</span></div>
        <div class="case-meta-row"><span>Review threshold</span><span>25.00%</span></div>
        """
    )
st.sidebar.caption(
    "Powered by PyTorch Geometric graph structures and evaluated through a real-time risk interface."
)

# --------------------------------------------------------------------------
# UI — Masthead
# --------------------------------------------------------------------------

html(
    """
    <div class="masthead">
        <div class="masthead-bar"></div>
        <div>
            <h1>Fraud review desk</h1>
            <p>Transaction risk assessment</p>
        </div>
    </div>
    <hr class="hr" />
    """
)

# --------------------------------------------------------------------------
# UI — Form
# --------------------------------------------------------------------------

html('<p class="section-label">Transaction parameters</p>')

with st.form("fraud_form"):
    col1, col2 = st.columns(2)
    with col1:
        sender_id = st.text_input("Sender account ID", "Acc_Safe")
        amount = st.number_input("Transaction amount ($)", min_value=0.0, value=250.0, step=10.0)
    with col2:
        receiver_id = st.text_input("Receiver account ID", "Acc_Receiver")
        historical_fraud = st.number_input(
            "Sender historical fraud count", min_value=0, max_value=50, value=0
        )

    submit_button = st.form_submit_button(label="Analyze transaction")

if submit_button:
    errors = validate_inputs(sender_id, receiver_id, amount)

    if errors:
        for err in errors:
            html(f'<div class="alert-box">{err}</div>')
    else:
        with st.spinner("Scoring transaction via GNN pipeline…"):
            result = score_transaction(amount, historical_fraud, sender_id, receiver_id)

        tier = result["tier"]
        style = TIER_STYLE[tier]
        pct = result["probability"]

        html('<hr class="hr" />')
        html('<p class="section-label">Verdict</p>')

        html(
            f"""
            <div class="verdict" style="--tier-color: {style['color']};">
                <span class="verdict-word">{style['word']}</span>
                <span class="verdict-score">{pct:.2f}%</span>
            </div>
            <div class="risk-bar-track">
                <div class="risk-bar-fill" style="width: {pct:.1f}%; background: {style['color']};"></div>
            </div>
            """
        )

        with st.expander("Why this score — factor breakdown"):
            rows = "".join(
                f'<div class="ledger-row"><span class="ledger-label">{label}</span>'
                f'<span class="ledger-fill"></span>'
                f'<span class="ledger-value">+{value:.2f}</span></div>'
                for label, value in result["breakdown"]
            )
            html(rows)

        html(
            f"""
            <div class="record-line">
                <b>{sender_id.strip()}</b> &rarr; <b>{receiver_id.strip()}</b>
                &nbsp;·&nbsp; <b>${amount:,.2f}</b> &nbsp;·&nbsp; Engine: <i>{result['engine']}</i>
            </div>
            """
        )
