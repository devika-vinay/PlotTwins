import pandas as pd
import streamlit as st
from pathlib import Path

from utilities.event_narrative_utils import get_or_create_event_narrative

CACHE_DIR = Path("data/cache")
EVENT_SUGGESTIONS_PATH = CACHE_DIR / "event_suggestions.parquet"

st.set_page_config(
    page_title="Business Event Planner",
    page_icon="🎟️",
    layout="wide"
)

# ----------------------------
# Page styling
# ----------------------------
st.markdown("""
<style>
:root {
    --graphite: #2D2D2A;
    --charcoal: #4C4C47;
    --lavender-grey: #848FA5;
    --blushed-brick: #C14953;
    --sand-dune: #E5DCC5;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(132,143,165,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(193,73,83,0.16), transparent 24%),
        linear-gradient(180deg, var(--graphite) 0%, #22221f 45%, #1d1d1a 100%);
    color: var(--sand-dune);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: var(--sand-dune) !important;
}

.page-kicker {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.8rem;
    color: rgba(229,220,197,0.75) !important;
    margin-bottom: 0.4rem;
    font-weight: 700;
}

.page-title {
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.0;
    margin-bottom: 0.7rem;
}

.page-subtitle {
    font-size: 1.08rem;
    color: rgba(229,220,197,0.82) !important;
    max-width: 820px;
    margin-bottom: 0.8rem;
}

.hero {
    border-radius: 34px;
    padding: 2.3rem 2.3rem 2rem 2.3rem;
    background: linear-gradient(135deg, var(--lavender-grey) 0%, var(--blushed-brick) 100%);
    box-shadow: 0 18px 50px rgba(0,0,0,0.28);
    margin-top: 1.1rem;
    margin-bottom: 1.2rem;
}

.hero-kicker {
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-size: 0.82rem;
    opacity: 0.9;
    font-weight: 700;
    margin-bottom: 0.9rem;
    color: white !important;
}

.hero-title {
    font-size: 2.5rem;
    line-height: 1.05;
    font-weight: 900;
    margin-bottom: 0.9rem;
    color: white !important;
}

.hero-sub {
    font-size: 1.08rem;
    max-width: 860px;
    opacity: 0.96;
    line-height: 1.6;
    color: white !important;
}

.metric-card {
    border-radius: 24px;
    padding: 1.2rem;
    background: linear-gradient(135deg, rgba(76,76,71,0.82), rgba(45,45,42,0.96));
    border: 1px solid rgba(229,220,197,0.08);
    min-height: 140px;
}

.metric-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.78rem;
    color: rgba(229,220,197,0.72) !important;
    margin-bottom: 0.65rem;
    font-weight: 700;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.25;
    color: white !important;
}

.metric-sub {
    margin-top: 0.55rem;
    font-size: 0.95rem;
    color: rgba(229,220,197,0.82) !important;
    line-height: 1.5;
}

.section-head {
    font-size: 2rem;
    font-weight: 900;
    margin-top: 1.1rem;
    margin-bottom: 0.8rem;
}

.panel {
    background: rgba(132,143,165,0.10);
    border: 1px solid rgba(229,220,197,0.08);
    border-radius: 28px;
    padding: 1.35rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.panel-dark {
    background: linear-gradient(135deg, rgba(76,76,71,0.78), rgba(45,45,42,0.95));
    border: 1px solid rgba(229,220,197,0.08);
    border-radius: 28px;
    padding: 1.35rem;
    margin-bottom: 1rem;
}

.chip {
    display: inline-block;
    padding: 0.45rem 0.9rem;
    margin: 0.25rem 0.35rem 0.25rem 0;
    border-radius: 999px;
    background: rgba(132,143,165,0.22);
    border: 1px solid rgba(229,220,197,0.08);
    font-size: 0.94rem;
    font-weight: 600;
    color: var(--sand-dune) !important;
}

.bullet-line {
    margin-bottom: 0.85rem;
    line-height: 1.7;
    color: rgba(229,220,197,0.90) !important;
    font-size: 1rem;
}

.small-note {
    color: rgba(229,220,197,0.70) !important;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------
def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def normalize_why_this_works(value):
    if isinstance(value, list):
        return [safe_text(v) for v in value if safe_text(v)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


# ----------------------------
# Data load
# ----------------------------
if not EVENT_SUGGESTIONS_PATH.exists():
    st.error("event_suggestions.parquet not found. Run the pipeline through step 13 first.")
    st.stop()

event_suggestions = pd.read_parquet(EVENT_SUGGESTIONS_PATH)

# ----------------------------
# Page intro
# ----------------------------
st.markdown('<div class="page-kicker">PlotTwins for Exhibitors</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Local Event Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Enter a theatre FSA to generate a screening concept that will bring audiences together and drive revenue.</div>',
    unsafe_allow_html=True
)

fsa_input = st.text_input(
    "Enter theatre FSA",
    placeholder="Example: M4Y"
)

if fsa_input:
    fsa_input = fsa_input.strip().upper()

    fsa_results = event_suggestions[
        event_suggestions["fsa"].astype(str).str.upper() == fsa_input
    ].copy()

    if fsa_results.empty:
        st.warning("No event suggestions found for that FSA.")
        st.stop()

    fsa_results = fsa_results.sort_values(
        by="business_score",
        ascending=False
    ).reset_index(drop=True)

    with st.spinner("Generating local event concept..."):
        narrative = get_or_create_event_narrative(fsa_input, fsa_results)

    event_title = safe_text(narrative.get("event_title"))
    event_pitch = safe_text(narrative.get("event_pitch"))
    persona_name = safe_text(narrative.get("persona_name"))
    event_theme = safe_text(narrative.get("event_theme"))
    primary_movie = safe_text(narrative.get("primary_movie"))
    why_points = normalize_why_this_works(narrative.get("why_this_works"))

    top_row = fsa_results.iloc[0]
    local_share_pct = round(float(top_row["cluster_share_in_fsa"]) * 100, 1)
    top_genres = safe_text(top_row.get("top_genres"))

    # ----------------------------
    # Hero section
    # ----------------------------
    st.markdown(f"""
    <div class="hero">
        <div class="hero-kicker">Recommended event for {fsa_input}</div>
        <div class="hero-title">{event_title}</div>
        <div class="hero-sub">{event_pitch}</div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------
    # Key info row
    # ----------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Target audience</div>
            <div class="metric-value">{persona_name}</div>
            <div class="metric-sub">{local_share_pct}% of the strongest local audience signal in this FSA.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary screening title</div>
            <div class="metric-value">{primary_movie}</div>
            <div class="metric-sub">Built from the highest-ranked representative movie for this local audience.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Seasonal context</div>
            <div class="metric-value">{event_theme}</div>
            <div class="metric-sub">Used as background context for the generated programming concept.</div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------
    # Why this works
    # ----------------------------
    st.markdown('<div class="section-head">Why this works</div>', unsafe_allow_html=True)

    why_html = ""
    if why_points:
        for point in why_points:
            why_html += f'<div class="bullet-line">• {point}</div>'
    else:
        why_html = '<div class="bullet-line">No business rationale was returned for this event yet.</div>'

    st.markdown(f"""
    <div class="panel">
        {why_html}
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------
    # Audience flavor
    # ----------------------------
    st.markdown('<div class="section-head">Audience profile</div>', unsafe_allow_html=True)

    genre_chips = []
    if top_genres:
        genre_chips = [g.strip() for g in top_genres.split("|") if g.strip()]

    chips_html = "".join([f'<span class="chip">{g}</span>' for g in genre_chips])

    st.markdown(f"""
    <div class="panel-dark">
        <div class="metric-label">Local taste signals</div>
        <div class="metric-value" style="font-size: 1.2rem; margin-bottom: 0.75rem;">{persona_name}</div>
        <div>{chips_html if chips_html else '<span class="small-note">No genre profile available.</span>'}</div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------
    # Technical details
    # ----------------------------
    with st.expander("See technical details"):
        st.write("**Narrative output**")
        st.json(narrative)

        st.write("**Underlying ranked suggestions for this FSA**")
        st.dataframe(
            fsa_results[
                [
                    "fsa",
                    "cluster",
                    "persona_name",
                    "event_theme",
                    "display_title",
                    "cluster_share_in_fsa",
                    "business_score",
                    "top_genres",
                ]
            ],
            use_container_width=True
        )