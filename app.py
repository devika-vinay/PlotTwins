import pandas as pd
import streamlit as st
from pathlib import Path

from utilities.narrative_utils import get_or_create_narrative

CACHE_DIR = Path("data/cache")

USER_DASHBOARD_PATH = CACHE_DIR / "user_dashboard.parquet"
USER_MATCHES_PATH = CACHE_DIR / "user_matches.parquet"

st.set_page_config(page_title="PlotTwins", page_icon="🎬", layout="wide")

# ----------------------------
# CSS with your palette
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
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1.0;
    margin-bottom: 0.7rem;
}

.page-subtitle {
    font-size: 1.08rem;
    color: rgba(229,220,197,0.82) !important;
    max-width: 780px;
    margin-bottom: 0.8rem;
}

.hero {
    border-radius: 34px;
    padding: 2.6rem 2.6rem 2.2rem 2.6rem;
    background: linear-gradient(135deg, var(--lavender-grey) 0%, var(--blushed-brick) 100%);
    box-shadow: 0 18px 50px rgba(0,0,0,0.28);
    margin-top: 1.2rem;
    margin-bottom: 1.3rem;
}

.hero-kicker {
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-size: 0.82rem;
    opacity: 0.9;
    font-weight: 700;
    margin-bottom: 0.9rem;
}

.hero-title {
    font-size: 3.3rem;
    line-height: 1.0;
    font-weight: 900;
    margin-bottom: 0.8rem;
    color: white !important;
}

.hero-sub {
    font-size: 1.15rem;
    max-width: 760px;
    opacity: 0.96;
    line-height: 1.5;
    color: white !important;
}

.story-card {
    min-height: 210px;
    border-radius: 28px;
    padding: 1.4rem;
    background: linear-gradient(135deg, rgba(76,76,71,0.85), rgba(45,45,42,0.95));
    border: 1px solid rgba(229,220,197,0.08);
}

.story-label {
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.8rem;
    color: rgba(229,220,197,0.72) !important;
    margin-bottom: 0.75rem;
    font-weight: 700;
}

.story-title {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.9rem;
}

.story-copy {
    color: rgba(229,220,197,0.84) !important;
    font-size: 1rem;
    line-height: 1.6;
}

.section-head {
    font-size: 2.2rem;
    font-weight: 900;
    margin-top: 1.2rem;
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

.big-chip {
    display: inline-block;
    padding: 0.65rem 1.05rem;
    margin: 0.35rem 0.45rem 0.35rem 0;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(132,143,165,0.42), rgba(193,73,83,0.42));
    border: 1px solid rgba(229,220,197,0.08);
    font-size: 1rem;
    font-weight: 700;
    color: white !important;
}

.movie-tile {
    border-radius: 24px;
    padding: 1rem;
    min-height: 110px;
    background: linear-gradient(135deg, rgba(193,73,83,0.30), rgba(76,76,71,0.85));
    border: 1px solid rgba(229,220,197,0.08);
    display: flex;
    align-items: end;
    font-weight: 800;
    font-size: 1.05rem;
    color: white !important;
}

.match-card {
    border-radius: 28px;
    padding: 1.25rem;
    min-height: 150px;
    background: linear-gradient(135deg, rgba(132,143,165,0.18), rgba(76,76,71,0.72));
    border: 1px solid rgba(229,220,197,0.08);
    margin-bottom: 0.9rem;
}

.match-name {
    font-size: 1.7rem;
    font-weight: 900;
    margin-bottom: 0.7rem;
}

.small-note {
    color: rgba(229,220,197,0.70) !important;
    font-size: 0.92rem;
}

.spacer {
    height: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Data load
# ----------------------------
if not USER_DASHBOARD_PATH.exists():
    st.error("user_dashboard.parquet not found. Run the pipeline through step 07 first.")
    st.stop()

if not USER_MATCHES_PATH.exists():
    st.error("user_matches.parquet not found. Run the pipeline through step 06 first.")
    st.stop()

dashboard = pd.read_parquet(USER_DASHBOARD_PATH)
matches = pd.read_parquet(USER_MATCHES_PATH)

# ----------------------------
# Helpers
# ----------------------------
def split_pipe_text(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]

def split_csv_text(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]

def prettify_popularity(pop_value):
    mapping = {
        "pop_high": "Blockbuster leaning",
        "pop_mid": "Mainstream + niche mix",
        "pop_low": "Hidden gem leaning",
    }
    return mapping.get(str(pop_value), str(pop_value))

def prettify_behavior(behavior):
    mapping = {
        "strongly positive": "Generous rater",
        "slightly positive": "Easy to please",
        "neutral": "Balanced taste",
        "more critical": "Selective watcher",
        "unknown": "Still learning your rating style",
    }
    return mapping.get(str(behavior), str(behavior))

def format_era_label(raw):
    if pd.isna(raw):
        return ""
    s = str(raw).strip().replace("era_", "")
    try:
        year = int(float(s))
        return f"{year}s"
    except Exception:
        return s

def format_eras(values):
    return [format_era_label(v) for v in values if str(v).strip()]

def fallback_flavor_text(user_row):
    genres = split_csv_text(user_row.get("user_top_genres", ""))
    eras = format_eras(split_csv_text(user_row.get("user_top_eras", "")))
    pop_raw = str(user_row.get("user_popularity_pref", ""))
    behavior_raw = str(user_row.get("user_behavior_profile", ""))

    parts = []

    if len(genres) >= 3:
        parts.append(
            f"You seem drawn to {genres[0].lower()}, {genres[1].lower()}, and {genres[2].lower()} stories, "
            f"which suggests you like emotionally or narratively rich films with a strong sense of tension, character, or atmosphere."
        )
    elif genres:
        parts.append(
            f"Your taste leans toward {', '.join([g.lower() for g in genres])}, which gives your profile a distinct viewing personality."
        )

    if len(eras) >= 2:
        parts.append(
            f"You also show a soft spot for films from the {eras[0]} and {eras[1]}, "
            f"which points to an appreciation for older storytelling styles and memorable classics."
        )
    elif eras:
        parts.append(
            f"You seem especially comfortable with films from the {eras[0]}, which adds a nostalgic layer to your taste."
        )

    if pop_raw == "pop_high":
        parts.append(
            "Your watch pattern leans toward well-known and widely loved titles, so you likely connect more with films that have made a big cultural impact."
        )
    elif pop_raw == "pop_mid":
        parts.append(
            "Your taste sits between mainstream favorites and less obvious picks, which suggests you enjoy both recognizable hits and the occasional deeper cut."
        )
    elif pop_raw == "pop_low":
        parts.append(
            "Your taste leans more toward less mainstream titles, which suggests you may enjoy exploring hidden gems rather than only sticking to the biggest releases."
        )

    if behavior_raw == "strongly positive":
        parts.append(
            "You also come across as a generous viewer, someone who is usually happy to reward a film when it lands emotionally or stylistically."
        )
    elif behavior_raw == "slightly_positive" or behavior_raw == "slightly positive":
        parts.append(
            "Your ratings suggest you are fairly easy to win over when a film gives you something memorable."
        )
    elif behavior_raw == "neutral":
        parts.append(
            "At the same time, your ratings suggest a balanced viewer: you appreciate a strong film when it works, but you are not handing out praise too easily."
        )
    elif behavior_raw == "more critical":
        parts.append(
            "Your ratings also suggest that you are fairly selective, which usually means you know exactly what works for you and notice when a film falls short."
        )

    return " ".join(parts)

def display_name(raw_user):
    raw = str(raw_user)
    if raw.isdigit():
        return f"Movie Twin {raw}"
    return f"@{raw}"

def unique_match_badges(row):
    badges = []
    if row.get("same_region", 0) == 1:
        badges.append("Same area")
    if row.get("same_cluster", 0) == 1:
        badges.append("Same vibe")
    if row.get("same_fsa", 0) == 1:
        badges.append("Very nearby")
    return badges

# ----------------------------
# Top intro
# ----------------------------
st.markdown('<div class="page-kicker">PlotTwins</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Find your movie twin.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Discover your cinema identity, the kind of stories you love, and the people nearby who watch like you.</div>',
    unsafe_allow_html=True
)

username = st.text_input("Enter your Letterboxd username", placeholder="Search by username or user id")

if username:
    user_row = dashboard[
        dashboard["user"].astype(str).str.lower() == str(username).lower()
    ].copy()

    if user_row.empty:
        st.warning("No user found with that username.")
        st.stop()

    user_row = user_row.iloc[0]

    persona_name = str(user_row.get("persona_name", "Movie Explorer"))
    short_label = str(user_row.get("short_label", persona_name))
    example_movies = split_pipe_text(user_row.get("example_movies", ""))
    cluster_genres = split_pipe_text(user_row.get("top_genres", ""))
    user_genres = split_csv_text(user_row.get("user_top_genres", ""))
    user_eras = format_eras(split_csv_text(user_row.get("user_top_eras", "")))
    style_1 = prettify_popularity(user_row.get("user_popularity_pref", ""))
    style_2 = prettify_behavior(user_row.get("user_behavior_profile", ""))

    # User matches for this username
    user_matches = matches[
        matches["user"].astype(str).str.lower() == str(username).lower()
    ].copy()

    if not user_matches.empty:
        user_matches = user_matches.sort_values(
            by=["same_region", "same_cluster", "similarity"],
            ascending=[False, False, False]
        ).head(5)

    # Narrative generation / cache lookup
    try:
        narrative, newly_generated = get_or_create_narrative(user_row, user_matches)
        hero_title = narrative.get("taste_headline", persona_name) or persona_name
        taste_story = narrative.get("taste_story", "").strip() or fallback_flavor_text(user_row)
        people_story = narrative.get("people_story", "").strip() or "These are the users whose taste aligns most closely with yours."
    except Exception as e:
        hero_title = persona_name
        taste_story = fallback_flavor_text(user_row)
        people_story = "These are the users whose taste aligns most closely with yours."
        newly_generated = False
        st.info(f"Using fallback copy because narrative generation is unavailable: {e}")

    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-kicker">Your PlotTwins identity</div>
        <div class="hero-title">{hero_title}</div>
    </div>
    """, unsafe_allow_html=True)

    if newly_generated:
        st.caption("Generated a fresh narrative and saved it for future visits.")

    # Story row
    c1, c2, c3 = st.columns(3)

    with c1:
        genre_html = "".join([f'<span class="big-chip">{g}</span>' for g in user_genres[:3]])
        st.markdown(f"""
        <div class="story-card">
            <div class="story-label">Your top genres</div>
            <div class="story-title">This is your taste core.</div>
            <div>{genre_html if genre_html else '<span class="small-note">Not enough data</span>'}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        era_html = "".join([f'<span class="big-chip">{e}</span>' for e in user_eras[:2]])
        st.markdown(f"""
        <div class="story-card">
            <div class="story-label">Your favorite eras</div>
            <div class="story-title">This is your comfort zone.</div>
            <div>{era_html if era_html else '<span class="small-note">Not enough data</span>'}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="story-card">
            <div class="story-label">Your movie style</div>
            <div class="story-title">{short_label}</div>
            <div class="story-copy">{style_1}<br>{style_2}</div>
        </div>
        """, unsafe_allow_html=True)

    # Narrative section
    st.markdown('<div class="section-head">What your taste says about you</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="panel">
        <div class="story-copy">{taste_story}</div>
    </div>
    """, unsafe_allow_html=True)

    # Cluster shelf full width
    st.markdown('<div class="section-head">Your cluster shelf</div>', unsafe_allow_html=True)

    chips = "".join([f'<span class="chip">{g}</span>' for g in cluster_genres[:5]])
    st.markdown(f"""
    <div class="panel-dark">
        <div class="story-label">Shared genre DNA</div>
        <div class="story-copy" style="margin-bottom:0.8rem;">
            People in your movie circle tend to love these kinds of stories.
        </div>
        <div>{chips if chips else '<span class="small-note">No cluster genre data available</span>'}</div>
    </div>
    """, unsafe_allow_html=True)

    if example_movies:
        m1, m2, m3 = st.columns(3)
        movie_cols = [m1, m2, m3]
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        for i, movie in enumerate(example_movies[:3]):
            with movie_cols[i]:
                st.markdown(f"""
                <div class="movie-tile">{movie}</div>
                """, unsafe_allow_html=True)

    # Similar users
    st.markdown('<div class="section-head">Your people</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="small-note" style="margin-bottom:0.8rem;">{people_story}</div>',
        unsafe_allow_html=True
    )

    if user_matches.empty:
        st.info("No matches found for this user.")
    else:
        cols = st.columns(2)

        for idx, (_, row) in enumerate(user_matches.iterrows()):
            badge_bits = unique_match_badges(row)
            badge_html = "".join([f'<span class="chip">{b}</span>' for b in badge_bits])

            with cols[idx % 2]:
                st.markdown(f"""
                <div class="match-card">
                    <div class="match-name">{display_name(row['match_user'])}</div>
                    <div>{badge_html}</div>
                </div>
                """, unsafe_allow_html=True)

    # Technical expander
    with st.expander("See technical details"):
        st.write("**User details**")
        st.write(f"Cluster: {user_row.get('cluster', 'N/A')}")
        st.write(f"Region: {user_row.get('region', 'N/A')}")
        st.write(f"FSA: {user_row.get('fsa', 'N/A')}")

        st.write("**Signals**")
        st.write(f"Top genres: {user_row.get('user_top_genres', 'N/A')}")
        st.write(f"Top eras: {user_row.get('user_top_eras', 'N/A')}")
        st.write(f"Popularity tendency: {user_row.get('user_popularity_pref', 'N/A')}")
        st.write(f"Behavior profile: {user_row.get('user_behavior_profile', 'N/A')}")

        st.write("**Narrative**")
        st.write(f"Headline: {hero_title}")
        st.write(f"Taste story: {taste_story}")
        st.write(f"People story: {people_story}")

        debug_matches = user_matches.copy()
        if not debug_matches.empty:
            show_cols = [
                c for c in [
                    "match_user", "similarity", "same_region", "same_fsa",
                    "same_cluster", "match_region", "match_fsa",
                    "user_cluster", "match_cluster"
                ] if c in debug_matches.columns
            ]
            st.write("**Match diagnostics**")
            st.dataframe(debug_matches[show_cols], use_container_width=True)