# Path & I/O imports
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Utilities for narrative generation
from utilities.narrative_utils import get_or_create_narrative
from utilities.event_narrative_utils import get_or_create_event_narrative

import json
import ast
from fastapi.responses import JSONResponse

# Cache and data paths
CACHE_DIR = Path("data/cache")
USER_DASHBOARD_PATH = CACHE_DIR / "user_dashboard.parquet"
USER_MATCHES_PATH = CACHE_DIR / "user_matches.parquet"
USER_PROFILES_PATH = CACHE_DIR / "user_profiles.parquet"
EVENT_SUGGESTIONS_PATH = Path("data/cache/event_suggestions.parquet")

# FastAPI app setup
app = FastAPI(title="PlotTwins API")
# Configure CORS so frontend apps can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://plot-twins.vercel.app",
        "https://plot-twins.vercel.app/business",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helper functions for text parsing and formatting
# -----------------------------
def split_pipe_text(value):
    """Split a string separated by '|' into a list"""
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def split_csv_text(value):
    """Split a string separated by ',' into a list"""
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def prettify_popularity(pop_value):
    """Map raw popularity codes to human-readable strings"""
    mapping = {
        "pop_high": "Blockbuster leaning",
        "pop_mid": "Mainstream + niche mix",
        "pop_low": "Hidden gem leaning",
    }
    return mapping.get(str(pop_value), str(pop_value))


def prettify_behavior(behavior):
    """Map raw behavior codes to human-readable strings"""
    mapping = {
        "strongly positive": "Generous rater",
        "slightly positive": "Easy to please",
        "neutral": "Balanced taste",
        "more critical": "Selective watcher",
        "unknown": "Still learning your rating style",
    }
    return mapping.get(str(behavior), str(behavior))


def format_era_label(raw):
    """Convert raw era value to readable format like '1990s'"""
    if pd.isna(raw):
        return ""
    s = str(raw).strip().replace("era_", "")
    try:
        year = int(float(s))
        return f"{year}s"
    except Exception:
        return s


def format_eras(values):
    """Apply format_era_label to a list of values"""
    return [format_era_label(v) for v in values if str(v).strip()]

def extract_numeric_eras_from_value(value):
    """Extract numeric years from a CSV-style string"""
    if pd.isna(value):
        return []

    eras = []
    for raw in str(value).split(","):
        cleaned = str(raw).strip().replace("era_", "")
        if not cleaned:
            continue
        try:
            eras.append(int(float(cleaned)))
        except Exception:
            continue
    return eras


def compute_era_bounds(dashboard: pd.DataFrame):
    """Compute min/max era across all users in the dashboard"""
    all_eras = []

    if "user_top_eras" not in dashboard.columns:
        return {"min": None, "max": None}

    for value in dashboard["user_top_eras"].dropna():
        all_eras.extend(extract_numeric_eras_from_value(value))

    if not all_eras:
        return {"min": None, "max": None}

    return {
        "min": int(min(all_eras)),
        "max": int(max(all_eras)),
    }

# -----------------------------
# User radar / persona utilities
# -----------------------------
def build_persona_radar(user_row):
    """Compute normalized radar metrics for user's taste profile"""
    def safe_float(value, default=0.0):
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    genre_focus = safe_float(user_row.get("top_genre_share", 0.0))
    genre_entropy = safe_float(user_row.get("genre_entropy", 0.0))
    classic = safe_float(user_row.get("classic_share", 0.0))
    modern = safe_float(user_row.get("modern_share", 0.0))
    english = safe_float(user_row.get("english_share", 1.0), 1.0)

    # Normalize entropy to a 0–1 UI score
    # 3.5 is a practical ceiling for display
    variety = min(max(genre_entropy / 3.5, 0.0), 1.0)
    international = min(max(1.0 - english, 0.0), 1.0)

    return [
        {"axis": "Focused Genres", "value": round(genre_focus, 3)},
        {"axis": "Explorer", "value": round(variety, 3)},
        {"axis": "Classic Films", "value": round(classic, 3)},
        {"axis": "Modern Films", "value": round(modern, 3)},
        {"axis": "Global Cinema", "value": round(international, 3)},
    ]


def fallback_flavor_text(user_row):
    """Generate fallback narrative if LLM narrative is missing"""
    genres = split_csv_text(user_row.get("user_top_genres", ""))
    eras = format_eras(split_csv_text(user_row.get("user_top_eras", "")))
    pop_raw = str(user_row.get("user_popularity_pref", ""))
    behavior_raw = str(user_row.get("user_behavior_profile", ""))

    parts = []
    # Build flavor text based on top genres
    if len(genres) >= 3:
        parts.append(
            f"You seem drawn to {genres[0].lower()}, {genres[1].lower()}, and {genres[2].lower()} stories, "
            f"which suggests you like emotionally or narratively rich films with a strong sense of tension, character, or atmosphere."
        )
    elif genres:
        parts.append(
            f"Your taste leans toward {', '.join([g.lower() for g in genres])}, which gives your profile a distinct viewing personality."
        )
    # Add flavor text based on top eras
    if len(eras) >= 2:
        parts.append(
            f"You also show a soft spot for films from the {eras[0]} and {eras[1]}, "
            f"which points to an appreciation for older storytelling styles and memorable classics."
        )
    elif eras:
        parts.append(
            f"You seem especially comfortable with films from the {eras[0]}, which adds a nostalgic layer to your taste."
        )
    # Popularity flavor
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
    # Behavior flavor
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

# -----------------------------
# Display / match helpers
# -----------------------------
def display_name(raw_user):
    """Format username or fallback numeric ID"""
    raw = str(raw_user)
    if raw.isdigit():
        return f"Movie Twin {raw}"
    return f"@{raw}"


def unique_match_badges(row):
    """Generate badge labels for user matches"""
    badges = []
    if row.get("same_region", 0) == 1:
        badges.append("Same area")
    if row.get("same_cluster", 0) == 1:
        badges.append("Same vibe")
    if row.get("same_fsa", 0) == 1:
        badges.append("Very nearby")
    return badges


def normalize_cluster_value(value):
    """Convert cluster value to int if possible"""
    if pd.isna(value):
        return None
    s = str(value)
    if s.replace(".", "", 1).isdigit():
        try:
            return int(float(s))
        except Exception:
            return s
    return s


def serialize_user_row(user_row):
    """Prepare user row for API response"""
    return {
        "user": str(user_row.get("user", "")),
        "cluster": normalize_cluster_value(user_row.get("cluster")),
        "region": None if pd.isna(user_row.get("region")) else str(user_row.get("region")),
        "fsa": None if pd.isna(user_row.get("fsa")) else str(user_row.get("fsa")),
        "persona_name": str(user_row.get("persona_name", "Movie Explorer")),
        "example_movies": split_pipe_text(user_row.get("example_movies", "")),
        "top_genres": split_pipe_text(user_row.get("top_genres", "")),
        "user_top_genres": split_csv_text(user_row.get("user_top_genres", "")),
        "user_top_eras": format_eras(split_csv_text(user_row.get("user_top_eras", ""))),
        "user_popularity_pref_raw": str(user_row.get("user_popularity_pref", "")),
        "user_popularity_pref": prettify_popularity(user_row.get("user_popularity_pref", "")),
        "user_behavior_profile_raw": str(user_row.get("user_behavior_profile", "")),
        "user_behavior_profile": prettify_behavior(user_row.get("user_behavior_profile", "")),
    }


def serialize_matches(user_matches):
    """Prepare user match data for API response."""
    results = []
    # Iterate over each row of the matches DataFrame
    for _, row in user_matches.iterrows():
        results.append(
            {
                "match_user": str(row.get("match_user", "")), # matched user's ID
                "display_name": display_name(row.get("match_user", "")), # formatted display name
                "similarity": None if pd.isna(row.get("similarity")) else float(row.get("similarity")), # similarity score
                "same_region": int(row.get("same_region", 0)), # region match flag
                "same_cluster": int(row.get("same_cluster", 0)), # cluster match flag
                "same_fsa": int(row.get("same_fsa", 0)), # FSA match flag
                "match_region": None if pd.isna(row.get("match_region")) else str(row.get("match_region")),
                "match_fsa": None if pd.isna(row.get("match_fsa")) else str(row.get("match_fsa")),
                "user_cluster": normalize_cluster_value(row.get("user_cluster")),
                "match_cluster": normalize_cluster_value(row.get("match_cluster")),
                "badges": unique_match_badges(row), # visual badges based on match characteristics
            }
        )

    return results


def build_cluster_cards(dashboard: pd.DataFrame):
    """Generate summary cards for each cluster showing persona and counts."""
    working = dashboard.copy()
    working["cluster_norm"] = working["cluster"].apply(normalize_cluster_value)
    # Count number of users per cluster
    counts = (
        working.groupby("cluster_norm", dropna=False)["user"]
        .nunique()
        .reset_index(name="user_count")
    )
    # Count number of users per cluster
    representative = (
        working.sort_values("user")
        .drop_duplicates(subset=["cluster_norm"])
        .copy()
    )
    # Merge counts into representative rows
    merged = representative.merge(counts, on="cluster_norm", how="left")

    cards = []
    for _, row in merged.iterrows():
        cluster_value = row.get("cluster_norm")
        if cluster_value is None:
            continue # skip invalid clusters

        cards.append(
            {
                "cluster": cluster_value,
                "persona_name": str(row.get("persona_name", "Movie Explorer")), # default persona name
                "top_genres": split_pipe_text(row.get("top_genres", "")), # default persona name
                "example_movies": split_pipe_text(row.get("example_movies", "")), # example movies
                "user_count": int(row.get("user_count", 0)), # number of users in this cluster
            }
        )
    # Sort cards descending by user count
    cards.sort(key=lambda x: x["user_count"], reverse=True)
    return cards

def safe_text(value) -> str:
    """Convert any input into a clean string, treating NaN/None as empty."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def normalize_why_this_works(value):
    """Normalize 'why this works' points into a clean list of strings."""
    if isinstance(value, list):
        return [safe_text(v) for v in value if safe_text(v)]

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []

        # Try parsing strings that look like Python lists
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, list):
                return [safe_text(v) for v in parsed if safe_text(v)]
        except Exception:
            pass

        # Fallback: split multiline / quoted blob into lines
        lines = [safe_text(line) for line in cleaned.split("\n") if safe_text(line)]
        if lines:
            return lines

        return [cleaned]

    return []


@app.get("/api/events/{fsa}")
def get_event_suggestion(fsa: str):
    """
    Return event suggestions for a given FSA code.
    - Reads event suggestions from parquet cache.
    - Sorts by business_score and generates narrative.
    """
    if not EVENT_SUGGESTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="event_suggestions.parquet not found. Run the pipeline through step 13 first.",
        )

    event_suggestions = pd.read_parquet(EVENT_SUGGESTIONS_PATH)

    fsa_input = str(fsa).strip().upper()
    # Filter events for this FSA
    fsa_results = event_suggestions[
        event_suggestions["fsa"].astype(str).str.upper() == fsa_input
    ].copy()

    if fsa_results.empty:
        raise HTTPException(
            status_code=404,
            detail="No event suggestions found for that FSA."
        )
    # Sort events by business score (descending)
    fsa_results = fsa_results.sort_values(
        by="business_score",
        ascending=False
    ).reset_index(drop=True)
    # Generate or fetch narrative for the FSA
    narrative = get_or_create_event_narrative(fsa_input, fsa_results)
    # Extract narrative components
    event_title = safe_text(narrative.get("event_title"))
    event_pitch = safe_text(narrative.get("event_pitch"))
    persona_name = safe_text(narrative.get("persona_name"))
    event_theme = safe_text(narrative.get("event_theme"))
    primary_movie = safe_text(narrative.get("primary_movie"))
    why_points = normalize_why_this_works(narrative.get("why_this_works"))
    # Fallback parsing if LLM response is missing
    if not why_points:
        raw_response = narrative.get("raw_response", "")
        if raw_response:
            try:
                parsed_raw = json.loads(raw_response)
                why_points = normalize_why_this_works(parsed_raw.get("why_this_works"))
            except Exception:
                pass

    top_row = fsa_results.iloc[0]
    local_share_pct = round(float(top_row["cluster_share_in_fsa"]) * 100, 1)
    top_genres = safe_text(top_row.get("top_genres"))
    genre_chips = [g.strip() for g in top_genres.split("|") if g.strip()] if top_genres else []
    # Keep only relevant columns for ranked suggestions
    ranked_cols = [
        "fsa",
        "cluster",
        "persona_name",
        "event_theme",
        "display_title",
        "cluster_share_in_fsa",
        "business_score",
        "top_genres",
    ]
    ranked_cols = [c for c in ranked_cols if c in fsa_results.columns]

    ranked_rows = fsa_results[ranked_cols].to_dict(orient="records")
    # Construct API response
    response = {
        "fsa": fsa_input,
        "hero": {
            "event_title": event_title,
            "event_pitch": event_pitch,
        },
        "metrics": {
            "persona_name": persona_name,
            "local_share_pct": local_share_pct,
            "primary_movie": primary_movie,
            "event_theme": event_theme,
        },
        "why_this_works": why_points,
        "audience_profile": {
            "persona_name": persona_name,
            "top_genres": genre_chips,
        },
        "technical_details": {
            "narrative": narrative,
            "ranked_suggestions": ranked_rows,
        },
    }

    safe_response = json.loads(json.dumps(response, default=str))
    return JSONResponse(content=safe_response)

@app.get("/api/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/api/user/{username}")
def get_user(username: str):
    """
    Return detailed user profile including:
    - Dashboard stats
    - Matches
    - Clusters
    - Era bounds
    - Persona radar
    - Narrative and fallback flavor text
    """
    # Ensure required cache files exist
    if not USER_DASHBOARD_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="user_dashboard.parquet not found. Run the pipeline through step 07 first.",
        )

    if not USER_MATCHES_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="user_matches.parquet not found. Run the pipeline through step 06 first.",
        )
    # Load user dashboard and matches
    dashboard = pd.read_parquet(USER_DASHBOARD_PATH)
    matches = pd.read_parquet(USER_MATCHES_PATH)

    # Filter for this user
    user_row_df = dashboard[
        dashboard["user"].astype(str).str.lower() == str(username).lower()
    ].copy()

    if user_row_df.empty:
        raise HTTPException(status_code=404, detail="No user found with that username.")

    user_row = user_row_df.iloc[0]
    # Load optional profile info
    profiles = pd.read_parquet(USER_PROFILES_PATH)

    profile_row_df = profiles[
        profiles["user"].astype(str).str.lower() == str(username).lower()
    ].copy()

    profile_row = profile_row_df.iloc[0] if not profile_row_df.empty else user_row
    # Filter matches for this user
    user_matches = matches[
        matches["user"].astype(str).str.lower() == str(username).lower()
    ].copy()

    if not user_matches.empty:
        # Keep top 5 matches sorted by similarity
        user_matches = user_matches.sort_values(
            by=["same_region", "same_cluster", "similarity"],
            ascending=[False, False, False],
        ).head(5)
    # Generate or fallback narrative
    try:
        narrative, newly_generated = get_or_create_narrative(user_row, user_matches)
        hero_title = narrative.get("taste_headline", str(user_row.get("persona_name", "Movie Explorer"))) or str(
            user_row.get("persona_name", "Movie Explorer")
        )
        taste_story = narrative.get("taste_story", "").strip() or fallback_flavor_text(user_row)
        people_story = narrative.get("people_story", "").strip() or "These are the users whose taste aligns most closely with yours."
        narrative_fallback_used = False
    except Exception:
        # Fallback if narrative generation fails
        hero_title = str(user_row.get("persona_name", "Movie Explorer"))
        taste_story = fallback_flavor_text(user_row)
        people_story = "These are the users whose taste aligns most closely with yours."
        newly_generated = False
        narrative_fallback_used = True

    # Build final API response
    response = {
        "user": serialize_user_row(user_row),
        "hero_title": hero_title,
        "taste_story": taste_story,
        "people_story": people_story,
        "newly_generated": newly_generated,
        "narrative_fallback_used": narrative_fallback_used,
        "matches": serialize_matches(user_matches),
        "clusters": build_cluster_cards(dashboard),
        "era_bounds": compute_era_bounds(dashboard),
        "persona_radar": build_persona_radar(profile_row),
        "technical_details": {
            "cluster": None if pd.isna(user_row.get("cluster")) else str(user_row.get("cluster")),
            "region": None if pd.isna(user_row.get("region")) else str(user_row.get("region")),
            "fsa": None if pd.isna(user_row.get("fsa")) else str(user_row.get("fsa")),
            "top_genres_raw": str(user_row.get("user_top_genres", "N/A")),
            "top_eras_raw": str(user_row.get("user_top_eras", "N/A")),
            "popularity_tendency_raw": str(user_row.get("user_popularity_pref", "N/A")),
            "behavior_profile_raw": str(user_row.get("user_behavior_profile", "N/A")),
            "headline": hero_title,
            "taste_story": taste_story,
            "people_story": people_story,
        },
    }

    return response