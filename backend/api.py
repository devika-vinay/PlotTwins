from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utilities.narrative_utils import get_or_create_narrative
from utilities.event_narrative_utils import get_or_create_event_narrative

import json
import ast
from fastapi.responses import JSONResponse


CACHE_DIR = Path("data/cache")
USER_DASHBOARD_PATH = CACHE_DIR / "user_dashboard.parquet"
USER_MATCHES_PATH = CACHE_DIR / "user_matches.parquet"
EVENT_SUGGESTIONS_PATH = Path("data/cache/event_suggestions.parquet")


app = FastAPI(title="PlotTwins API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://plot-twins.vercel.app",
        "https://plot-twins.vercel.app/business"
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

def extract_numeric_eras_from_value(value):
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


def normalize_cluster_value(value):
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
    results = []

    for _, row in user_matches.iterrows():
        results.append(
            {
                "match_user": str(row.get("match_user", "")),
                "display_name": display_name(row.get("match_user", "")),
                "similarity": None if pd.isna(row.get("similarity")) else float(row.get("similarity")),
                "same_region": int(row.get("same_region", 0)),
                "same_cluster": int(row.get("same_cluster", 0)),
                "same_fsa": int(row.get("same_fsa", 0)),
                "match_region": None if pd.isna(row.get("match_region")) else str(row.get("match_region")),
                "match_fsa": None if pd.isna(row.get("match_fsa")) else str(row.get("match_fsa")),
                "user_cluster": normalize_cluster_value(row.get("user_cluster")),
                "match_cluster": normalize_cluster_value(row.get("match_cluster")),
                "badges": unique_match_badges(row),
            }
        )

    return results


def build_cluster_cards(dashboard: pd.DataFrame):
    working = dashboard.copy()
    working["cluster_norm"] = working["cluster"].apply(normalize_cluster_value)

    counts = (
        working.groupby("cluster_norm", dropna=False)["user"]
        .nunique()
        .reset_index(name="user_count")
    )

    representative = (
        working.sort_values("user")
        .drop_duplicates(subset=["cluster_norm"])
        .copy()
    )

    merged = representative.merge(counts, on="cluster_norm", how="left")

    cards = []
    for _, row in merged.iterrows():
        cluster_value = row.get("cluster_norm")
        if cluster_value is None:
            continue

        cards.append(
            {
                "cluster": cluster_value,
                "persona_name": str(row.get("persona_name", "Movie Explorer")),
                "top_genres": split_pipe_text(row.get("top_genres", "")),
                "example_movies": split_pipe_text(row.get("example_movies", "")),
                "user_count": int(row.get("user_count", 0)),
            }
        )

    cards.sort(key=lambda x: x["user_count"], reverse=True)
    return cards

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
    if not EVENT_SUGGESTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="event_suggestions.parquet not found. Run the pipeline through step 13 first.",
        )

    event_suggestions = pd.read_parquet(EVENT_SUGGESTIONS_PATH)

    fsa_input = str(fsa).strip().upper()

    fsa_results = event_suggestions[
        event_suggestions["fsa"].astype(str).str.upper() == fsa_input
    ].copy()

    if fsa_results.empty:
        raise HTTPException(
            status_code=404,
            detail="No event suggestions found for that FSA."
        )

    fsa_results = fsa_results.sort_values(
        by="business_score",
        ascending=False
    ).reset_index(drop=True)

    narrative = get_or_create_event_narrative(fsa_input, fsa_results)

    event_title = safe_text(narrative.get("event_title"))
    event_pitch = safe_text(narrative.get("event_pitch"))
    persona_name = safe_text(narrative.get("persona_name"))
    event_theme = safe_text(narrative.get("event_theme"))
    primary_movie = safe_text(narrative.get("primary_movie"))
    why_points = normalize_why_this_works(narrative.get("why_this_works"))

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
    return {"status": "ok"}


@app.get("/api/user/{username}")
def get_user(username: str):
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

    dashboard = pd.read_parquet(USER_DASHBOARD_PATH)
    matches = pd.read_parquet(USER_MATCHES_PATH)

    user_row_df = dashboard[
        dashboard["user"].astype(str).str.lower() == str(username).lower()
    ].copy()

    if user_row_df.empty:
        raise HTTPException(status_code=404, detail="No user found with that username.")

    user_row = user_row_df.iloc[0]

    user_matches = matches[
        matches["user"].astype(str).str.lower() == str(username).lower()
    ].copy()

    if not user_matches.empty:
        user_matches = user_matches.sort_values(
            by=["same_region", "same_cluster", "similarity"],
            ascending=[False, False, False],
        ).head(5)

    try:
        narrative, newly_generated = get_or_create_narrative(user_row, user_matches)
        hero_title = narrative.get("taste_headline", str(user_row.get("persona_name", "Movie Explorer"))) or str(
            user_row.get("persona_name", "Movie Explorer")
        )
        taste_story = narrative.get("taste_story", "").strip() or fallback_flavor_text(user_row)
        people_story = narrative.get("people_story", "").strip() or "These are the users whose taste aligns most closely with yours."
        narrative_fallback_used = False
    except Exception:
        hero_title = str(user_row.get("persona_name", "Movie Explorer"))
        taste_story = fallback_flavor_text(user_row)
        people_story = "These are the users whose taste aligns most closely with yours."
        newly_generated = False
        narrative_fallback_used = True

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