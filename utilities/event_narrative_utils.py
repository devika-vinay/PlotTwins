from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_DIR = Path("data/cache")
EVENT_NARRATIVES_PATH = CACHE_DIR / "event_narratives.parquet"

client = OpenAI()


def _ensure_cache_file() -> pd.DataFrame:
    if EVENT_NARRATIVES_PATH.exists():
        return pd.read_parquet(EVENT_NARRATIVES_PATH)

    empty_df = pd.DataFrame(
        columns=[
            "fsa",
            "generated_at",
            "event_title",
            "event_pitch",
            "why_this_works",
            "primary_movie",
            "secondary_movie",
            "tertiary_movie",
            "persona_name",
            "event_theme",
            "raw_response",
        ]
    )
    return empty_df


def _build_prompt(fsa: str, fsa_rows: pd.DataFrame) -> str:
    top_rows = (
        fsa_rows.sort_values("business_score", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )

    suggestions_text = []
    for i, (_, row) in enumerate(top_rows.iterrows(), start=1):
        suggestions_text.append(
            f"""
Suggestion {i}
- Persona: {row['persona_name']}
- Candidate movie: {row['display_title']}
- Top genres: {row['top_genres']}
- Local audience share: {round(float(row['cluster_share_in_fsa']) * 100, 1)}%
- Business explanation: {row['business_explanation']}
""".strip()
        )

    joined_suggestions = "\n\n".join(suggestions_text)

    return f"""
You are a cinema programming strategist designing a SPECIAL EVENT for a theatre.

Create one grounded, creative, and realistic cinema event concept for a theatre in this FSA.

Your goal is NOT to simply restate the supplied event theme.
Your goal is to DESIGN a cinema event that a theatre could realistically market and run.

The event should feel like a real activation, such as:
- a themed screening night
- a double feature
- trivia night tied to the movie
- costume / dress-up element
- audience voting
- fandom competition
- themed concessions or photo moment
- other light, realistic audience participation

The supplied event theme is optional background context and should not dictate the event format.
The event format should primarily come from audience taste and movie fit.

Event design guidelines:
- Choose ONE clear event format (e.g. trivia night, cosplay night, double feature, themed screening)
- Include at most ONE or TWO supporting activations (not more)
- Do not stack too many elements into one event
- The event should feel realistic for a theatre to execute in one evening

Avoid repeating the same event format across different FSAs.
If multiple strong movies are provided, you may choose:
- a single-feature event
- a themed screening
- a community or social event
- OR a double feature if it truly fits

Do not default to double feature.

Write the event_pitch as a short 2-4 sentence programming blurb.
It should feel cinematic, specific, inviting, and audience-facing rather than analytical.
It should describe the event experience, not just summarize the audience data.

Write why_this_works as 2-3 concise bullet points for an internal business audience.
These bullet points should be grounded in the data and can be more analytical than the pitch.

Rules:
- Use only the information provided below
- Do not invent movie titles
- Do not invent audience facts beyond what is provided
- Write in a polished, natural, human tone for cinema programmers
- Avoid robotic business jargon like "activate the segment", "targets local users", or "aligns with the profile"
- The event title should feel marketable, audience-facing, and more specific than the supplied theme label
- The pitch should sound like a real special-event concept a theatre could present internally or publicly
- Include at least one realistic audience interaction or event activation idea
- Keep the writing grounded in the supplied data, but make it vivid and engaging
- Prefer the strongest local audience signals
- Do not include the FSA code in the event title
- Do not include percentages in the event title or event pitch unless absolutely necessary
- Save quantitative reasoning for why_this_works
- Use only the provided candidate movies as movie recommendations
- The primary_movie, secondary_movie, and tertiary_movie fields must come only from the provided suggestions
- If fewer than 3 strong movie options are available, use null for the remaining movie fields
- Return valid JSON only
- Do not wrap the JSON in markdown

Provided suggestions:
{joined_suggestions}

Return JSON with exactly these keys:
event_title
event_pitch
why_this_works
primary_movie
secondary_movie
tertiary_movie
persona_name
event_theme
""".strip()

def _generate_event_narrative(fsa: str, fsa_rows: pd.DataFrame) -> dict:
    prompt = _build_prompt(fsa, fsa_rows)

    response = client.responses.create(
        model="gpt-5",
        input=prompt,
    )

    raw_text = response.output_text.strip()

    data = json.loads(raw_text)
    data["raw_response"] = raw_text
    return data


def get_or_create_event_narrative(fsa: str, fsa_rows: pd.DataFrame) -> dict:
    fsa = str(fsa).strip().upper()

    cache_df = _ensure_cache_file()

    existing = cache_df[cache_df["fsa"].astype(str).str.upper() == fsa]
    if not existing.empty:
        return existing.iloc[0].to_dict()

    generated = _generate_event_narrative(fsa, fsa_rows)

    new_row = {
        "fsa": fsa,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "event_title": generated.get("event_title", ""),
        "event_pitch": generated.get("event_pitch", ""),
        "why_this_works": generated.get("why_this_works", ""),
        "primary_movie": generated.get("primary_movie", ""),
        "secondary_movie": generated.get("secondary_movie", ""),
        "tertiary_movie": generated.get("tertiary_movie", ""),
        "persona_name": generated.get("persona_name", ""),
        "event_theme": generated.get("event_theme", ""),
        "raw_response": generated.get("raw_response", ""),
    }

    updated_cache = pd.concat([cache_df, pd.DataFrame([new_row])], ignore_index=True)
    updated_cache.to_parquet(EVENT_NARRATIVES_PATH, index=False)

    return new_row