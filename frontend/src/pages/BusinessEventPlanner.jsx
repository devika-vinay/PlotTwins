import React, { useMemo, useState } from "react";
import { fetchBusinessEvent } from "../api";
import "../styles.css";

import actionIcon from "../assets/genres/action.png";
import adventureIcon from "../assets/genres/adventure.png";
import animationIcon from "../assets/genres/animation.png";
import comedyIcon from "../assets/genres/comedy.png";
import crimeIcon from "../assets/genres/crime.png";
import dramaIcon from "../assets/genres/drama.png";
import familyIcon from "../assets/genres/family.png";
import fantasyIcon from "../assets/genres/fantasy.png";
import horrorIcon from "../assets/genres/horror.png";
import mysteryIcon from "../assets/genres/mystery.png";
import romanceIcon from "../assets/genres/romance.png";
import scifiIcon from "../assets/genres/science.png";
import thrillerIcon from "../assets/genres/thriller.png";
import tvIcon from "../assets/genres/tv.png";
import warIcon from "../assets/genres/war.png";
import westernIcon from "../assets/genres/western.png";

const genreIconMap = {
  Action: actionIcon,
  Adventure: adventureIcon,
  Animation: animationIcon,
  Comedy: comedyIcon,
  Crime: crimeIcon,
  Drama: dramaIcon,
  Family: familyIcon,
  Fantasy: fantasyIcon,
  Horror: horrorIcon,
  Mystery: mysteryIcon,
  Romance: romanceIcon,
  "Science Fiction": scifiIcon,
  Thriller: thrillerIcon,
  "TV Movie": tvIcon,
  War: warIcon,
  Western: westernIcon,
};

function GenreIcons({ genres = [] }) {
  // Display a row of genre icons for the given genres
  return (
    <div className="genre-icon-row">
      {genres.map((g) => (
        <div key={g} className="genre-icon">
          {genreIconMap[g] && <img src={genreIconMap[g]} alt={g} />}
          <span>{g}</span>
        </div>
      ))}
    </div>
  );
}

function TechnicalDetails({ data }) {
  const [open, setOpen] = useState(false);
  // Collapsible panel showing raw JSON and underlying ranked suggestions
  return (
    <div className="technical-wrapper">
      <button className="expander-btn" onClick={() => setOpen((prev) => !prev)}>
        {open ? "Hide technical details" : "See technical details"}
      </button>

      {open && (
        <div className="panel technical-panel">
          <div className="tech-section">
            <h4>Narrative output</h4>
            <pre className="json-block">
              {JSON.stringify(data.narrative, null, 2)}
            </pre>
          </div>

          <div className="tech-section">
            <h4>Underlying ranked suggestions for this FSA</h4>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    {data.ranked_suggestions?.[0] &&
                      Object.keys(data.ranked_suggestions[0]).map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {(data.ranked_suggestions || []).map((row, idx) => (
                    <tr key={idx}>
                      {Object.keys(row).map((col) => (
                        <td key={col}>{String(row[col] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function WhyThisWorks({ points }) {
  // Display rationale points for the suggested event, or fallback text
  if (!points || points.length === 0) {
    return (
      <div className="why-list">
        <div className="why-item">
          <span className="why-check">✓</span>
          <span>No business rationale was returned for this event yet.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="why-list">
      {points.map((point, idx) => (
        <div key={idx} className="why-item">
          <span className="why-check">✓</span>
          <span>{point}</span>
        </div>
      ))}
    </div>
  );
}

function normalizeThemeKey(value = "") {
  const text = String(value).toLowerCase();
  // Map text from the event theme or persona name to a CSS class for hero styling
  if (text.includes("horror") || text.includes("spooky") || text.includes("halloween")) return "theme-horror";
  if (text.includes("spring")) return "theme-spring";
  if (text.includes("summer")) return "theme-summer";
  if (text.includes("holiday") || text.includes("winter")) return "theme-winter";
  if (text.includes("family")) return "theme-family";
  if (text.includes("fantasy") || text.includes("adventure")) return "theme-fantasy";
  if (text.includes("crime") || text.includes("thriller") || text.includes("mystery")) return "theme-crime";

  return "theme-default";
}

export default function BusinessEventPlanner() {
  const [fsaInput, setFsaInput] = useState("");
  const [submittedFsa, setSubmittedFsa] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(e) {
    e.preventDefault();
    // Normalize FSA input
    const trimmed = fsaInput.trim().toUpperCase();
    if (!trimmed) return;

    setLoading(true);
    setError("");

    try {
      const result = await fetchBusinessEvent(trimmed);
      setData(result);
      setSubmittedFsa(trimmed);
    } catch (err) {
      setData(null);
      setError(err.message || "Failed to fetch event suggestion.");
    } finally {
      setLoading(false);
    }
  }
  // Determine hero banner theme based on event_theme or persona_name
  const heroThemeClass = useMemo(() => {
    return normalizeThemeKey(data?.metrics?.event_theme || data?.metrics?.persona_name || "");
  }, [data]);

  return (
    <div className="app-shell">
      <main className="container">
        <div className="page-kicker planner-kicker">PlotTwins for Exhibitors</div>
        <div className="page-title">Local Event Planner</div>
        <div className="page-subtitle planner-subtitle">
          Enter a theatre FSA to generate a screening concept that will bring audiences together and drive revenue based on local taste clusters.
        </div>

        <form className="search-form planner-search" onSubmit={handleSearch}>
          <input
            className="search-input"
            type="text"
            value={fsaInput}
            onChange={(e) => setFsaInput(e.target.value)}
            placeholder="Example: M4Y"
          />
          <button className="search-button" type="submit" disabled={loading}>
            {loading ? "Loading..." : "Search"}
          </button>
        </form>

        {error && <div className="message warning">{error}</div>}

        {!data && !loading && !error && (
          <div className="empty-state">
            Enter a theatre FSA to generate a local event concept.
          </div>
        )}

        {data && (
          <>
            {/* Hero section showing recommended event */}
            <div className={`hero planner-hero ${heroThemeClass}`}>
              <div className="hero-overlay" />
              <div className="hero-content">
                <div className="hero-kicker">Recommended event for {submittedFsa}</div>
                <div className="hero-title">{data.hero.event_title}</div>
                <div className="hero-sub">{data.hero.event_pitch}</div>
              </div>
            </div>
            {/* Key metrics cards for the event */}
            <div className="metric-grid">
              <div className="metric-card">
                <div className="metric-label">Target audience</div>
                <div className="metric-value">{data.metrics.persona_name}</div>
                <div className="metric-sub">
                  {data.metrics.local_share_pct}% of the strongest local audience signal in this FSA.
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Primary screening title</div>
                <div className="metric-value">{data.metrics.primary_movie}</div>
                <div className="metric-sub">
                  Built from the highest-ranked representative movie for this local audience.
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Seasonal context</div>
                <div className="metric-value">{data.metrics.event_theme}</div>
                <div className="metric-sub">
                  Used as background context for the generated programming concept.
                </div>
              </div>
            </div>

            <div className="panel-dark planner-bottom-card">
              <div className="planner-bottom-grid">
                <div>
                  <div className="section-head planner-section-head">Why this works</div>
                  <WhyThisWorks points={data.why_this_works} />
                </div>

                <div>
                  <div className="section-head planner-section-head">Audience profile</div>
                  <div className="planner-audience-card">
                    <div className="metric-label">Local taste signals</div>
                    <div className="metric-value audience-metric-value">
                      {data.audience_profile.persona_name}
                    </div>
                    <GenreIcons genres={data.audience_profile.top_genres || []} />
                  </div>
                </div>
              </div>
            </div>
            {/* Raw technical info panel */}
            <TechnicalDetails data={data.technical_details} />
          </>
        )}
      </main>
    </div>
  );
}