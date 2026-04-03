import React, { useState } from "react";
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


function GenreIcons({ genres = [] }) {
  const iconMap = {
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

  return (
    <div className="genre-icon-row">
      {genres.map((g) => (
        <div key={g} className="genre-icon">
          {iconMap[g] && <img src={iconMap[g]} alt={g} />}
          <span>{g}</span>
        </div>
      ))}
    </div>
  );
}


function TechnicalDetails({ data }) {
  const [open, setOpen] = useState(false);

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
  if (!points || points.length === 0) {
    return (
      <div className="panel">
        <div className="bullet-line">No business rationale was returned for this event yet.</div>
      </div>
    );
  }

  return (
    <div className="panel">
      {points.map((point, idx) => (
        <div key={idx} className="bullet-line">
          • {point}
        </div>
      ))}
    </div>
  );
}

export default function BusinessEventPlanner() {
  const [fsaInput, setFsaInput] = useState("");
  const [submittedFsa, setSubmittedFsa] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(e) {
    e.preventDefault();

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

  return (
    <div className="app-shell">
      <main className="container">
        <div className="page-kicker">PlotTwins for Exhibitors</div>
        <div className="page-title">Local Event Planner</div>
        <div className="page-subtitle">
          Enter a theatre FSA to generate a screening concept that will bring audiences together and drive revenue.
        </div>

        <form className="search-form" onSubmit={handleSearch}>
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
            <div className="hero">
              <div className="hero-kicker">Recommended event for {submittedFsa}</div>
              <div className="hero-title">{data.hero.event_title}</div>
              <div className="hero-sub">{data.hero.event_pitch}</div>
            </div>

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

            <div className="section-head">Why this works</div>
            <WhyThisWorks points={data.why_this_works} />

            <div className="section-head">Audience profile</div>
            <div className="panel-dark">
              <div className="metric-label">Local taste signals</div>
              <div className="metric-value audience-metric-value">
                {data.audience_profile.persona_name}
              </div>
              <GenreIcons genres={data.audience_profile.top_genres || []} />
            </div>

            <TechnicalDetails data={data.technical_details} />
          </>
        )}
      </main>
    </div>
  );
}