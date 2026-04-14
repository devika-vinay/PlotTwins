import React, { useMemo, useState, useEffect } from "react";
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from "recharts";
import { fetchUserDashboard } from "./api";
import "./styles.css";

import actionIcon from "./assets/genres/action.png";
import adventureIcon from "./assets/genres/adventure.png";
import animationIcon from "./assets/genres/animation.png";
import comedyIcon from "./assets/genres/comedy.png";
import crimeIcon from "./assets/genres/crime.png";
import dramaIcon from "./assets/genres/drama.png";
import familyIcon from "./assets/genres/family.png";
import fantasyIcon from "./assets/genres/fantasy.png";
import horrorIcon from "./assets/genres/horror.png";
import mysteryIcon from "./assets/genres/mystery.png";
import romanceIcon from "./assets/genres/romance.png";
import scifiIcon from "./assets/genres/science.png";
import thrillerIcon from "./assets/genres/thriller.png";
import tvIcon from "./assets/genres/tv.png";
import warIcon from "./assets/genres/war.png";
import westernIcon from "./assets/genres/western.png";

import blowOutPoster from "./assets/movies/blow out.jpg";
import catchingFirePoster from "./assets/movies/catching fire.jpg";
import cocoPoster from "./assets/movies/coco.jpg";
import curePoster from "./assets/movies/cure.jpg";
import daffyDucksPoster from "./assets/movies/daffy ducks quackbusters.jpg";
import everythingEverywherePoster from "./assets/movies/everything everywhere all at once.jpg";
import hungerGamesPoster from "./assets/movies/hunger games.jpg";
import lostHighwayPoster from "./assets/movies/lost highway.jpg";
import nightGalleryPoster from "./assets/movies/night gallery.jpg";
import spidermanPoster from "./assets/movies/spiderman.jpg";
import ghostOfSierraPoster from "./assets/movies/the ghost of sierra de cobre.jpg";
import girlWhoKnewTooMuchPoster from "./assets/movies/the girl who knew too much.jpg";
import girlWithDragonTattooPoster from "./assets/movies/the girl with the dragon tattoo.jpg";
import snowmanPoster from "./assets/movies/the snowman.jpg";
import thorPoster from "./assets/movies/thor ragnarok.jpg";
import toyStoryPoster from "./assets/movies/toy story.jpg";
import windRiverPoster from "./assets/movies/wind river.jpg";
import zodiacPoster from "./assets/movies/zodiac.jpg";

import avatar1 from "./assets/profilepics/p1.png";
import avatar2 from "./assets/profilepics/p2.png";
import avatar3 from "./assets/profilepics/p3.png";
import avatar4 from "./assets/profilepics/p4.png";
import avatar5 from "./assets/profilepics/p5.png";

const avatarMap = [avatar1, avatar2, avatar3, avatar4, avatar5];

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

const posterMap = {
  "Blow Out": blowOutPoster,
  "The Hunger Games Catching Fire": catchingFirePoster,
  Coco: cocoPoster,
  Cure: curePoster,
  "Daffy Ducks Quackbusters": daffyDucksPoster,
  "Everything Everywhere All At Once": everythingEverywherePoster,
  "The Hunger Games": hungerGamesPoster,
  "Lost Highway": lostHighwayPoster,
  "Night Gallery": nightGalleryPoster,
  "Spider Man Into The Spider Verse": spidermanPoster,
  "The Ghost Of Sierra De Cobre": ghostOfSierraPoster,
  "The Girl Who Knew Too Much": girlWhoKnewTooMuchPoster,
  "The Girl With The Dragon Tattoo": girlWithDragonTattooPoster,
  "The Snowman": snowmanPoster,
  "Thor Ragnarok": thorPoster,
  "Toy Story": toyStoryPoster,
  "Wind River": windRiverPoster,
  Zodiac: zodiacPoster,
};

function cleanClusterName(name) {
  if (!name) return "";
  // Remove trailing "Viewer" or "& Viewer" from cluster persona names
  return String(name).replace(/\s*&\s*Viewer$/i, "").replace(/\s+Viewer$/i, "").trim();
}

function getGenreIcon(genre) {
  // Return the matching genre icon or null if unavailable
  return genreIconMap[genre] || null;
}

function MatchCard({ match, avatarIndex }) {
  // Display a single match with avatar, name, and badges
  const avatar = avatarMap[avatarIndex % avatarMap.length];

  return (
    <div className="match-card cinematic-match-card">
      <div className="match-avatar-glow">
        <div className="match-avatar">
          <img src={avatar} alt={match.display_name} />
        </div>
      </div>

      <div className="match-name centered">{match.display_name}</div>

      <div className="chip-wrap centered">
        {(match.badges || []).map((badge) => (
          <span key={badge} className="chip twin-chip">
            {badge}
          </span>
        ))}
      </div>
    </div>
  );
}

function MovieShelf({ movies }) {
  if (!movies || movies.length === 0) return null;
  // Show up to 3 movie posters for a cluster
  return (
    <div className="movie-grid">
      {movies.slice(0, 3).map((movie) => (
        <div key={movie} className="movie-tile poster">
          <img src={posterMap[movie]} alt={movie} />
          <div className="poster-overlay">{movie}</div>
        </div>
      ))}
    </div>
  );
}

function TechnicalDetails({ data }) {
  const [open, setOpen] = useState(false);
  // Collapsible panel to show raw technical info about user
  return (
    <div className="technical-wrapper">
      <button className="expander-btn" onClick={() => setOpen((prev) => !prev)}>
        {open ? "Hide technical details" : "See technical details"}
      </button>

      {open && (
        <div className="panel technical-panel">
          <div className="tech-section">
            <h4>User details</h4>
            <p><strong>Cluster:</strong> {data.cluster || "N/A"}</p>
            <p><strong>Region:</strong> {data.region || "N/A"}</p>
            <p><strong>FSA:</strong> {data.fsa || "N/A"}</p>
          </div>

          <div className="tech-section">
            <h4>Signals</h4>
            <p><strong>Top genres:</strong> {data.top_genres_raw || "N/A"}</p>
            <p><strong>Top eras:</strong> {data.top_eras_raw || "N/A"}</p>
            <p><strong>Popularity tendency:</strong> {data.popularity_tendency_raw || "N/A"}</p>
            <p><strong>Behavior profile:</strong> {data.behavior_profile_raw || "N/A"}</p>
          </div>

          <div className="tech-section">
            <h4>Narrative</h4>
            <p><strong>Headline:</strong> {data.headline || "N/A"}</p>
            <p><strong>Taste story:</strong> {data.taste_story || "N/A"}</p>
            <p><strong>People story:</strong> {data.people_story || "N/A"}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function PersonaRadar({ data = [] }) {
  return (
    <div className="genre-radar-wrap">
      <ResponsiveContainer width="100%" height={240}>
        <RadarChart data={data}>
          <PolarGrid stroke="rgba(245,247,251,0.14)" />
          <PolarAngleAxis
            dataKey="axis"
            tick={{ fill: "rgba(245,247,251,0.78)", fontSize: 12 }}
          />
          <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
          <Radar
            name="Persona"
            dataKey="value"
            stroke="#d35cff"
            strokeWidth={2}
            fill="#a855f7"
            fillOpacity={0.5}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

function GenreIcons({ genres = [] }) {
  return (
    <div className="genre-icon-row">
      {genres.map((g) => {
        const icon = getGenreIcon(g);

        return (
          <div key={g} className="genre-icon">
            {icon && <img src={icon} alt={g} />}
            <span>{g}</span>
          </div>
        );
      })}
    </div>
  );
}

function ClusterUniverse({ clusters, selectedCluster, selectedUserCluster, onSelectCluster }) {
  const positioned = useMemo(() => {
    if (!clusters || clusters.length === 0) return [];
    // Place the user's cluster at center and up to 5 other clusters around in a "flower" layout
    const userCluster =
      clusters.find((c) => c.cluster === selectedUserCluster) || clusters[0];

    const others = clusters.filter((c) => c.cluster !== userCluster.cluster).slice(0, 5);

    const centerX = 50;
    const centerY = 53;
    const orbitRadius = 31;
    const centerSize = 200;
    const outerSize = 180;

    const result = [
      {
        ...userCluster,
        x: centerX,
        y: centerY,
        size: centerSize,
      },
    ];

    const flowerAngles = [-90, -20, 55, 125, 200];

    others.forEach((cluster, idx) => {
      const angle = (flowerAngles[idx] * Math.PI) / 180;

      result.push({
        ...cluster,
        x: centerX + Math.cos(angle) * orbitRadius,
        y: centerY + Math.sin(angle) * orbitRadius,
        size: outerSize,
      });
    });

    return result;
  }, [clusters, selectedUserCluster]);

  const detailCluster =
    selectedCluster ||
    clusters.find((c) => c.cluster === selectedUserCluster) ||
    null;

  return (
    <>

      <div className="cluster-layout">
        <div className="cluster-canvas panel-dark">
          {positioned.map((cluster) => {
            const isSelected = detailCluster?.cluster === cluster.cluster;
            const isUserCluster = selectedUserCluster === cluster.cluster;
            const bubbleIcon = getGenreIcon(cluster?.top_genres?.[0]);

            return (
              <button
                key={String(cluster.cluster)}
                className={`cluster-bubble ${isSelected ? "selected" : ""} ${isUserCluster ? "user-cluster" : ""}`}
                style={{
                  left: `${cluster.x}%`,
                  top: `${cluster.y}%`,
                  width: `${cluster.size}px`,
                  height: `${cluster.size}px`,
                }}
                onClick={() => onSelectCluster(cluster)}
                type="button"
              >
                <div className="cluster-bubble-inner">
                  <span className="cluster-bubble-title">
                    {cleanClusterName(cluster.persona_name)}
                  </span>
                </div>
              </button>
            );
          })}
        </div>

        <div className="panel cluster-detail-panel">
          <div className="story-label">
            {detailCluster?.cluster === selectedUserCluster ? "Your cluster" : "Selected cluster"}
          </div>

          <div className="cluster-detail-title">
            {cleanClusterName(detailCluster?.persona_name || "No cluster selected")}
          </div>

          <div className="story-label" style={{ marginBottom: "14px" }}>
            Shared genre DNA
          </div>
          <GenreIcons genres={detailCluster?.top_genres || []} />

          <div className="story-label" style={{ marginTop: "30px", marginBottom: "14px" }}>
            Movies from this cluster
          </div>
          <MovieShelf movies={detailCluster?.example_movies || []} />
        </div>
      </div>
    </>
  );
}

function EraTimeline({ userEras = [] }) {
  const decades = [1920, 1940, 1960, 1980, 2000, 2020];
  // Convert eras like "1980s" to integer values and remove invalid entries
  const parsedUserEras = (userEras || [])
    .map((e) => parseInt(String(e).replace("s", ""), 10))
    .filter((e) => !Number.isNaN(e));

  if (parsedUserEras.length === 0) {
    return <div className="small-note">Not enough era data</div>;
  }

  const highlighted = new Set(parsedUserEras);
  // Render timeline track and highlight user's preferred eras
  return (
    <div className="era-timeline-modern">
      <div className="era-track" />

      {decades.map((year) => {
        const active = highlighted.has(year);

        return (
          <div
            key={year}
            className={`era-node ${active ? "active" : ""}`}
            style={{ left: `${((year - 1920) / (2020 - 1920)) * 100}%` }}
          >
            <div className="era-node-dot" />
            <div className="era-node-label">{year}s</div>
          </div>
        );
      })}

      {parsedUserEras.map((year) => (
        <div
          key={`highlight-${year}`}
          className="era-highlight"
          style={{ left: `${((year - 1920) / (2020 - 1920)) * 100}%` }}
        >
          <div className="era-highlight-pill">{year}s</div>
        </div>
      ))}
    </div>
  );
}


function DashboardView({
  data,
  submittedUsername,
  usernameInput,
  setUsernameInput,
  handleSearch,
  loading,
}) {
  const user = data?.user;
  const matches = data?.matches || [];
  const technicalDetails = data?.technical_details;
  const clusters = data?.clusters || [];
  const [selectedCluster, setSelectedCluster] = useState(null);

  useEffect(() => {
    const nextInitial = clusters.find((c) => c.cluster === user.cluster) || null;
    setSelectedCluster(nextInitial);
  }, [clusters, user.cluster]);

  return (
    <>
      <div className="page-kicker">PlotTwins</div>
      <div className="page-title">Find your movie twin.</div>
      <div className="page-subtitle">
        Discover your cinema identity, the kind of stories you love, and the people nearby who watch like you.
      </div>

      <form className="search-form" onSubmit={handleSearch}>
        <input
          className="search-input"
          type="text"
          value={usernameInput}
          onChange={(e) => setUsernameInput(e.target.value)}
          placeholder="Enter any user id from 0 to 6518"
        />
        <button className="search-button" type="submit" disabled={loading}>
          {loading ? "Loading..." : "Search"}
        </button>
      </form>

      <div className="hero">
        <div className="hero-kicker">Your PlotTwins identity</div>
        <div className="hero-title">{data.hero_title}</div>
      </div>

      <div className="section-head">Cluster universe</div>
      <div className="small-note">Where you sit in the cinematic cosmos.</div>

      <ClusterUniverse
        clusters={clusters}
        selectedCluster={selectedCluster}
        selectedUserCluster={user.cluster}
        onSelectCluster={setSelectedCluster}
      />

      <div className="section-head" style={{ marginTop: "6px" }}>
        What your taste says about you
      </div>

      <div className="panel" style={{ marginBottom: "18px" }}>
        <div className="story-copy">{data.taste_story}</div>
      </div>

      <div className="story-grid">
        <div className="story-card">
          <div className="story-label">Your persona shape</div>
          <div className="story-title">This is your taste profile.</div>
          <PersonaRadar data={data.persona_radar || []} />
        </div>

        <div className="story-card">
          <div className="story-label">Your favorite eras</div>
          <div className="story-title">This is your comfort zone.</div>
          <EraTimeline userEras={user.user_top_eras} />
        </div>
      </div>

      <div className="section-head" style={{ marginTop: "34px" }}>
        Your people
      </div>

      <div className="small-note people-note">{data.people_story}</div>

      {matches.length === 0 ? (
        <div className="message info">No matches found for this user.</div>
      ) : (
        <div className="matches-grid people-row">
          {matches.map((match, index) => (
            <MatchCard
              key={`${submittedUsername}-${match.match_user}`}
              match={match}
              avatarIndex={index}
            />
          ))}
        </div>
      )}

      <TechnicalDetails data={technicalDetails} />
    </>
  );
}


function LandingView({
  usernameInput,
  setUsernameInput,
  handleSearch,
  loading,
  error,
}) {
  return (
    <div className="landing-view">
      <div className="page-kicker">PlotTwins</div>

      <div className="page-title">Find your movie twin.</div>

      <div className="page-subtitle">
        Discover your cinema identity, the kind of stories you love,
        and the people nearby who watch like you.
      </div>

      <form className="search-form search-form-hero" onSubmit={handleSearch}>
        <input
          className="search-input"
          type="text"
          value={usernameInput}
          onChange={(e) => setUsernameInput(e.target.value)}
          placeholder="Search by username or user id"
        />
        <button className="search-button" type="submit" disabled={loading}>
          {loading ? "Loading..." : "Search"}
        </button>
      </form>

      {error && <div className="message warning">{error}</div>}
    </div>
  );
}

export default function App() {
  const [usernameInput, setUsernameInput] = useState("");
  const [submittedUsername, setSubmittedUsername] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState("landing");

  async function handleSearch(e) {
    e.preventDefault();

    const trimmed = usernameInput.trim();
    if (!trimmed) return;

    setLoading(true);
    setError("");

    try {
      const result = await fetchUserDashboard(trimmed);
      setData(result);
      setSubmittedUsername(trimmed);
      setView("dashboard");
    } catch (err) {
      setError(err.message || "Failed to fetch user.");
      setData(null);
      setView("landing");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <main className="container">
        {view === "landing" && (
          <LandingView
            usernameInput={usernameInput}
            setUsernameInput={setUsernameInput}
            handleSearch={handleSearch}
            loading={loading}
            error={error}
          />
        )}

        {view === "dashboard" && data && (
          <DashboardView
            data={data}
            submittedUsername={submittedUsername}
            usernameInput={usernameInput}
            setUsernameInput={setUsernameInput}
            handleSearch={handleSearch}
            loading={loading}
          />
        )}
      </main>
    </div>
  );
}
