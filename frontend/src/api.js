const API_BASE = "http://127.0.0.1:8000";

export async function fetchUserDashboard(username) {
  const response = await fetch(`${API_BASE}/api/user/${encodeURIComponent(username)}`);

  if (!response.ok) {
    let detail = "Something went wrong.";
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {}
    throw new Error(detail);
  }

  return response.json();
}

export async function fetchBusinessEvent(fsa) {
  const response = await fetch(`${API_BASE}/api/events/${encodeURIComponent(fsa)}`);

  if (!response.ok) {
    let detail = "Something went wrong.";
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {}
    throw new Error(detail);
  }

  return response.json();
}