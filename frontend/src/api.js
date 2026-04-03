const API_BASE = import.meta.env.VITE_API_BASE;

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