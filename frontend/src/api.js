// All backend calls live here, separated from UI. Change API_BASE for deploy.
const API_BASE = "http://localhost:8000";

export async function createSession() {
  const res = await fetch(`${API_BASE}/session`, { method: "POST" });
  if (!res.ok) throw new Error("Could not create session");
  return res.json(); // { session_id }
}

export async function uploadPdf(sessionId, file) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json(); // { filename, chunks_indexed, documents }
}

export async function chat(sessionId, question) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat failed");
  }
  return res.json(); // { answer, citations: [{n, source}] }
}

export async function deleteSession(sessionId) {
  await fetch(`${API_BASE}/session/${sessionId}`, { method: "DELETE" });
}