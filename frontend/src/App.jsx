import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { createSession, uploadPdf, chat, deleteSession } from "./api";
import "./App.css";

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    createSession()
      .then((d) => setSessionId(d.session_id))
      .catch(() => setError("Could not reach the server. Is the API running on :8000?"));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  async function handleFiles(files) {
    if (!sessionId || !files?.length) return;
    setError(""); setUploading(true);
    try {
      for (const file of files) {
        const res = await uploadPdf(sessionId, file);
        setDocs(res.documents);
      }
    } catch (e) { setError(e.message); } finally { setUploading(false); }
  }

  async function handleAsk() {
    const q = input.trim();
    if (!q || thinking) return;
    if (!docs.length) { setError("Upload a PDF first."); return; }
    setError(""); setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setThinking(true);
    try {
      const res = await chat(sessionId, q);
      setMessages((m) => [...m, { role: "assistant", text: res.answer, citations: res.citations }]);
    } catch (e) { setError(e.message); } finally { setThinking(false); }
  }

  function clearChat() {
    setMessages([]);
    setError("");
  }

  async function newSession() {
    if (sessionId) deleteSession(sessionId).catch(() => {});  // best-effort teardown
    setMessages([]);
    setDocs([]);
    setError("");
    setInput("");
    const d = await createSession();
    setSessionId(d.session_id);
  }

  return (
    <div className="app">
      <div className="wrap">
        <motion.header className="masthead"
          initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <h1>Ask My Docs</h1>
          <p>Upload PDFs and ask questions — every answer is grounded in your sources, with citations.</p>
        </motion.header>

        <div className="layout">
          <motion.aside className="sidebar"
            initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.1 }}>
            <p className="label">Documents</p>
            <div
              className={`dropzone ${uploading ? "busy" : ""}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); handleFiles(Array.from(e.dataTransfer.files)); }}
            >
              <input ref={fileRef} type="file" accept="application/pdf" multiple hidden
                onChange={(e) => handleFiles(Array.from(e.target.files))} />
              <span className="icon">⊕</span>
              {uploading ? "Indexing…" : "Drop a PDF here, or click to browse"}
            </div>

            <ul className="doclist">
              {docs.length === 0 && <li className="empty">No documents yet</li>}
              <AnimatePresence>
                {docs.map((d, i) => (
                  <motion.li key={i} className="doc"
                    initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
                    <span className="dot" /><span className="name">{d}</span>
                  </motion.li>
                ))}
              </AnimatePresence>
            </ul>
          </motion.aside>

          <motion.section className="chat"
            initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.15 }}>
            <div className="chat-bar">
              <span className="chat-title">Conversation</span>
              <div className="chat-actions">
                <button className="ghost" onClick={clearChat} disabled={!messages.length}>
                  Clear
                </button>
                <button className="ghost" onClick={newSession}>
                  New
                </button>
              </div>
            </div>
            <div className="messages">
              {messages.length === 0 && (
                <div className="placeholder">
                  {docs.length ? "Ask anything about your documents." : "Upload a document to begin."}
                </div>
              )}
              <AnimatePresence>
                {messages.map((m, i) => (
                  <motion.div key={i} className={`msg ${m.role}`}
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
                    <div className="bubble">{m.text}</div>
                    {m.citations?.length > 0 && (
                      <div className="citations">
                        {m.citations.map((c) => (
                          <div key={c.n} className="cite"><span className="cite-n">[{c.n}]</span>{c.source}</div>
                        ))}
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
              {thinking && (
                <div className="msg assistant">
                  <div className="bubble thinking">
                    Thinking<span className="dots"><span>.</span><span>.</span><span>.</span></span>
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>

            {error && <div className="error">{error}</div>}

            <div className="composer">
              <input value={input} placeholder="Ask a question…" disabled={!sessionId}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAsk()} />
              <button onClick={handleAsk} disabled={thinking || !input.trim()}>Ask</button>
            </div>
          </motion.section>
        </div>
      </div>
    </div>
  );
}