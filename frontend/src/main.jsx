import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const readableError = (error) => error.message === 'Failed to fetch'
  ? `Cannot reach the API at ${API}. Start FastAPI on port 8000 and check the browser URL.`
  : error.message;

function App() {
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const loadDocuments = async () => {
    const response = await fetch(`${API}/documents`);
    if (!response.ok) throw new Error('Could not load documents');
    setDocuments((await response.json()).documents);
  };
  useEffect(() => { loadDocuments().catch((e) => setError(readableError(e))); }, []);

  const upload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true); setError('');
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch(`${API}/ingest`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Upload failed');
      await loadDocuments();
    } catch (e) { setError(readableError(e)); } finally { setBusy(false); event.target.value = ''; }
  };

  const ask = async (event) => {
    event.preventDefault(); if (!question.trim()) return;
    setBusy(true); setError(''); setAnswer(null);
    try {
      const response = await fetch(`${API}/query`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question, top_k: 5}) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Query failed');
      setAnswer(data);
    } catch (e) { setError(readableError(e)); } finally { setBusy(false); }
  };

  return <main className="shell">
    <header><div className="mark">D</div><div><h1>DocMind</h1><p>Private document intelligence</p></div></header>
    <section className="hero"><h2>Ask your documents.</h2><p>Upload a document, then ask a grounded question with citations.</p></section>
    <section className="grid">
      <aside className="card"><h3>Documents</h3><label className="upload">{busy ? 'Processing…' : '+ Upload file'}<input type="file" accept=".txt,.md,.markdown,.pdf" onChange={upload} disabled={busy}/></label>{documents.length === 0 ? <p className="muted">No documents yet.</p> : <ul>{documents.map((doc) => <li key={doc.source}><strong>{doc.filename}</strong><span>{doc.chunks} chunks · {doc.modality}</span></li>)}</ul>}</aside>
      <section className="card answer"><h3>Question</h3><form onSubmit={ask}><textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What would you like to know?" rows="4"/><button disabled={busy || !question.trim()}>{busy ? 'Working…' : 'Ask DocMind'}</button></form>{error && <div className="error">{error}</div>}{answer && <div className="result"><h3>Answer</h3><p>{answer.answer}</p>{answer.citations?.length > 0 && <><h4>Citations</h4><div className="citations">{answer.citations.map((citation) => <span key={citation}>[{citation}]</span>)}</div></>}</div>}</section>
    </section>
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
