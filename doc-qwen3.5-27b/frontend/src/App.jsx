import { useState } from 'react'
import UploadZone from './components/UploadZone'
import PageViewer from './components/PageViewer'


function upsertPage(prevPages, incomingPage) {
  const existingIdx = prevPages.findIndex((page) => page.page_number === incomingPage.page_number)
  if (existingIdx === -1) {
    return [...prevPages, incomingPage].sort((a, b) => a.page_number - b.page_number)
  }

  const updated = [...prevPages]
  updated[existingIdx] = { ...updated[existingIdx], ...incomingPage }
  return updated.sort((a, b) => a.page_number - b.page_number)
}


function parseErrorPayload(raw) {
  if (!raw) return null
  if (raw.detail) return raw.detail
  if (raw.error_message) return raw.error_message
  return null
}

const styles = {
  app: {
    minHeight: '100vh',
    background: '#0f0f0f',
    color: '#e8e8e8',
  },
  header: {
    padding: '20px 32px',
    borderBottom: '1px solid #222',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  badge: {
    background: '#7c3aed',
    color: '#fff',
    fontSize: '11px',
    fontWeight: 700,
    padding: '3px 10px',
    borderRadius: '20px',
    letterSpacing: '0.5px',
  },
  title: {
    fontSize: '18px',
    fontWeight: 600,
    color: '#fff',
  },
  body: {
    padding: '32px',
    maxWidth: '1400px',
    margin: '0 auto',
  },
  statusBar: {
    padding: '10px 16px',
    borderRadius: '8px',
    marginBottom: '14px',
    fontSize: '13px',
    fontWeight: 500,
  },
  info: {
    background: '#10213f',
    border: '1px solid #1f3f7f',
    color: '#8cb7ff',
  },
  error: {
    background: '#3f1010',
    border: '1px solid #7f2020',
    color: '#ff8080',
  },
  progressMeta: {
    color: '#777',
    fontSize: '12px',
    marginBottom: '20px',
  },
}

export default function App() {
  const [pages, setPages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [totalPages, setTotalPages] = useState(0)
  const [processedPages, setProcessedPages] = useState(0)
  const [sessionId, setSessionId] = useState(null)
  const [streamStatus, setStreamStatus] = useState('')
  const [mode, setMode] = useState('mock')

  function applyStreamEvent(event) {
    if (!event || !event.event) return

    if (typeof event.total_pages === 'number') {
      setTotalPages(event.total_pages)
    }
    if (typeof event.processed_pages === 'number') {
      setProcessedPages(event.processed_pages)
    }

    switch (event.event) {
      case 'started':
        setSessionId(event.session_id || null)
        setMode(event.mode || 'mock')
        setStreamStatus(event.message || 'PDF upload received')
        break
      case 'pdf_converted':
        setStreamStatus(`PDF converted into ${event.total_pages || 0} page(s)`)
        break
      case 'page_started':
        setStreamStatus(`Processing page ${event.page_number}...`)
        break
      case 'page_completed':
        if (event.page) {
          setPages((prev) => upsertPage(prev, event.page))
          setStreamStatus(`Completed page ${event.page.page_number}`)
        }
        break
      case 'page_error':
        if (event.page) {
          setPages((prev) => upsertPage(prev, event.page))
          setStreamStatus(`Page ${event.page.page_number} failed; continuing`) 
        }
        break
      case 'finished':
        setStreamStatus('All pages processed')
        break
      case 'fatal_error':
        setError(event.error_message || 'Processing failed')
        break
      default:
        break
    }
  }

  async function handleUpload(file) {
    setLoading(true)
    setError(null)
    setPages([])
    setTotalPages(0)
    setProcessedPages(0)
    setSessionId(null)
    setStreamStatus('Uploading PDF...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/process/stream', { method: 'POST', body: formData })
      if (!res.ok) {
        let errText = 'Processing failed'
        try {
          const err = await res.json()
          errText = parseErrorPayload(err) || errText
        } catch {
          // Ignore parse failures and use default message.
        }
        throw new Error(errText)
      }

      if (!res.body) {
        throw new Error('Streaming is not supported in this browser')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        let newlineIdx = buffer.indexOf('\n')
        while (newlineIdx !== -1) {
          const rawLine = buffer.slice(0, newlineIdx).trim()
          buffer = buffer.slice(newlineIdx + 1)

          if (rawLine) {
            try {
              applyStreamEvent(JSON.parse(rawLine))
            } catch {
              // Skip malformed stream lines and continue parsing.
            }
          }

          newlineIdx = buffer.indexOf('\n')
        }
      }

      if (buffer.trim()) {
        try {
          applyStreamEvent(JSON.parse(buffer.trim()))
        } catch {
          // Ignore trailing malformed payload.
        }
      }
    } catch (e) {
      setError(e.message)
      setStreamStatus('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>Govt Doc Extractor</h1>
        <span style={styles.badge}>Qwen 3.5 · 27B</span>
      </header>

      <div style={styles.body}>
        <UploadZone onUpload={handleUpload} loading={loading} statusText={streamStatus} />

        {(loading || streamStatus) && (
          <div style={{ ...styles.statusBar, ...styles.info }}>
            {streamStatus || 'Processing...'}
          </div>
        )}

        {error && (
          <div style={{ ...styles.statusBar, ...styles.error }}>
            ⚠ {error}
          </div>
        )}

        {(loading || pages.length > 0) && (
          <p style={styles.progressMeta}>
            {processedPages}/{totalPages || '?'} pages processed
            {sessionId ? ` · session ${sessionId.slice(0, 8)}` : ''}
            {mode ? ` · mode ${mode}` : ''}
          </p>
        )}

        {pages.length > 0 && (
          <>
            <p style={{ color: '#666', fontSize: '13px', marginBottom: '24px' }}>
              Showing {pages.length} page{pages.length !== 1 ? 's' : ''} so far
            </p>
            {pages.map((page) => (
              <PageViewer key={page.page_number} page={page} />
            ))}
          </>
        )}
      </div>
    </div>
  )
}
