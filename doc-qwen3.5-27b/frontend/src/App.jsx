import { useState } from 'react'
import UploadZone from './components/UploadZone'
import PageViewer from './components/PageViewer'

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
    marginBottom: '24px',
    fontSize: '13px',
    fontWeight: 500,
  },
  error: {
    background: '#3f1010',
    border: '1px solid #7f2020',
    color: '#ff8080',
  },
}

export default function App() {
  const [pages, setPages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [totalPages, setTotalPages] = useState(0)

  async function handleUpload(file) {
    setLoading(true)
    setError(null)
    setPages([])

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/process', { method: 'POST', body: formData })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Processing failed')
      }
      const data = await res.json()
      setPages(data.pages)
      setTotalPages(data.total_pages)
    } catch (e) {
      setError(e.message)
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
        <UploadZone onUpload={handleUpload} loading={loading} />

        {error && (
          <div style={{ ...styles.statusBar, ...styles.error }}>
            ⚠ {error}
          </div>
        )}

        {loading && (
          <div style={{ textAlign: 'center', padding: '48px', color: '#888' }}>
            <p style={{ fontSize: '15px' }}>Processing PDF...</p>
            <p style={{ fontSize: '12px', marginTop: '8px', color: '#555' }}>
              Each page is sent to the model individually
            </p>
          </div>
        )}

        {pages.length > 0 && (
          <>
            <p style={{ color: '#666', fontSize: '13px', marginBottom: '24px' }}>
              {totalPages} page{totalPages !== 1 ? 's' : ''} extracted
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
