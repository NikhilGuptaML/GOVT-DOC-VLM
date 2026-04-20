import { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const styles = {
  card: {
    background: '#141414',
    border: '1px solid #222',
    borderRadius: '12px',
    marginBottom: '32px',
    overflow: 'hidden',
  },
  cardHeader: {
    padding: '12px 20px',
    borderBottom: '1px solid #222',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '10px',
    background: '#111',
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  pageLabel: {
    fontSize: '12px',
    fontWeight: 700,
    color: '#7c3aed',
    letterSpacing: '0.5px',
    textTransform: 'uppercase',
  },
  body: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
    minHeight: '400px',
  },
  imagePanel: {
    borderRight: '1px solid #222',
    padding: '20px',
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'center',
    background: '#0a0a0a',
  },
  img: {
    width: '100%',
    maxWidth: '100%',
    borderRadius: '4px',
    border: '1px solid #2a2a2a',
  },
  textPanel: {
    padding: '20px 24px',
    overflowY: 'auto',
    maxHeight: '700px',
  },
  panelLabel: {
    fontSize: '11px',
    color: '#555',
    fontWeight: 600,
    letterSpacing: '0.5px',
    textTransform: 'uppercase',
    marginBottom: '14px',
  },
  statusPill: {
    fontSize: '10px',
    fontWeight: 700,
    letterSpacing: '0.4px',
    textTransform: 'uppercase',
    padding: '3px 8px',
    borderRadius: '999px',
  },
  statusCompleted: {
    background: '#11331d',
    color: '#87e7a3',
    border: '1px solid #275436',
  },
  statusError: {
    background: '#3f1010',
    color: '#ff9999',
    border: '1px solid #7f2020',
  },
  actionButton: {
    background: '#1f1f2a',
    color: '#b3b3ff',
    border: '1px solid #35354e',
    padding: '6px 10px',
    borderRadius: '8px',
    fontSize: '12px',
    cursor: 'pointer',
  },
  actionButtonDisabled: {
    background: '#151515',
    color: '#666',
    border: '1px solid #2a2a2a',
    cursor: 'not-allowed',
  },
  errorText: {
    marginTop: '10px',
    color: '#ff9d9d',
    fontSize: '12px',
    border: '1px solid #7f2020',
    borderRadius: '8px',
    background: '#2f0d0d',
    padding: '8px 10px',
  },
  modalBackdrop: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0, 0, 0, 0.72)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2000,
    padding: '20px',
  },
  modalCard: {
    width: 'min(960px, 100%)',
    maxHeight: '80vh',
    background: '#0f0f16',
    border: '1px solid #2a2a3d',
    borderRadius: '12px',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '14px 16px',
    borderBottom: '1px solid #222',
    background: '#0a0a12',
    color: '#d9d9ff',
    fontSize: '14px',
    fontWeight: 600,
  },
  modalCloseBtn: {
    background: 'transparent',
    border: '1px solid #2f2f42',
    color: '#ddd',
    borderRadius: '8px',
    padding: '5px 10px',
    cursor: 'pointer',
  },
  modalBody: {
    overflowY: 'auto',
    padding: '16px',
  },
  reasoningText: {
    whiteSpace: 'pre-wrap',
    fontSize: '12px',
    lineHeight: 1.6,
    color: '#d0d0dd',
    fontFamily: 'JetBrains Mono, Fira Code, Consolas, monospace',
  },
}

// Markdown styles injected via className would need CSS file
// Using inline style overrides via wrapper
const mdWrapper = {
  fontSize: '13px',
  lineHeight: '1.7',
  color: '#ccc',
}

export default function PageViewer({ page }) {
  const [showReasoning, setShowReasoning] = useState(false)

  const hasReasoning = useMemo(
    () => Boolean(page.reasoning_text && page.reasoning_text.trim()),
    [page.reasoning_text],
  )

  const statusStyle = page.status === 'error'
    ? { ...styles.statusPill, ...styles.statusError }
    : { ...styles.statusPill, ...styles.statusCompleted }

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <div style={styles.headerLeft}>
          <span style={styles.pageLabel}>Page {page.page_number}</span>
          <span style={statusStyle}>{page.status || 'completed'}</span>
        </div>
        <button
          type="button"
          style={{
            ...styles.actionButton,
            ...(hasReasoning ? {} : styles.actionButtonDisabled),
          }}
          disabled={!hasReasoning}
          onClick={() => setShowReasoning(true)}
        >
          View Reasoning
        </button>
      </div>

      <div style={styles.body}>
        {/* Left: Scanned image */}
        <div style={styles.imagePanel}>
          <img
            src={page.image_url}
            alt={`Page ${page.page_number}`}
            style={styles.img}
          />
        </div>

        {/* Right: Extracted text */}
        <div style={styles.textPanel}>
          <p style={styles.panelLabel}>Extracted Content</p>
          <div style={mdWrapper} className="md-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {page.extracted_text}
            </ReactMarkdown>
          </div>
          {page.error_message && (
            <p style={styles.errorText}>Error: {page.error_message}</p>
          )}
        </div>
      </div>

      {showReasoning && (
        <div style={styles.modalBackdrop} onClick={() => setShowReasoning(false)}>
          <div style={styles.modalCard} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <span>Model Reasoning · Page {page.page_number}</span>
              <button type="button" style={styles.modalCloseBtn} onClick={() => setShowReasoning(false)}>
                Close
              </button>
            </div>
            <div style={styles.modalBody}>
              <pre style={styles.reasoningText}>
                {page.reasoning_text || 'No reasoning content was provided for this page.'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
