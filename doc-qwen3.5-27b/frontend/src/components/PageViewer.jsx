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
    gap: '10px',
    background: '#111',
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
    gridTemplateColumns: '1fr 1fr',
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
}

// Markdown styles injected via className would need CSS file
// Using inline style overrides via wrapper
const mdWrapper = {
  fontSize: '13px',
  lineHeight: '1.7',
  color: '#ccc',
}

export default function PageViewer({ page }) {
  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.pageLabel}>Page {page.page_number}</span>
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
        </div>
      </div>
    </div>
  )
}
