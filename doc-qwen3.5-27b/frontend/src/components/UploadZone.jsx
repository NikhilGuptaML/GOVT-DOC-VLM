import { useRef, useState } from 'react'

const styles = {
  zone: {
    border: '2px dashed #333',
    borderRadius: '12px',
    padding: '48px',
    textAlign: 'center',
    marginBottom: '32px',
    cursor: 'pointer',
    transition: 'border-color 0.2s, background 0.2s',
    background: '#141414',
  },
  zoneHover: {
    borderColor: '#7c3aed',
    background: '#1a1025',
  },
  label: {
    fontSize: '15px',
    color: '#888',
    marginBottom: '8px',
  },
  sub: {
    fontSize: '12px',
    color: '#555',
  },
  btn: {
    marginTop: '16px',
    padding: '10px 28px',
    background: '#7c3aed',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  btnDisabled: {
    background: '#333',
    color: '#666',
    cursor: 'not-allowed',
  },
}

export default function UploadZone({ onUpload, loading }) {
  const inputRef = useRef()
  const [hover, setHover] = useState(false)
  const [fileName, setFileName] = useState(null)

  function handleFile(file) {
    if (!file || !file.name.endsWith('.pdf')) return
    setFileName(file.name)
    onUpload(file)
  }

  function handleDrop(e) {
    e.preventDefault()
    setHover(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div
      style={{ ...styles.zone, ...(hover ? styles.zoneHover : {}) }}
      onDragOver={(e) => { e.preventDefault(); setHover(true) }}
      onDragLeave={() => setHover(false)}
      onDrop={handleDrop}
      onClick={() => !loading && inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
      <p style={styles.label}>
        {loading
          ? `⏳ Processing ${fileName || 'document'}...`
          : fileName
          ? `✓ ${fileName}`
          : 'Drop a PDF here or click to upload'}
      </p>
      <p style={styles.sub}>Scanned government documents · Mixed Hindi/English · Tables supported</p>
      {!loading && (
        <button
          style={styles.btn}
          onClick={(e) => { e.stopPropagation(); inputRef.current.click() }}
        >
          Select PDF
        </button>
      )}
    </div>
  )
}
