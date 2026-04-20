# Govt Doc VLM — Qwen 3.5 Streaming Extractor

Upload scanned PDFs, process each page independently, and stream results live to the UI.

The backend now uses Hugging Face Inference API (`Qwen/Qwen3.5-27B`) when `HF_TOKEN` is available. If token is missing, it automatically falls back to mock mode.

## What Changed

- Inference backend migrated from local GGUF runtime to Hugging Face API.
- New streaming endpoint: `POST /process/stream` returns page-by-page NDJSON events.
- Frontend renders pages as soon as each page completes.
- Per-page failures do not stop the whole document pipeline.
- Each page includes optional raw reasoning text for popup view.

## Runtime Mode Selection

Runtime mode is automatic:

- `FORCE_MOCK=true` -> always mock
- `USE_MOCK=true` -> mock
- `HF_TOKEN` present and mock flags false -> Hugging Face
- otherwise -> mock

Check mode via `GET /` health endpoint.

## Quick Start

### 1. Backend

```bash
cd doc-qwen3.5-27b/backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

Create env file:

```bash
cd ..
cp .env.example .env
```

Set `HF_TOKEN` in `.env` for real inference.

Start backend:

```bash
cd backend
python -m uvicorn main:app --port 8001 --reload
```

### 2. Frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API Endpoints

### `GET /`

Health endpoint with active runtime mode.

### `POST /process`

Legacy non-streaming endpoint; returns full result when all pages complete.

### `POST /process/stream`

Streaming endpoint used by frontend. Returns NDJSON events:

- `started`
- `pdf_converted`
- `page_started`
- `page_completed`
- `page_error`
- `finished`
- `fatal_error`

Each page event includes:

- `page_number`
- `image_url`
- `extracted_text`
- `reasoning_text`
- `status`
- `error_message`

## Folder Notes

- `doc-qwen3.5-27b/backend/main.py` - streaming and non-streaming API routes.
- `doc-qwen3.5-27b/backend/model_client.py` - HF inference calls and reasoning parsing.
- `doc-qwen3.5-27b/backend/mock_client.py` - local mock fallback behavior.
- `doc-qwen3.5-27b/frontend/src/App.jsx` - stream parsing and live page updates.
- `doc-qwen3.5-27b/frontend/src/components/PageViewer.jsx` - page card + reasoning popup.

## Troubleshooting

- If all pages are mock output, verify `HF_TOKEN` in `.env` and restart backend.
- If one page fails due to token/context/provider errors, later pages still continue by design.
- If stream does not update in UI, verify frontend proxy points to backend and `POST /process/stream` returns NDJSON.
