import os
import uuid
import shutil
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse

from config import active_runtime_mode, runtime_mode_reason
from pdf_processor import pdf_to_images
from mock_client import process_image as process_image_mock
from model_client import process_image as process_image_hf


def _get_processor():
    return process_image_hf if active_runtime_mode() == "hf" else process_image_mock


def _event_line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def _append_jsonl(file_path: str, payload: dict) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _save_upload(file: UploadFile) -> tuple[str, str, str]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    pdf_path = os.path.join(session_dir, "document.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return session_id, session_dir, pdf_path


def _build_page_payload(session_id: str, page: dict, result: dict) -> dict:
    error_message = result.get("error")
    status = "error" if error_message else "completed"
    return {
        "page_number": page["page_number"],
        "image_url": f"/images/{session_id}/{page['image_filename']}",
        "extracted_text": result.get("extracted_text", ""),
        "reasoning_text": result.get("reasoning_text", ""),
        "status": status,
        "error_message": error_message,
    }


def _process_pages(session_id: str, session_dir: str, pages: list[dict], processor):
    results_path = os.path.join(session_dir, "results.jsonl")
    total_pages = len(pages)

    for idx, page in enumerate(pages, start=1):
        page_number = page["page_number"]
        yield {
            "event": "page_started",
            "session_id": session_id,
            "page_number": page_number,
            "processed_pages": idx - 1,
            "total_pages": total_pages,
        }

        try:
            result = processor(page["image_path"], page_number)
        except Exception as exc:
            result = {
                "extracted_text": f"[Page {page_number} ERROR]: {str(exc)}",
                "reasoning_text": "",
                "error": str(exc),
            }

        page_payload = _build_page_payload(session_id, page, result)
        event_name = "page_error" if page_payload["status"] == "error" else "page_completed"
        page_event = {
            "event": event_name,
            "session_id": session_id,
            "processed_pages": idx,
            "total_pages": total_pages,
            "page": page_payload,
        }
        _append_jsonl(results_path, page_event)
        yield page_event


if active_runtime_mode() == "hf":
    print("🟢 Running with Hugging Face model client")
else:
    print("🟡 Running with MOCK client")
print(f"   Mode reason: {runtime_mode_reason()}")

app = FastAPI(title="Govt Doc VLM API")

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve page images as static files
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=UPLOADS_DIR), name="images")


@app.get("/")
def health():
    mode = active_runtime_mode()
    return {
        "status": "ok",
        "mode": mode,
        "mock": mode == "mock",
        "reason": runtime_mode_reason(),
    }


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    Main endpoint:
    1. Save uploaded PDF
    2. Convert PDF → images page by page
    3. Send each image to model (or mock)
    4. Return all pages with image URLs + extracted text
    """

    session_id, session_dir, pdf_path = _save_upload(file)

    try:
        pages = pdf_to_images(pdf_path, session_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(exc)}")

    processor = _get_processor()
    results = []
    for event in _process_pages(session_id, session_dir, pages, processor):
        page = event.get("page")
        if page:
            results.append(page)

    return JSONResponse({
        "session_id": session_id,
        "mode": active_runtime_mode(),
        "total_pages": len(results),
        "pages": results,
    })


@app.post("/process/stream")
async def process_pdf_stream(file: UploadFile = File(...)):
    """
    Streaming endpoint for live per-page updates.
    Returns newline-delimited JSON events.
    """
    session_id, session_dir, pdf_path = _save_upload(file)

    def event_stream():
        mode = active_runtime_mode()
        processor = _get_processor()

        yield _event_line({
            "event": "started",
            "session_id": session_id,
            "mode": mode,
            "message": "Upload received; starting PDF conversion",
        })

        try:
            pages = pdf_to_images(pdf_path, session_dir)
        except Exception as exc:
            yield _event_line({
                "event": "fatal_error",
                "session_id": session_id,
                "error_message": f"PDF conversion failed: {str(exc)}",
            })
            return

        yield _event_line({
            "event": "pdf_converted",
            "session_id": session_id,
            "total_pages": len(pages),
        })

        completed_pages = []
        try:
            for page_event in _process_pages(session_id, session_dir, pages, processor):
                page_payload = page_event.get("page")
                if page_payload:
                    completed_pages.append(page_payload)
                yield _event_line(page_event)
        except Exception as exc:
            yield _event_line({
                "event": "fatal_error",
                "session_id": session_id,
                "error_message": f"Streaming pipeline failed: {str(exc)}",
                "processed_pages": len(completed_pages),
                "total_pages": len(pages),
            })
            return

        final_payload = {
            "event": "finished",
            "session_id": session_id,
            "mode": mode,
            "processed_pages": len(completed_pages),
            "total_pages": len(pages),
            "message": "All pages processed",
        }
        _append_jsonl(os.path.join(session_dir, "results.jsonl"), final_payload)
        yield _event_line(final_payload)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
