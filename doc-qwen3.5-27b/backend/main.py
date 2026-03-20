import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from config import USE_MOCK
from pdf_processor import pdf_to_images

# ── Client switch ──────────────────────────────────────────
if USE_MOCK:
    from mock_client import process_image
    print("🟡 Running with MOCK client — no model loaded")
else:
    from model_client import process_image
    print("🟢 Running with REAL model client")
# ──────────────────────────────────────────────────────────

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
    return {"status": "ok", "mock": USE_MOCK}


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    Main endpoint:
    1. Save uploaded PDF
    2. Convert PDF → images page by page
    3. Send each image to model (or mock)
    4. Return all pages with image URLs + extracted text
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Create unique session folder
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save uploaded PDF
    pdf_path = os.path.join(session_dir, "document.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Convert PDF to images
    try:
        pages = pdf_to_images(pdf_path, session_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")

    # Process each page through model
    results = []
    for page in pages:
        try:
            extracted_text = process_image(page["image_path"], page["page_number"])
        except Exception as e:
            extracted_text = f"[ERROR on page {page['page_number']}]: {str(e)}"

        results.append({
            "page_number": page["page_number"],
            "image_url": f"/images/{session_id}/{page['image_filename']}",
            "extracted_text": extracted_text,
        })

    return JSONResponse({
        "session_id": session_id,
        "total_pages": len(results),
        "pages": results,
    })
