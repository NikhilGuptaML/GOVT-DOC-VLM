from PIL import Image, ImageDraw, ImageFont
import os
from config import PDF_DPI, MAX_IMAGE_WIDTH, USE_MOCK


def pdf_to_images(pdf_path: str, output_dir: str) -> list[dict]:
    """
    Convert each page of a PDF to a PNG image.
    Returns list of dicts: { page_number, image_path, image_filename }
    """
    os.makedirs(output_dir, exist_ok=True)

    # Import PyMuPDF lazily so the server can start even if the dependency
    # isn't installed (e.g., UI mock mode on a workstation).
    try:
        import pymupdf as fitz  # PyMuPDF 1.24+ uses pymupdf instead of fitz  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            import fitz  # Older PyMuPDF exposes `fitz`  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            fitz = None

    # Mock fallback: avoid hard dependency on PDF conversion.
    # This keeps the `/process` endpoint usable for UI flow testing.
    if fitz is None:
        if not USE_MOCK:
            raise ModuleNotFoundError(
                "PyMuPDF is required for PDF conversion. Install `pymupdf` "
                "(or `PyMuPDF`) to use REAL inference mode."
            )

        # If PyMuPDF isn't available, we still want the UI to show *all pages*.
        # We can only create placeholders, but we can get the page count using
        # a lightweight pure-Python PDF reader (PyPDF2).
        page_count = 1
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages) or 1
        except Exception:
            page_count = 1

        # Try to load a bigger font so placeholders are visible in the UI.
        big_font = None
        small_font = None
        for fp in [
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]:
            if not os.path.exists(fp):
                continue
            try:
                big_font = ImageFont.truetype(fp, 64)
                small_font = ImageFont.truetype(fp, 28)
                break
            except Exception:
                continue

        if big_font is None:
            big_font = ImageFont.load_default()
        if small_font is None:
            small_font = ImageFont.load_default()

        pages: list[dict] = []
        for page_num in range(1, page_count + 1):
            filename = f"page_{page_num}.png"
            placeholder_path = os.path.join(output_dir, filename)
            img = Image.new("RGB", (1024, 1024), color=(25, 25, 25))
            draw = ImageDraw.Draw(img)
            # Make it obvious in the UI that this is a mock placeholder.
            draw.rectangle([40, 40, 984, 984], outline=(255, 0, 0), width=18)
            draw.text((80, 120), f"MOCK PAGE {page_num}", fill=(255, 255, 255), font=big_font)
            draw.text((80, 260), "PyMuPDF not installed", fill=(255, 255, 0), font=small_font)
            img.save(placeholder_path, "PNG")
            pages.append({
                "page_number": page_num,
                "image_path": placeholder_path,
                "image_filename": filename,
            })
        return pages

    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        # Render page at specified DPI
        mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
        pix = page.get_pixmap(matrix=mat)

        # Save as PNG temporarily
        raw_path = os.path.join(output_dir, f"page_{i + 1}_raw.png")
        pix.save(raw_path)

        # Resize if too wide (saves VRAM during inference)
        img = Image.open(raw_path)
        if img.width > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / img.width
            new_height = int(img.height * ratio)
            img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.LANCZOS)

        # Save final image
        filename = f"page_{i + 1}.png"
        final_path = os.path.join(output_dir, filename)
        img.save(final_path, "PNG")

        # Remove raw temp file
        os.remove(raw_path)

        pages.append({
            "page_number": i + 1,
            "image_path": final_path,
            "image_filename": filename,
        })

    doc.close()
    return pages
