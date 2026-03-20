import pymupdf  # PyMuPDF 1.24+ — install with: pip install pymupdf
from PIL import Image
import os
from config import PDF_DPI, MAX_IMAGE_WIDTH


def pdf_to_images(pdf_path: str, output_dir: str) -> list[dict]:
    """
    Convert each page of a PDF to a PNG image.
    Always runs with real pymupdf — USE_MOCK only affects AI model calls, not this.
    Returns list of dicts: { page_number, image_path, image_filename }
    """
    os.makedirs(output_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        # Render page at specified DPI (200 DPI = good quality for govt docs)
        mat = pymupdf.Matrix(PDF_DPI / 72, PDF_DPI / 72)
        pix = page.get_pixmap(matrix=mat)

        # Save raw PNG from pymupdf
        raw_path = os.path.join(output_dir, f"page_{i + 1}_raw.png")
        pix.save(raw_path)

        # Resize if too wide (saves VRAM during model inference)
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
