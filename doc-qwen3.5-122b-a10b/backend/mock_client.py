import time


def process_image(image_path: str, page_number: int) -> str:
    """
    MOCK CLIENT — used on laptop for UI/flow testing.
    No model needed. Returns fake structured output.
    Switch to model_client.py on GPU machine by setting USE_MOCK=False in config.py
    """
    time.sleep(1.5)  # simulate model thinking time

    return f"""## Page {page_number} — Mock Output

**[This is a mock response. Real model output will appear here on GPU machine.]**

---

### Section 1 — Header Block

| Field        | Value              |
|--------------|--------------------|
| Document No  | PWD/BCD/2024/0042  |
| Date         | 15-Jan-2024        |
| Division     | Bridge Construction|
| District     | Bhopal             |

---

### Section 2 — Extracted Text

Lorem ipsum extracted text from page {page_number} of the scanned government document.
This area will show actual OCR/VLM output once running on GPU machine with real Qwen model.

---

### Section 3 — Financial Table

| Sr | Description        | Unit | Qty  | Rate   | Amount     |
|----|--------------------|------|------|--------|------------|
| 1  | Cement M30         | Bags | 240  | 420.00 | 1,00,800   |
| 2  | Steel TMT 12mm     | MT   | 4.5  | 68,000 | 3,06,000   |
| 3  | Labour Charges     | LS   | --   | --     | 45,000     |
|    | **Total**          |      |      |        | **4,51,800**|

---

*Image path processed: `{image_path}`*
*Replace mock_client with model_client in main.py when ready.*
"""
