import base64
import httpx
from config import MODEL_SERVER_URL, MODEL_NAME, MAX_NEW_TOKENS

# Plain HTTP client — calls local model server directly, no cloud, no OpenAI
# transformers serve exposes a REST API at MODEL_SERVER_URL
# Timeouts: connect 30s, read 10min (model inference on large pages takes time)
client = httpx.Client(
    timeout=httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=10.0)
)

# Auto prompt injected on every image
AUTO_PROMPT = """You are a document reconstruction expert.
Analyze this scanned government document page image and extract ALL content with exact structure preserved.

Output requirements:
- Reproduce all text exactly as it appears, preserving reading order
- Reconstruct all tables in markdown table format with all rows and columns
- Preserve section headings, sub-headings, numbering
- For handwritten content: transcribe exactly, mark unclear parts as [UNCLEAR]
- For Hindi/mixed text: transcribe in the original script, do not translate
- Do NOT add any commentary, explanations or summaries
- Output ONLY the reconstructed document content

Begin reconstruction:"""


def process_image(image_path: str, page_number: int) -> str:
    """
    Send page image to local Qwen model server via plain HTTP POST.
    No cloud. No OpenAI. Calls localhost only.
    Returns structured text output.
    """
    # Read image and encode as base64
    with open(image_path, "rb") as f:
        raw = f.read()

    # Warn if image very large — may slow model inference
    size_mb = len(raw) / 1024 / 1024
    if size_mb > 3:
        print(f"⚠️  Page {page_number} image is {size_mb:.1f}MB — consider lowering PDF_DPI in config.py")

    b64_image = base64.b64encode(raw).decode("utf-8")

    # Build request payload — same format transformers serve expects
    payload = {
        "model": MODEL_NAME,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": AUTO_PROMPT
                    }
                ]
            }
        ]
    }

    try:
        response = client.post(
            f"{MODEL_SERVER_URL}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()

        # Extract text from response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            return f"[Page {page_number}: Model returned empty response — try again]"

        return content

    except httpx.TimeoutException:
        return f"[Page {page_number} TIMEOUT]: Model took too long — try reducing PDF_DPI in config.py"
    except httpx.ConnectError:
        return f"[Page {page_number} CONNECTION ERROR]: Model server not running — run start_model.sh first"
    except Exception as e:
        return f"[Page {page_number} ERROR]: {str(e)}"
