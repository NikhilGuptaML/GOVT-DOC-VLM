import base64
from openai import OpenAI
from config import MODEL_SERVER_URL, MODEL_NAME, MAX_NEW_TOKENS

# OpenAI-compatible client pointing to local transformers serve
client = OpenAI(
    base_url=MODEL_SERVER_URL,
    api_key="none",  # no key needed for local server
)

# Auto prompt injected on every image — do not change structure
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
    Send page image to local Qwen model server.
    Returns structured text output.
    """
    # Read and encode image as base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": AUTO_PROMPT,
                    },
                ],
            }
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.1,  # low temp = more accurate, less hallucination
    )

    return response.choices[0].message.content
