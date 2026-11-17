import os
import io
import json
import glob
import re
import boto3
import pdfplumber
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()

INPUT_DIR = "data/incoming_docs"
OUTPUT_JSON = "data/auto_batch_chunks.json"
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
TABLE_ROWS_PER_CHUNK = 10  # Optimal slice size for dense tables

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
NOVA_MODEL_ID = "apac.amazon.nova-lite-v1:0"

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_vision_with_nova_lite(image_pil, prompt_instruction: str) -> str:
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format="PNG")
    image_bytes = img_byte_arr.getvalue()

    try:
        response = bedrock.converse(
            modelId=NOVA_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                        {
                            "text": (
                                f"{prompt_instruction}\n"
                                "Instructions:\n"
                                "1. Transcribe all text, numbers, dates, codes, and signatories exactly as visible.\n"
                                "2. Format tables strictly as clean Markdown tables.\n"
                                "3. Preserve all bilingual English and Hindi terms and subject codes.\n"
                                "4. Output ONLY the extracted text/tables without commentary."
                            )
                        }
                    ]
                }
            ],
            inferenceConfig={"temperature": 0.0, "maxTokens": 3000}
        )
        output_blocks = response["output"]["message"]["content"]
        return clean_text("".join([b["text"] for b in output_blocks if "text" in b]))
    except Exception as e:
        print(f"    [Nova Lite Vision Error] {e}")
        return ""

def detect_metadata(cover_text: str, filename: str) -> tuple[str, str]:
    notif = ""
    subj = ""

    m_notif = re.search(
        r'(?:NOTIFICATION|CIRCULAR|अधिसूचना|परिपत्र\s*संख्या)\s*[-–:\s]*\s*([0-9]{1,3}\s*[/–\-]\s*[0-9]{4})',
        cover_text,
        re.IGNORECASE
    )
    if m_notif:
        prefix = "Notification" if ("notif" in cover_text.lower() or "अधिसूचना" in cover_text) else "Circular"
        num = clean_text(m_notif.group(1)).replace(" ", "")
        notif = f"{prefix} {num}"
    elif "DATE SHEET" in cover_text.upper() or "तिथि तालिका" in cover_text or "8HJJoy" in filename:
        notif = "Notification 17/2022 | Date Sheet for Theory Examination (Oct-Nov 2022)"

    m_subj = re.search(r'(?:Subject|विषय)\s*[:–-]?\s*([^\n\r]+)', cover_text, re.IGNORECASE)
    if m_subj:
        subj = clean_text(m_subj.group(1))[:100]

    title = " | ".join([p for p in [notif, subj] if p]) or os.path.splitext(filename)[0].replace("_", " ")
    return notif or title, title

def chunk_table_with_headers(table: list, rows_per_chunk: int = TABLE_ROWS_PER_CHUNK) -> list[str]:
    """Splits large tables into small row-windows while re-injecting the header into every chunk."""
    if not table or len(table) < 2:
        return []

    cleaned = []
    for r in table:
        c_row = [clean_text(str(c or "")) if c is not None else "-" for c in r]
        if any(cell and cell != "-" for cell in c_row):
            cleaned.append(c_row)
    if len(cleaned) < 2:
        return []

    headers = cleaned[0]
    num_cols = len(headers)
    header_block = (
        "| " + " | ".join(headers) + " |\n"
        "| " + " | ".join(["---"] * num_cols) + " |"
    )

    data_rows = cleaned[1:]
    table_chunks = []

    for i in range(0, len(data_rows), rows_per_chunk):
        slice_rows = data_rows[i:i + rows_per_chunk]
        chunk_lines = [header_block]
        for r in slice_rows:
            if len(r) < num_cols:
                r.extend(["-"] * (num_cols - len(r)))
            chunk_lines.append("| " + " | ".join(r[:num_cols]) + " |")
        table_chunks.append("\n".join(chunk_lines))

    return table_chunks

def process_pdf(pdf_path: str, start_id: int):
    filename = os.path.basename(pdf_path)
    chunks = []
    c_id = start_id

    print(f"--> Parsing: {filename}")

    page_images = None
    try:
        page_images = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        print(f"    [Notice] Rendering skipped: {e}")

    with pdfplumber.open(pdf_path) as pdf:
        p1_raw = pdf.pages[0].extract_text() or "" if len(pdf.pages) > 0 else ""
        if len(clean_text(p1_raw)) < 20 and page_images and len(page_images) > 0:
            print("    [Nova Lite Vision] Extracting Page 1 Executive Directives...")
            p1_raw = extract_vision_with_nova_lite(
                page_images[0],
                "Extract all administrative notices, signatories, bank details, and dates from this cover page."
            )

        p1_clean = clean_text(p1_raw)
        short_label, doc_title = detect_metadata(p1_clean, filename)

        cover_summary = (
            f"Document Title: {doc_title} (Source: {filename})\n"
            f"Executive Directives & Payment Guidelines (Page 1):\n{p1_clean}\n"
            f"--------------------------------------------------\n"
        )

        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""

            is_date_sheet = "date" in filename.lower() or "8hjjoy" in filename.lower()
            needs_vision = (len(clean_text(raw_text)) < 20 or is_date_sheet)

            if needs_vision and page_images and page_idx < len(page_images):
                print(f"    [Nova Lite Vision] Parsing Page {page_num}...")
                prompt = (
                    "Extract the examination timetable into a structured Markdown table. "
                    "Columns: Date & Day | Senior Secondary Subject (with Code & Time) | Secondary Subject (with Code & Time)."
                    if is_date_sheet else
                    "Extract all tables, notices, and textual details from this document page into structured Markdown."
                    "Extract both the Practical Examination Schedule (date ranges and batch guidelines) and the Theory Examination Timetable into distinct Markdown sections with their respective footnotes."
                )
                raw_text = extract_vision_with_nova_lite(page_images[page_idx], prompt)

            cleaned_body = clean_text(raw_text)

            # Strategy 1: Header-Preserving Table Chunking
            tables = page.extract_tables()
            for tbl in tables:
                sub_tables = chunk_table_with_headers(tbl, rows_per_chunk=TABLE_ROWS_PER_CHUNK)
                for tbl_idx, md_tbl in enumerate(sub_tables):
                    header = (
                        f"Document: {doc_title} (Source: {filename}) | Page: {page_num} | Table Section {tbl_idx + 1}\n"
                        f"Type: Structured Data Table\n\n"
                    )
                    content = (cover_summary if page_num > 1 else "") + md_tbl
                    chunks.append({
                        "chunk_id": f"chunk_{c_id:05d}",
                        "text": header + content,
                        "metadata": {
                            "source": filename,
                            "doc_title": doc_title,
                            "page": page_num,
                            "type": "table_data",
                            "section": short_label
                        }
                    })
                    c_id += 1

            # Strategy 2: Narrative / Section Chunking
            if len(cleaned_body) > 20:
                if page_num == 1:
                    header = (
                        f"Document: {doc_title} (Source: {filename}) | Page: 1\n"
                        f"Section: Executive Directives and Guidelines\n\n"
                    )
                    chunks.append({
                        "chunk_id": f"chunk_{c_id:05d}",
                        "text": header + cleaned_body,
                        "metadata": {
                            "source": filename,
                            "doc_title": doc_title,
                            "page": 1,
                            "type": "prose_section",
                            "section": "Executive Directives"
                        }
                    })
                    c_id += 1
                else:
                    paragraphs = [p.strip() for p in cleaned_body.split('\n\n') if p.strip()]
                    buffer = ""
                    for p in paragraphs:
                        if len(buffer) + len(p) < MAX_CHUNK_CHARS:
                            buffer += "\n\n" + p
                        else:
                            if len(buffer.strip()) > 20:
                                header = (
                                    f"Document: {doc_title} (Source: {filename}) | Page: {page_num}\n"
                                    f"Section: {short_label}\n\n"
                                )
                                chunks.append({
                                    "chunk_id": f"chunk_{c_id:05d}",
                                    "text": header + buffer.strip(),
                                    "metadata": {
                                        "source": filename,
                                        "doc_title": doc_title,
                                        "page": page_num,
                                        "type": "prose_section",
                                        "section": short_label
                                    }
                                })
                                c_id += 1
                            buffer = buffer[-CHUNK_OVERLAP:] + "\n\n" + p if len(buffer) > CHUNK_OVERLAP else p

                    if len(buffer.strip()) > 20:
                        header = (
                            f"Document: {doc_title} (Source: {filename}) | Page: {page_num}\n"
                            f"Section: {short_label}\n\n"
                        )
                        chunks.append({
                            "chunk_id": f"chunk_{c_id:05d}",
                            "text": header + buffer.strip(),
                            "metadata": {
                                "source": filename,
                                "doc_title": doc_title,
                                "page": page_num,
                                "type": "prose_section",
                                "section": short_label
                            }
                        })
                        c_id += 1

    return chunks, c_id

def batch_process():
    os.makedirs(INPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))
    all_chunks = []
    c_id = 1
    for f in files:
        f_chunks, c_id = process_pdf(f, c_id)
        all_chunks.extend(f_chunks)

    print(f"\nTotal Chunks Generated: {len(all_chunks)} across {len(files)} files.")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(all_chunks, out, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    batch_process()