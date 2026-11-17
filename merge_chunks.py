import json

INPUT_FILE = "data/nios_prospectus_chunks.json"
OUTPUT_FILE = "data/nios_prospectus_chunks_merged.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} raw chunk fragments.")

merged_chunks = []
buffer_chunk = None

for c in chunks:
    text = c.get("text", "").strip()
    meta = c.get("metadata", {})
    page = meta.get("page")
    block_id = meta.get("block_id")

    # If it's a continuing bullet, table row, or fragment on the same page
    is_fragment = (
        text.startswith("➣") or 
        text.startswith("➤") or 
        text.startswith("➥") or 
        text.startswith("•") or
        text.startswith("(") or
        len(text.split()) < 12
    )

    if buffer_chunk and buffer_chunk["metadata"].get("page") == page and is_fragment:
        buffer_chunk["text"] += "\n" + text
    else:
        if buffer_chunk:
            merged_chunks.append(buffer_chunk)
        buffer_chunk = {
            "text": text,
            "metadata": meta
        }

if buffer_chunk:
    merged_chunks.append(buffer_chunk)

print(f"Consolidated into {len(merged_chunks)} cohesive chunks.")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged_chunks, f, indent=2, ensure_ascii=False)

print(f"Saved optimized dataset to {OUTPUT_FILE}.")