import json
from utils.utils import (
    get_config,
    load_faiss_and_metadata,
    embed_text,
    add_or_update,
)
from utils.file_handler import (
    save_state,
    parse_news_block_to_faiss,
)

# Get configuration
(
    model,
    DIM,
    faiss_index,
    DATA_DIR,
    INDEX_PATH,
    METADATA_PATH,
    SIM_THRESHOLD,
) = get_config()

# Load existing data
faiss_index, metadata, next_id = load_faiss_and_metadata(
    DATA_DIR, INDEX_PATH, METADATA_PATH, DIM
)

# Track existing IDs to identify new ones
existing_ids = set(metadata.keys())

# Parse and process new data
with open("raw_data/test.txt", "r", encoding="utf-8") as f:
    text = f.read()

incoming_data = parse_news_block_to_faiss(text)

# Process each headline
for headline, timestamp, source in incoming_data:
    next_id, faiss_index = add_or_update(
        headline,
        timestamp,
        source,
        lambda x: embed_text(model, x),
        SIM_THRESHOLD,
        metadata,
        faiss_index,
        next_id,
        model,
        DIM,
    )

# Save final state
save_state(DATA_DIR, INDEX_PATH, faiss_index, metadata, METADATA_PATH)

# Get only new headlines added in this run
new_headlines = {k: v for k, v in metadata.items() if k not in existing_ids}

# Group new headlines by source and sort by timestamp
grouped_new = {}
for k, v in new_headlines.items():
    source = v["source"]
    if source not in grouped_new:
        grouped_new[source] = []
    grouped_new[source].append(
        {"headline": v["headline"], "timestamp": v["timestamp"].isoformat()}
    )

# Sort each source by timestamp (newest first)
for source in grouped_new:
    grouped_new[source].sort(key=lambda x: x["timestamp"], reverse=True)

# Output only new unique headlines
print(json.dumps(grouped_new, ensure_ascii=False, indent=2))
