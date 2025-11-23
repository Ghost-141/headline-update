import os
import json
import faiss
from typing import Dict, List
import re
from datetime import datetime


def save_state(DATA_DIR: str, INDEX_PATH: str, faiss_index, metadata, METADATA_PATH):
    """Save FAISS index and metadata to disk for persistence.
    
    Args:
        DATA_DIR: Directory path for data storage
        INDEX_PATH: File path for FAISS index
        faiss_index: FAISS index instance to save
        metadata: Metadata dictionary to save
        METADATA_PATH: File path for metadata JSON
    """
    # ensure directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # write index
    try:
        faiss.write_index(faiss_index, INDEX_PATH)
    except Exception as e:
        # If writing the IndexIDMap fails for any reason, try writing its underlying index
        try:
            faiss.write_index(faiss_index.index, INDEX_PATH)
        except Exception:
            print("Failed to write FAISS index:", e)

    # write metadata (timestamps as isoformat)
    serializable = {
        str(k): {
            "headline": v["headline"],
            "source": v["source"],
            "timestamp": (v["timestamp"].isoformat() if v["timestamp"] else None),
        }
        for k, v in metadata.items()
    }

    with open(METADATA_PATH, "w", encoding="utf8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def get_all_headlines(metadata):
    """Retrieve all headlines from metadata in list format.
    
    Args:
        metadata: Metadata dictionary containing headlines
        
    Returns:
        list: List of headline dictionaries with id, headline, timestamp, source
    """
    return [
        {
            "id": k,
            "headline": v["headline"],
            "timestamp": v["timestamp"].isoformat(),
            "source": v["source"],
        }
        for k, v in metadata.items()
    ]


def get_grouped_output(metadata) -> Dict[str, List[Dict[str, str]]]:
    """Return a dict grouped by source with headlines sorted by timestamp (newest first)."""
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for id_, item in metadata.items():
        src = item["source"]
        grouped.setdefault(src, []).append(
            {
                "headline": item["headline"],
                "timestamp": (item["timestamp"].isoformat() if item["timestamp"] else None),  # type: ignore
            }
        )

    # sort each list by timestamp desc
    for src, items in grouped.items():
        items.sort(key=lambda x: x["timestamp"] or "", reverse=True)

    return grouped


# Regex patterns for parsing news text
SECTION_HEADER_RE = re.compile(r"Fetching:\s*(.+)")
HEADLINE_RE = re.compile(r"^\d+\.\s*(.+)\s+\(([^()]+)\)$")


def parse_news_block_to_faiss(raw_text: str):
    """Parse raw news text into structured headline data.
    
    Extracts headlines, timestamps, and sources from formatted news text.
    Handles various timestamp formats and skips relative timestamps.
    
    Args:
        raw_text: Raw news text with headlines and sources
        
    Returns:
        list: List of (headline, timestamp, source) tuples
    """
    parsed = []
    current_source = None

    # Updated regex patterns to handle variations
    SECTION_HEADER_RE = re.compile(r"(?:📰\s*)?Fetching:\s*(.+)")
    HEADLINE_RE = re.compile(r"^\d+\.\s*(.+?)\s+\(([^()]+)\)(?:\s+\d+)?$")

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue

        # detect source section
        m = SECTION_HEADER_RE.search(line)
        if m:
            current_source = m.group(1).strip()
            continue

        # detect numbered headline with timestamp
        m = HEADLINE_RE.match(line)
        if m and current_source:
            headline = m.group(1).strip()
            ts_str = m.group(2).strip()

            # Handle different timestamp formats
            if "আগে" in ts_str or "সেকেন্ড" in ts_str:
                # Skip relative timestamps like "৩৫ সেকেন্ড আগে"
                continue

            try:
                ts = datetime.fromisoformat(ts_str)
                parsed.append((headline, ts, current_source))
            except Exception:
                continue

    return parsed
