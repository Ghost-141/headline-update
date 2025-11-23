# Headline Deduplication System

A semantic similarity-based news headline deduplication system using FAISS vector search and sentence transformers.

## Features

- **Semantic Deduplication**: Uses sentence transformers to detect similar headlines across different sources
- **Vector Search**: FAISS-based efficient similarity search for large-scale headline processing
- **Timestamp-based Indexing**: Headlines sorted by publication timestamp (latest first)
- **Multi-source Support**: Handles headlines from multiple news sources
- **Persistent Storage**: Saves processed data to disk for incremental updates

## Project Structure

```
headline/
├── database/                 # Persistent storage
│   ├── metadata.json        # Headline metadata (text, timestamp, source)
│   └── vector_store.index   # FAISS vector index
├── raw_data/                # Input data files
│   ├── news.txt             # Raw news data
│   └── test.txt             # Test data
├── utils/                   # Core utilities
│   ├── file_handler.py      # File I/O and parsing functions
│   └── utils.py             # Vector operations and deduplication logic
├── main.py                  # Main processing script
└── README.md
```

## Input Format

Raw news data should be formatted as:

```
==============================
Fetching: Source Name
==============================
1. Headline text here (2025-11-23 14:01:14)
2. Another headline (2025-11-23 13:58:00)
...

==============================
📰 Fetching: Another Source
==============================
1. Different headline (2025-11-23 13:55:00)
...
```

## Output Format

The system outputs only new unique headlines added in the current run:

```json
{
  "Source Name": [
    {
      "headline": "New unique headline",
      "timestamp": "2025-11-23T14:01:14"
    }
  ],
  "Another Source": [
    {
      "headline": "Another unique headline", 
      "timestamp": "2025-11-23T13:55:00"
    }
  ]
}
```

## Configuration

- **Similarity Threshold**: 0.95 (95% similarity required for duplicate detection)
- **Vector Dimension**: 768 (sentence-transformers/all-mpnet-base-v2)
- **Storage**: Local filesystem (database/ directory)

## Usage

1. Place raw news data in `raw_data/test.txt`
2. Run the main script:
   ```bash
   python main.py
   ```
3. View output of new unique headlines
4. Subsequent runs with same data will output empty `{}` (no duplicates added)

## Key Functions

### utils.py
- `get_config()`: Model & Vector DB configuration
- `load_faiss_and_metadata()`: Load existing data from disk
- `add_or_update()`: Process headlines with duplicate detection
- `embed_text()`: Generate text embeddings

### file_handler.py
- `parse_news_block_to_faiss()`: Parse raw news text
- `save_state()`: Persist data to disk
- `get_grouped_output()`: Format output by source

## Dependencies

- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Text embedding generation
