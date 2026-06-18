# API — Audio Processing and Indexing

Audio tagging and indexing pipeline using a ConvNeXT model pretrained on AudioSet, with OpenSearch for scalable kNN retrieval. Accepts `.wav` files, generates embeddings, tags them with genre/event labels, and stores everything in an indexed vector store.

## What It Does

1. **Tagging** (`src/tagging/tagging.py`) — runs ConvNeXT (audioset-convnext-inf) inference on a `.wav` clip; outputs class labels and embedding vectors. Optimised for 10-second clips (auto-truncated internally).
2. **Indexing** (`src/indexing/audio_indexing.py`) — stores embeddings + labels in OpenSearch; supports kNN genre classification over large audio corpora.
3. **Streamlit UI** (`src/streamlit-ui/streamlit_app.py`) — upload a WAV, see tags and nearest-neighbour results in a web UI.

## Tech Stack

| Component | Library / Service |
|-----------|------------------|
| Audio tagging | audioset-convnext-inf (ConvNeXT) |
| Deep learning | PyTorch + torchaudio |
| Embeddings / kNN | Towhee + OpenSearch |
| Vector store | OpenSearch (Docker) |
| REST API | FastAPI + Uvicorn |
| UI | Streamlit |
| Language | Python 3.10+ |

## Setup

### 1. Start OpenSearch

```bash
cd src
docker-compose up -d
```

OpenSearch runs on port 9200, Streamlit UI on 8501.

### 2. Install Python dependencies

```bash
cd src
pip install -r requirements.txt
pip install git+https://github.com/topel/audioset-convnext-inf@pip-install
```

### 3. Tag an audio file

```bash
cd src/tagging
python tagging.py
```

Input: any `.wav` file (edit the path in `tagging.py`).

### 4. Run the UI

```bash
cd src/streamlit-ui
streamlit run streamlit_app.py
```

## Known-Working Environment

| Property | Value |
|----------|-------|
| OS | Ubuntu 22.04 / Windows 11 (WSL2) |
| Python | 3.10 – 3.11 |
| Docker | 24+ (for OpenSearch) |
| CUDA | Optional (CPU inference works) |

## Notes

- The `pranaav` branch contains additional multi-modal indexing work — merge to main before deploying.
- AudioSet labels: 527 classes. See `src/tagging/class_labels_indices.csv`.

## Course Context

Audio Processing and Indexing (API) — Semester 4, Leiden University MSc Computer Science (AI) programme.
