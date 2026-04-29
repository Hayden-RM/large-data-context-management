from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = REPO_ROOT / "data" / "samples" / "corpus.txt"
NEEDLES_PATH = REPO_ROOT / "data" / "samples" / "needles.json"
RESULTS_DIR = REPO_ROOT / "results" / "exp01"

# Model used for answering (map, reduce, RAG answer step)
ANSWER_MODEL = "claude-haiku-4-5"

# Chunking
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64

# RAG
RAG_TOP_K = 5

# Hierarchical summarisation target before answering
HIER_TARGET_TOKENS = 8_000

# Sliding window
WINDOW_TOKENS = 4_096
STEP_TOKENS = 2_048

# Embedding backend: "local" or "openai"
EMBED_BACKEND = "local"
