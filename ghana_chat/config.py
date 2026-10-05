"""Configuration settings for Ghana Chat."""

import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MODEL_ID = os.environ.get("GHANA_CHAT_MODEL", "Qwen/Qwen3.5-2B")
KG_PATH = os.environ.get("GHANA_CHAT_KG_PATH", str(DATA_DIR / "ghana_triples.parquet"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DEFAULT_HOST = os.environ.get("GHANA_CHAT_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("GHANA_CHAT_PORT", "8000"))

MAX_RETRIEVED_TRIPLES = int(os.environ.get("MAX_RETRIEVED_TRIPLES", "10"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
