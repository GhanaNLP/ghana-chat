"""Ghana Chat: Head-focused Knowledge Graph Grounded QA powered by Qwen 2B on NVIDIA H200."""

__version__ = "0.1.0"

from .retriever import HeadKGRetriever
from .generator import GroundedGenerator

__all__ = ["HeadKGRetriever", "GroundedGenerator", "__version__"]
