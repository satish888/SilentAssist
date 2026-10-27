"""
SilentAssist: Silent Speech Reconstruction for Non-Verbal Quadriplegics
Using Vision Transformers and LLM-Augmented Language Prior

Published in: ANTIC 2027, Springer CCIS (Scopus indexed)
Author: Satish Namballa, Walmart Global Tech
"""

from .pipeline            import SilentAssistPipeline
from .vit_model           import SilentAssistViT, ctc_beam_search, ctc_greedy_decode
from .landmark_extraction import extract_landmarks_from_video, LANDMARK_INDICES
from .llm_reranker        import LLMReranker, CLINICAL_VOCAB
from .tts_output          import PiperTTS

__version__ = "1.0.0"
__author__  = "Satish Namballa"
__all__     = [
    "SilentAssistPipeline",
    "SilentAssistViT",
    "LLMReranker",
    "PiperTTS",
    "ctc_beam_search",
    "ctc_greedy_decode",
    "extract_landmarks_from_video",
    "CLINICAL_VOCAB",
    "LANDMARK_INDICES",
]
