"""
pipeline.py
End-to-end SilentAssist pipeline runner
SilentAssist — Silent Speech Reconstruction Pipeline

Usage:
    from silentassist.pipeline import SilentAssistPipeline
    pipe = SilentAssistPipeline.from_pretrained("./checkpoints/silentassist_v1.pt")
    result = pipe.run("path/to/video.mp4")
    print(result["text"])   # recognised utterance
    # result["audio_path"] -> WAV file spoken by Piper TTS

Reference:
    Namballa, S. (2027). SilentAssist: Silent Speech Reconstruction for
    Non-Verbal Quadriplegics Using Vision Transformers and LLM-Augmented
    Language Prior. ANTIC 2027, Springer CCIS.
"""

import time
import torch
from pathlib import Path
from typing import Optional, Dict

from .landmark_extraction import extract_landmarks_from_video, LANDMARK_DIM
from .vit_model            import SilentAssistViT, ctc_beam_search, VOCAB_SIZE
from .llm_reranker         import LLMReranker
from .tts_output           import PiperTTS


class SilentAssistPipeline:
    """
    Full end-to-end pipeline:
        Video -> Landmarks -> ViT/CTC -> LLM Rerank -> Piper TTS
    """

    def __init__(
        self,
        vit_model:   SilentAssistViT,
        reranker:    LLMReranker,
        tts:         PiperTTS,
        beam_width:  int = 10,
        n_best:      int = 20,
        device:      str = "cpu",
    ):
        self.vit        = vit_model.to(device).eval()
        self.reranker   = reranker
        self.tts        = tts
        self.beam_width = beam_width
        self.n_best     = n_best
        self.device     = device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        use_phi3: bool = False,
        phi3_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        tts_model: Optional[str] = None,
    ) -> "SilentAssistPipeline":
        """Load pipeline from a saved checkpoint."""
        vit = SilentAssistViT()
        ckpt = torch.load(checkpoint_path, map_location=device)
        vit.load_state_dict(ckpt["model_state_dict"])

        reranker = LLMReranker(
            use_phi3=use_phi3,
            phi3_model_path=phi3_path,
            lora_adapter_path=lora_path,
        )
        tts = PiperTTS(model_path=tts_model)
        return cls(vit, reranker, tts, device=device)

    @classmethod
    def build_new(
        cls,
        device: str = "cpu",
        use_phi3: bool = False,
    ) -> "SilentAssistPipeline":
        """Initialise pipeline with a fresh (untrained) ViT for testing."""
        vit      = SilentAssistViT()
        reranker = LLMReranker(use_phi3=use_phi3)
        tts      = PiperTTS()
        return cls(vit, reranker, tts, device=device)

    def run(
        self,
        video_path: str,
        return_timings: bool = True,
    ) -> Dict:
        """
        Process a single video file end-to-end.

        Returns dict:
            text        -> final recognised string
            audio_path  -> path to synthesised WAV (or None if TTS unavailable)
            timings_ms  -> per-stage latency breakdown
            n_best      -> full n-best list from CTC
        """
        timings: Dict[str, float] = {}

        # Stage 1 — Landmark extraction
        t0 = time.perf_counter()
        landmarks, _ = extract_landmarks_from_video(video_path)
        timings["landmark_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if len(landmarks) == 0:
            return {"text": "", "error": "No face detected in video",
                    "timings_ms": timings}

        # Stage 2 — ViT inference + CTC beam search
        t0  = time.perf_counter()
        x   = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.vit(x)            # (1, T, 29)
        n_best_raw = ctc_beam_search(
            log_probs[0],
            beam_width=self.beam_width,
            n_best=self.n_best,
        )
        timings["vit_ctc_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 3 — LLM reranking
        t0 = time.perf_counter()
        best_text, best_score, scored_list = self.reranker.rerank(n_best_raw)
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 4 — TTS synthesis
        t0  = time.perf_counter()
        tts_result = self.tts.synthesise(best_text)
        timings["tts_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        timings["total_ms"] = round(sum(timings.values()), 2)

        return {
            "text":       best_text,
            "score":      best_score,
            "audio_path": tts_result.get("output_path"),
            "timings_ms": timings,
            "n_best":     scored_list[:5],
        }
