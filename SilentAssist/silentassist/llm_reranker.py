"""
llm_reranker.py
Algorithm 3: LLM Reranking with Clinical Vocabulary
SilentAssist — Silent Speech Reconstruction Pipeline

Uses Phi-3 Mini with LoRA fine-tuning (4-bit quantised) to score
n-best hypotheses from the ViT/CTC stage.  Score interpolation:

    s(h_i) = lambda1 * log P_ViT(h_i) + lambda2 * log P_LLM(h_i)

Reference:
    Namballa, S. (2027). SilentAssist: Silent Speech Reconstruction for
    Non-Verbal Quadriplegics Using Vision Transformers and LLM-Augmented
    Language Prior. ANTIC 2027, Springer CCIS.
"""

import math
import time
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# Clinical / AAC vocabulary for Phi-3 Mini LoRA fine-tuning
# These phrases seed the LoRA training corpus
# ---------------------------------------------------------------------------

CLINICAL_VOCAB = [
    # Basic needs
    "I need water", "I am in pain", "I need help", "call the nurse",
    "I am hot", "I am cold", "I need to rest", "please adjust my pillow",
    "I cannot breathe", "I feel dizzy", "I am nauseous", "please call my family",
    # AAC phrases
    "yes", "no", "thank you", "please", "stop", "more", "less",
    "I do not understand", "can you repeat that", "I need a moment",
    # Medical
    "my back hurts", "I need medication", "where is my wheelchair",
    "I need to use the bathroom", "turn off the light", "turn on the light",
    "I want to watch television", "please open the window",
    # Social
    "good morning", "good night", "how are you", "I am fine",
    "I am happy", "I am tired", "I am frustrated", "I am scared",
]


class LLMReranker:
    """
    Reranks n-best CTC hypotheses using an LLM language score.

    In production: loads Phi-3 Mini (3.8B) with 4-bit GPTQ quantisation
    and a clinical LoRA adapter.

    In this reference implementation: uses a bigram log-probability
    approximation over the clinical vocabulary as a drop-in substitute
    that runs without a GPU, producing qualitatively correct reranking
    behaviour for demonstration purposes.
    """

    def __init__(
        self,
        lambda1: float = 0.6,
        lambda2: float = 0.4,
        use_phi3: bool = False,           # Set True when GPU + model available
        phi3_model_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
    ):
        self.lambda1    = lambda1
        self.lambda2    = lambda2
        self.use_phi3   = use_phi3

        if use_phi3:
            self._load_phi3(phi3_model_path, lora_adapter_path)
        else:
            self._build_clinical_ngram()

    # ------------------------------------------------------------------
    # Phi-3 Mini + LoRA (production path)
    # ------------------------------------------------------------------

    def _load_phi3(self, model_path: str, lora_path: Optional[str]):
        """Load Phi-3 Mini with 4-bit quantisation and LoRA adapter."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            import torch

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model     = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            if lora_path:
                self.phi3 = PeftModel.from_pretrained(base_model, lora_path)
            else:
                self.phi3 = base_model
            self.phi3.eval()
            print(f"[LLMReranker] Phi-3 Mini loaded from {model_path}")
        except ImportError as e:
            print(f"[LLMReranker] WARNING: {e}. Falling back to n-gram proxy.")
            self.use_phi3 = False
            self._build_clinical_ngram()

    def _phi3_log_prob(self, text: str) -> float:
        """Compute per-token mean log-prob under Phi-3 Mini."""
        import torch
        tokens = self.tokenizer(text, return_tensors="pt").to(self.phi3.device)
        with torch.no_grad():
            out    = self.phi3(**tokens, labels=tokens["input_ids"])
            loss   = out.loss.item()       # mean negative log-likelihood
        return -loss

    # ------------------------------------------------------------------
    # N-gram proxy (reference / CPU path)
    # ------------------------------------------------------------------

    def _build_clinical_ngram(self):
        """
        Build character-level bigram model from clinical vocabulary.
        Used as a lightweight proxy for Phi-3 when no GPU is available.
        """
        from collections import defaultdict
        self.bigram_counts  = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)

        for phrase in CLINICAL_VOCAB:
            phrase = phrase.lower()
            for i, ch in enumerate(phrase):
                self.unigram_counts[ch] += 1
                if i > 0:
                    self.bigram_counts[phrase[i-1]][ch] += 1

        # Convert to smoothed log-probs
        self.bigram_logprob: Dict[str, Dict[str, float]] = {}
        for prev, nexts in self.bigram_counts.items():
            total = sum(nexts.values()) + 26
            self.bigram_logprob[prev] = {
                ch: math.log((cnt + 1) / total)
                for ch, cnt in nexts.items()
            }

    def _ngram_log_prob(self, text: str) -> float:
        """Character bigram log-prob (smoothed)."""
        text = text.lower()
        lp   = 0.0
        for i in range(1, len(text)):
            prev = text[i-1]
            curr = text[i]
            lp  += self.bigram_logprob.get(prev, {}).get(curr, -6.0)
        return lp / max(len(text) - 1, 1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, hypothesis: str) -> float:
        """Return LLM log-prob score for a single hypothesis string."""
        if self.use_phi3:
            return self._phi3_log_prob(hypothesis)
        return self._ngram_log_prob(hypothesis)

    def rerank(
        self,
        n_best: List[Tuple[str, float]],   # [(text, log_P_ViT), ...]
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Rerank n-best list using interpolated score.

        Returns:
            best_hypothesis: string
            best_score:      interpolated score
            scored_list:     full list sorted by interpolated score
        """
        scored = []
        for hyp, log_p_vit in n_best:
            log_p_llm = self.score(hyp)
            s = self.lambda1 * log_p_vit + self.lambda2 * log_p_llm
            scored.append((hyp, s, log_p_vit, log_p_llm))

        scored.sort(key=lambda x: x[1], reverse=True)
        best    = scored[0]
        results = [(h, s) for h, s, _, _ in scored]
        return best[0], best[1], results

    def timed_rerank(
        self,
        n_best: List[Tuple[str, float]],
        runs: int = 20,
    ) -> Dict:
        """Run reranking multiple times and report mean latency."""
        times = []
        result = None
        for _ in range(runs):
            t0     = time.perf_counter()
            result = self.rerank(n_best)
            times.append((time.perf_counter() - t0) * 1000)

        import statistics
        return {
            "rerank_ms":      round(statistics.mean(times), 2),
            "best_hypothesis": result[0],
            "best_score":      round(result[1], 4),
        }
