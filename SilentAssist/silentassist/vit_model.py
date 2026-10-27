"""
vit_model.py
Algorithm 2: ViT Inference and CTC Beam Search
SilentAssist — Silent Speech Reconstruction Pipeline

Architecture:
    Conv1D patch embedding  ->  6-layer Transformer encoder  ->
    Linear projection (vocab=29)  ->  CTC beam search

Reference:
    Namballa, S. (2027). SilentAssist: Silent Speech Reconstruction for
    Non-Verbal Quadriplegics Using Vision Transformers and LLM-Augmented
    Language Prior. ANTIC 2027, Springer CCIS.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

# Vocabulary: 26 letters + blank + space + apostrophe
VOCAB     = ["-"] + list("abcdefghijklmnopqrstuvwxyz") + [" ", "'"]
BLANK_IDX = 0
VOCAB_SIZE = len(VOCAB)        # 29
CHAR2IDX  = {c: i for i, c in enumerate(VOCAB)}


class SilentAssistViT(nn.Module):
    """
    Vision Transformer for landmark-sequence-to-text transcription.

    Input:  (B, T, 171)  — normalised landmark sequences
    Output: (B, T, 29)   — log-softmax character probabilities (CTC)
    """

    def __init__(
        self,
        input_dim:   int = 171,
        d_model:     int = 256,
        nhead:       int = 8,
        num_layers:  int = 6,
        dim_ff:      int = 1024,
        dropout:     float = 0.1,
        vocab_size:  int = VOCAB_SIZE,
    ):
        super().__init__()
        self.d_model = d_model

        # Conv1D patch embedding (Eq. 1 in paper)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Learnable positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Output projection -> character vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, 171)
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                          # (B, T, 29)
        # Conv1D expects (B, C, T)
        x = x.transpose(1, 2)                  # (B, 171, T)
        x = self.patch_embed(x)                # (B, D, T)
        x = x.transpose(1, 2)                  # (B, T, D)

        x = self.pos_encoding(x)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_proj(x)           # (B, T, vocab)
        return F.log_softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# CTC Greedy Decoder (fast, used during training)
# ---------------------------------------------------------------------------

def ctc_greedy_decode(log_probs: torch.Tensor) -> List[str]:
    """
    Args:
        log_probs: (B, T, vocab) log-softmax probabilities
    Returns:
        List of decoded strings, one per batch item
    """
    predictions = log_probs.argmax(dim=-1)   # (B, T)
    results = []
    for seq in predictions:
        seq = seq.tolist()
        # Collapse repeats and remove blanks
        chars = []
        prev  = None
        for idx in seq:
            if idx != prev:
                if idx != BLANK_IDX:
                    chars.append(VOCAB[idx])
                prev = idx
        results.append("".join(chars))
    return results


# ---------------------------------------------------------------------------
# CTC Beam Search Decoder
# ---------------------------------------------------------------------------

def ctc_beam_search(
    log_probs: torch.Tensor,         # (T, vocab) — single utterance
    beam_width: int = 10,
    n_best: int = 20,
    lm_dict: Optional[set] = None,  # optional lexicon for pruning
) -> List[Tuple[str, float]]:
    """
    Prefix beam search following Hannun (2014).

    Returns:
        n_best list of (hypothesis_string, log_prob) tuples,
        sorted by descending log_prob.
    """
    T, V = log_probs.shape
    probs = log_probs.exp().cpu().numpy()

    # beam: dict of prefix -> (prob_blank, prob_nonblank)
    beam: Dict[tuple, Tuple[float, float]] = {(): (1.0, 0.0)}

    for t in range(T):
        new_beam: Dict[tuple, Tuple[float, float]] = {}

        # Only expand top-k characters for speed
        top_k = min(V, beam_width * 2)
        top_chars = probs[t].argsort()[-top_k:][::-1]

        for prefix, (p_b, p_nb) in beam.items():
            p_total = p_b + p_nb

            for c in top_chars:
                p_c = float(probs[t, c])

                if c == BLANK_IDX:
                    # Extend with blank
                    key = prefix
                    pb, pnb = new_beam.get(key, (0.0, 0.0))
                    new_beam[key] = (pb + p_total * p_c, pnb)
                else:
                    # Extend with character
                    new_prefix = prefix + (c,)
                    pb, pnb = new_beam.get(new_prefix, (0.0, 0.0))

                    if prefix and prefix[-1] == c:
                        # Same char as last — only blank can extend
                        new_beam[new_prefix] = (pb, pnb + p_nb * p_c)
                    else:
                        new_beam[new_prefix] = (pb, pnb + p_total * p_c)

        # Prune to beam_width
        beam = dict(
            sorted(new_beam.items(),
                   key=lambda kv: kv[1][0] + kv[1][1],
                   reverse=True)[:beam_width]
        )

    # Convert to strings
    results = []
    for prefix, (p_b, p_nb) in beam.items():
        text  = "".join(VOCAB[c] for c in prefix if c != BLANK_IDX)
        score = math.log(p_b + p_nb + 1e-30)
        if lm_dict is None or not text or any(w in lm_dict for w in text.split()):
            results.append((text, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n_best]


# ---------------------------------------------------------------------------
# Timed inference helper
# ---------------------------------------------------------------------------

def timed_inference(
    model: SilentAssistViT,
    landmarks: torch.Tensor,          # (1, T, 171)
    beam_width: int = 10,
    n_best: int = 20,
    warmup: int = 3,
    runs: int = 10,
) -> Dict:
    """
    Run inference N times and report mean latency breakdown.

    Returns dict with keys: embed_ms, transformer_ms, ctc_ms, total_ms, n_best_list
    """
    model.eval()
    timings = {"embed": [], "transformer": [], "projection": [], "ctc": []}

    with torch.no_grad():
        for i in range(warmup + runs):
            t0 = time.perf_counter()

            # Embedding
            x = landmarks.transpose(1, 2)
            x = model.patch_embed(x)
            x = x.transpose(1, 2)
            x = model.pos_encoding(x)
            t1 = time.perf_counter()

            # Transformer
            h = model.transformer(x)
            t2 = time.perf_counter()

            # Projection
            logits = model.output_proj(h)
            log_p  = F.log_softmax(logits, dim=-1)
            t3 = time.perf_counter()

            # CTC beam search (single utterance)
            nbest = ctc_beam_search(log_p[0], beam_width=beam_width, n_best=n_best)
            t4 = time.perf_counter()

            if i >= warmup:
                timings["embed"].append((t1 - t0) * 1000)
                timings["transformer"].append((t2 - t1) * 1000)
                timings["projection"].append((t3 - t2) * 1000)
                timings["ctc"].append((t4 - t3) * 1000)

    import statistics
    return {
        "embed_ms":       round(statistics.mean(timings["embed"]), 2),
        "transformer_ms": round(statistics.mean(timings["transformer"]), 2),
        "projection_ms":  round(statistics.mean(timings["projection"]), 2),
        "ctc_ms":         round(statistics.mean(timings["ctc"]), 2),
        "total_ms":       round(sum(statistics.mean(v) for v in timings.values()), 2),
        "n_best":         nbest,
    }
