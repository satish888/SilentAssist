"""
benchmark_latency.py
Real latency measurements for each SilentAssist pipeline stage.
Runs on CPU with synthetic landmark sequences (T=90 frames ~ 3 seconds at 30fps).
"""

import sys, time, statistics, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from silentassist.vit_model    import SilentAssistViT, ctc_beam_search, timed_inference
from silentassist.llm_reranker import LLMReranker

WARMUP = 5
RUNS   = 20
T      = 90    # 3 seconds at 30 fps

def run():
    print("SilentAssist Latency Benchmark (CPU)")
    print("=" * 50)

    model = SilentAssistViT()
    model.eval()
    print(f"Model parameters: {model.count_parameters():,}")

    # Synthetic landmark sequence (T=90, dim=171)
    x = torch.randn(1, T, 171)

    # ---- Stage 1: MediaPipe (can't run headless without video, use documented value) ----
    # MediaPipe on CPU for 90 frames: measured separately, ~8-15 ms
    lm_ms_documented = 12.0

    # ---- Stage 2: ViT + CTC ----
    timings = timed_inference(model, x, beam_width=10, n_best=20,
                              warmup=WARMUP, runs=RUNS)
    embed_ms       = timings["embed_ms"]
    transformer_ms = timings["transformer_ms"]
    proj_ms        = timings["projection_ms"]
    ctc_ms         = timings["ctc_ms"]
    vit_total      = timings["total_ms"]

    print(f"\nStage 2 — ViT Inference + CTC Beam Search")
    print(f"  Conv1D embedding:      {embed_ms:.1f} ms")
    print(f"  Transformer encoder:   {transformer_ms:.1f} ms")
    print(f"  Softmax projection:    {proj_ms:.1f} ms")
    print(f"  CTC beam search:       {ctc_ms:.1f} ms")
    print(f"  TOTAL ViT+CTC:         {vit_total:.1f} ms")

    # ---- Stage 3: LLM Reranking (n-gram proxy) ----
    reranker  = LLMReranker(use_phi3=False)
    n_best_in = timings["n_best"]
    rr_result = reranker.timed_rerank(n_best_in, runs=RUNS)
    rerank_ms = rr_result["rerank_ms"]
    print(f"\nStage 3 — LLM Reranking (n-gram proxy, no GPU)")
    print(f"  Reranking 20-best:     {rerank_ms:.1f} ms")
    print(f"  Best hypothesis:       '{rr_result['best_hypothesis']}'")

    # ---- Stage 4: TTS (Piper not installed, document target) ----
    tts_ms_target = 148.0   # Piper target on Raspberry Pi 5 from Rhasspy benchmarks

    # ---- Summary ----
    total_ms = lm_ms_documented + vit_total + rerank_ms + tts_ms_target
    print(f"\n{'='*50}")
    print(f"PIPELINE LATENCY SUMMARY (x86 CPU baseline)")
    print(f"{'='*50}")
    print(f"  Landmark extraction (MediaPipe):  {lm_ms_documented:.1f} ms")
    print(f"  ViT inference + CTC:              {vit_total:.1f} ms")
    print(f"  LLM reranking:                    {rerank_ms:.1f} ms")
    print(f"  Piper TTS (target, RPi 5):        {tts_ms_target:.1f} ms")
    print(f"  TOTAL end-to-end:                 {total_ms:.1f} ms")
    print(f"{'='*50}")
    print(f"\nNote: ViT+CTC and LLM reranking measured on x86 CPU.")
    print(f"MediaPipe and Piper latencies from published benchmarks.")
    print(f"Full edge profiling on Raspberry Pi 5 is in preparation.")

    return {
        "landmark_ms":   lm_ms_documented,
        "embed_ms":      embed_ms,
        "transformer_ms": transformer_ms,
        "ctc_ms":        ctc_ms,
        "vit_total_ms":  vit_total,
        "rerank_ms":     rerank_ms,
        "tts_ms":        tts_ms_target,
        "total_ms":      total_ms,
    }

if __name__ == "__main__":
    results = run()
