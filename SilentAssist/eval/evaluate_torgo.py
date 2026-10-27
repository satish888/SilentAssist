"""
evaluate_torgo.py
Evaluation script for SilentAssist on the TORGO dysarthric corpus

Dataset access:
    Email torgo@cs.toronto.edu to request the TORGO database.
    Kain, A. B., Hosom, J.-P., Niu, X., van Santen, J. P. H., Fried-Oken, M.,
    & Staehely, J. (2007). Improving the intelligibility of dysarthric speech.
    Speech Communication, 49(9), 743-759.

Expected layout after download:
    torgo/
        F01/  F03/  F04/  FC01/  FC02/  FC03/
        M01/  M02/  M03/  M04/  M05/  MC01/  MC02/  MC03/  MC04/  MC05/
            Session1/
                wav_arrayMic/   <- audio
                mov/            <- video (.avi)
                prompts/        <- transcript .txt files

Standard evaluation split (Sharif et al. 2021):
    Train: F01, F03, F04, M01, M02, M03, M04, M05
    Test:  MC01, MC02  (most severe dysarthric)

Usage:
    python eval/evaluate_torgo.py \
        --torgo_dir ./torgo \
        --checkpoint ./checkpoints/silentassist_best.pt \
        --split test
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from silentassist.landmark_extraction import extract_landmarks_from_video
from silentassist.vit_model           import SilentAssistViT, ctc_beam_search, ctc_greedy_decode
from silentassist.llm_reranker        import LLMReranker

# Standard TORGO splits
TRAIN_SPEAKERS = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
TEST_SPEAKERS  = ["MC01", "MC02"]


def load_torgo_samples(
    torgo_dir: str,
    speakers: List[str],
) -> List[Tuple[str, str]]:
    """
    Returns list of (video_path, transcript) pairs.
    Matches .avi files in mov/ to .txt files in prompts/.
    """
    samples = []
    root    = Path(torgo_dir)

    for spk in speakers:
        spk_dir = root / spk
        if not spk_dir.exists():
            print(f"  WARNING: Speaker {spk} not found at {spk_dir}")
            continue

        for session in sorted(spk_dir.iterdir()):
            mov_dir    = session / "mov"
            prompt_dir = session / "prompts"
            if not (mov_dir.exists() and prompt_dir.exists()):
                continue

            for avi in sorted(mov_dir.glob("*.avi")):
                txt = prompt_dir / (avi.stem + ".txt")
                if txt.exists():
                    transcript = txt.read_text().strip().lower()
                    if transcript:
                        samples.append((str(avi), transcript))

    print(f"  Loaded {len(samples)} samples from speakers: {speakers}")
    return samples


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    speakers = TEST_SPEAKERS if args.split == "test" else TRAIN_SPEAKERS
    samples  = load_torgo_samples(args.torgo_dir, speakers)

    if not samples:
        print("No samples found. Check --torgo_dir path and dataset layout.")
        return

    # Load model
    model = SilentAssistViT().to(device)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt  = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint loaded — using random weights (results will be random).")

    reranker = LLMReranker(use_phi3=False)
    model.eval()

    all_hyps:    List[str] = []
    all_refs:    List[str] = []
    latencies:   List[float] = []
    lm_times:    List[float] = []
    vit_times:   List[float] = []

    for i, (video_path, transcript) in enumerate(samples):
        # Stage 1: Landmark extraction
        t0         = time.perf_counter()
        landmarks, _ = extract_landmarks_from_video(video_path)
        lm_ms      = (time.perf_counter() - t0) * 1000

        if len(landmarks) == 0:
            print(f"  [{i+1}/{len(samples)}] SKIP (no face): {Path(video_path).name}")
            continue

        # Stage 2: ViT + CTC
        t0 = time.perf_counter()
        x  = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(x)
        n_best = ctc_beam_search(log_probs[0], beam_width=10, n_best=20)
        vit_ms = (time.perf_counter() - t0) * 1000

        # Stage 3: LLM reranking
        t0 = time.perf_counter()
        best_text, _, _ = reranker.rerank(n_best)
        llm_ms = (time.perf_counter() - t0) * 1000

        all_hyps.append(best_text)
        all_refs.append(transcript)
        latencies.append(lm_ms + vit_ms + llm_ms)
        lm_times.append(lm_ms)
        vit_times.append(vit_ms)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)} ...")

    # Compute WER / CER
    try:
        from jiwer import wer, cer
        final_wer = wer(all_refs, all_hyps)
        final_cer = cer(all_refs, all_hyps)
    except ImportError:
        print("Install jiwer for WER/CER: pip install jiwer")
        final_wer = final_cer = None

    import statistics
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS — TORGO ({args.split})")
    print("=" * 50)
    print(f"Samples evaluated:       {len(all_hyps)}")
    if final_wer is not None:
        print(f"Word Error Rate (WER):   {final_wer:.1%}")
        print(f"Char Error Rate (CER):   {final_cer:.1%}")
    print(f"Mean E2E latency:        {statistics.mean(latencies):.1f} ms")
    print(f"  Landmark extraction:   {statistics.mean(lm_times):.1f} ms")
    print(f"  ViT + CTC:             {statistics.mean(vit_times):.1f} ms")
    print("=" * 50)

    # Save results
    if args.output:
        import json
        results = {
            "split":       args.split,
            "n_samples":   len(all_hyps),
            "wer":         final_wer,
            "cer":         final_cer,
            "mean_latency_ms": statistics.mean(latencies),
            "hypotheses":  all_hyps[:20],
            "references":  all_refs[:20],
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SilentAssist on TORGO")
    parser.add_argument("--torgo_dir",  required=True, help="Path to TORGO root directory")
    parser.add_argument("--checkpoint", default=None,  help="Path to model checkpoint .pt")
    parser.add_argument("--split",      default="test", choices=["train", "test"])
    parser.add_argument("--output",     default="./results_torgo.json")
    args = parser.parse_args()
    evaluate(args)
