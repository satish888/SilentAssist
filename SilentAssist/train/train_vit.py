"""
train_vit.py
Training script for SilentAssist ViT model

Expected dataset layout:
    data/
        train/
            speaker_id/
                utterance_id.mp4
                utterance_id.txt    <- ground-truth transcript
        test/
            ...

Usage:
    python train/train_vit.py --data_dir ./data --epochs 50 --batch_size 8
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from silentassist.landmark_extraction import extract_landmarks_from_video, LANDMARK_DIM
from silentassist.vit_model           import SilentAssistViT, CHAR2IDX, VOCAB_SIZE, BLANK_IDX


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LandmarkDataset(Dataset):
    """
    Loads pre-extracted landmark .npy files and their transcripts.
    Run extract_landmarks_from_video() once per video and save to .npy
    to avoid re-extracting on every epoch.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        self.samples: List[Tuple[str, str]] = []
        root = Path(data_dir) / split
        for speaker in sorted(root.iterdir()):
            for npy_file in sorted(speaker.glob("*.npy")):
                txt_file = npy_file.with_suffix(".txt")
                if txt_file.exists():
                    self.samples.append((str(npy_file), txt_file.read_text().strip().lower()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, transcript = self.samples[idx]
        landmarks = np.load(npy_path).astype(np.float32)   # (T, 171)
        labels    = [CHAR2IDX.get(c, 0) for c in transcript if c in CHAR2IDX]
        return torch.tensor(landmarks), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    landmarks, labels = zip(*batch)
    T_max = max(x.shape[0] for x in landmarks)
    L_max = max(len(l) for l in labels)

    padded_lm = torch.zeros(len(landmarks), T_max, LANDMARK_DIM)
    padded_lb = torch.zeros(len(labels),    L_max, dtype=torch.long)
    lm_lens   = torch.zeros(len(landmarks), dtype=torch.long)
    lb_lens   = torch.zeros(len(labels),    dtype=torch.long)

    for i, (lm, lb) in enumerate(zip(landmarks, labels)):
        padded_lm[i, :lm.shape[0]] = lm
        padded_lb[i, :len(lb)]     = lb
        lm_lens[i]                 = lm.shape[0]
        lb_lens[i]                 = len(lb)

    return padded_lm, padded_lb, lm_lens, lb_lens


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_ds = LandmarkDataset(args.data_dir, "train")
    test_ds  = LandmarkDataset(args.data_dir, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=4)

    model     = SilentAssistViT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    print(f"Model parameters: {model.count_parameters():,}")
    os.makedirs(args.output_dir, exist_ok=True)
    best_wer = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for lm, lb, lm_lens, lb_lens in train_loader:
            lm = lm.to(device)
            lb = lb.to(device)

            log_probs = model(lm)                          # (B, T, V)
            log_probs = log_probs.permute(1, 0, 2)        # (T, B, V) for CTC

            loss = ctc_loss(log_probs, lb, lm_lens, lb_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        if epoch % args.eval_every == 0:
            wer = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | WER {wer:.2%} | {elapsed:.1f}s")
            if wer < best_wer:
                best_wer = wer
                ckpt_path = os.path.join(args.output_dir, "silentassist_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "wer": wer,
                }, ckpt_path)
                print(f"  Saved best checkpoint -> {ckpt_path}")
        else:
            print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | {elapsed:.1f}s")

    print(f"\nTraining complete. Best WER: {best_wer:.2%}")


def evaluate(model, loader, device) -> float:
    """Compute WER on a DataLoader using greedy decoding."""
    from jiwer import wer as compute_wer
    from silentassist.vit_model import ctc_greedy_decode, VOCAB

    model.eval()
    all_hyps, all_refs = [], []

    with torch.no_grad():
        for lm, lb, lm_lens, lb_lens in loader:
            lm        = lm.to(device)
            log_probs = model(lm)
            hyps      = ctc_greedy_decode(log_probs)
            all_hyps.extend(hyps)

            for i, length in enumerate(lb_lens):
                ref = "".join(VOCAB[c] for c in lb[i, :length].tolist())
                all_refs.append(ref)

    return compute_wer(all_refs, all_hyps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SilentAssist ViT")
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int,   default=5)
    args = parser.parse_args()
    train(args)
