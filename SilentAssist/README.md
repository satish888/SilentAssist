# SilentAssist

**Silent Speech Reconstruction for Non-Verbal Quadriplegics Using Vision Transformers and LLM-Augmented Language Prior**

> Published in: *ANTIC 2027*, Springer Communications in Computer and Information Science (CCIS), Scopus indexed.
> Author: Satish Namballa, Walmart Global Tech, Bengaluru, India

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Springer CCIS](https://img.shields.io/badge/Published-ANTIC%202027-orange.svg)]()

---

## Overview

SilentAssist is an open-source, real-time silent speech recognition pipeline designed for non-verbal quadriplegic patients with dysarthria. It runs entirely on commodity hardware (webcam + Raspberry Pi 5) with no cloud dependency, targeting AAC (Augmentative and Alternative Communication) applications.

**Pipeline:**
```
Video (30 fps)
    -> MediaPipe landmark extraction (57 lip/jaw/cheek landmarks)
    -> Vision Transformer + CTC beam search (n-best hypotheses)
    -> Phi-3 Mini LoRA reranking with clinical vocabulary
    -> Piper TTS speech output
```

**Target performance:**
- End-to-end latency: < 350 ms (Raspberry Pi 5)
- Operates offline, no internet required
- 4.1M parameter ViT (1.3 MB INT8 after ONNX quantisation)

---

## Repository Structure

```
SilentAssist/
├── silentassist/
│   ├── landmark_extraction.py   # Algorithm 1: MediaPipe + normalisation
│   ├── vit_model.py             # Algorithm 2: ViT encoder + CTC beam search
│   ├── llm_reranker.py          # Algorithm 3: Phi-3 Mini LoRA reranking
│   ├── tts_output.py            # Piper TTS wrapper
│   └── pipeline.py              # End-to-end pipeline runner
├── train/
│   ├── train_vit.py             # ViT training on landmark datasets
│   └── finetune_lora.py         # Phi-3 Mini LoRA fine-tuning
├── eval/
│   ├── evaluate_torgo.py        # Evaluation on TORGO dysarthric corpus
│   └── benchmark_latency.py     # Per-stage latency measurement
├── configs/
│   └── default.yaml             # All hyperparameters
├── paper/
│   └── ANTIC2027_SilentSpeech_Paper.pdf
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/satishnamballa/SilentAssist.git
cd SilentAssist
pip install -r requirements.txt
```

For Piper TTS:
```bash
pip install piper-tts
python -m piper --download-dir ./voices en_US-lessac-medium
```

---

## Quick Start

```python
from silentassist import SilentAssistPipeline

# Load from checkpoint
pipe = SilentAssistPipeline.from_pretrained("./checkpoints/silentassist_best.pt")

# Run on a video file
result = pipe.run("patient_utterance.mp4")
print(result["text"])         # recognised text
print(result["timings_ms"])   # per-stage latency breakdown
# result["audio_path"]        # path to Piper TTS WAV output
```

---

## Datasets

| Dataset | Access | Used for |
|---------|--------|---------|
| [TORGO](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) | Email torgo@cs.toronto.edu | Primary dysarthric evaluation |
| [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/) | Free download | ViT pre-training |
| [OuluVS2](http://www.oulu.fi/cmvs/node/41316) | Request form | Generalisation testing |

---

## Training

**Step 1: Extract landmarks from video files**
```bash
python -c "
from silentassist.landmark_extraction import extract_landmarks_from_video
import numpy as np
lm, _ = extract_landmarks_from_video('video.mp4')
np.save('video.npy', lm)
"
```

**Step 2: Train the ViT**
```bash
python train/train_vit.py --data_dir ./data --epochs 50 --batch_size 8
```

**Step 3: Fine-tune Phi-3 Mini LoRA (optional, requires GPU)**
```bash
python train/finetune_lora.py \
    --base_model microsoft/Phi-3-mini-4k-instruct \
    --output_dir ./lora_clinical
```

---

## Evaluation on TORGO

```bash
python eval/evaluate_torgo.py \
    --torgo_dir ./torgo \
    --checkpoint ./checkpoints/silentassist_best.pt \
    --split test
```

---

## Latency Benchmark

```bash
python eval/benchmark_latency.py
```

Measured on x86 CPU (Intel Core i7, single thread):

| Stage | Latency |
|-------|---------|
| MediaPipe landmark extraction | ~12 ms |
| ViT inference + CTC beam search | ~36 ms |
| LLM reranking | ~0.1 ms (n-gram) / ~85 ms (Phi-3) |
| Piper TTS | ~148 ms (Raspberry Pi 5 target) |
| **Total** | **~196 ms (x86) / <350 ms (RPi 5 target)** |

---

## Citation

If you use SilentAssist in your research, please cite:

```bibtex
@inproceedings{namballa2027silentassist,
  title     = {SilentAssist: Silent Speech Reconstruction for Non-Verbal
               Quadriplegics Using Vision Transformers and
               LLM-Augmented Language Prior},
  author    = {Namballa, Satish},
  booktitle = {Proceedings of ANTIC 2027},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2027}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
