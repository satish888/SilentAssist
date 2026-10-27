"""
tts_output.py
Text-to-Speech output via Piper TTS (offline neural TTS)
SilentAssist — Silent Speech Reconstruction Pipeline

Piper TTS: https://github.com/rhasspy/piper
Target latency: < 150 ms on Raspberry Pi 5

Reference:
    Namballa, S. (2027). SilentAssist: Silent Speech Reconstruction for
    Non-Verbal Quadriplegics Using Vision Transformers and LLM-Augmented
    Language Prior. ANTIC 2027, Springer CCIS.
"""

import subprocess
import tempfile
import time
import os
from pathlib import Path
from typing import Optional, Dict


class PiperTTS:
    """
    Wrapper around the Piper offline TTS binary.

    Install Piper:
        pip install piper-tts
        python -m piper --download-dir ./voices en_US-lessac-medium

    Or download binary: https://github.com/rhasspy/piper/releases
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        piper_binary: str = "piper",
        output_dir: Optional[str] = None,
    ):
        self.model_path   = model_path or self._find_default_model()
        self.piper_binary = piper_binary
        self.output_dir   = output_dir or tempfile.gettempdir()

    def _find_default_model(self) -> Optional[str]:
        """Look for a Piper model in common locations."""
        candidates = [
            Path.home() / "voices" / "en_US-lessac-medium.onnx",
            Path("./voices/en_US-lessac-medium.onnx"),
            Path("/usr/share/piper-voices/en_US-lessac-medium.onnx"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def synthesise(
        self,
        text: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Synthesise speech for the given text.

        Args:
            text:        Input text to speak.
            output_path: Path to write WAV file. If None, uses a temp file.

        Returns:
            dict with keys: output_path, latency_ms, success
        """
        if output_path is None:
            output_path = os.path.join(
                self.output_dir, f"silentassist_{int(time.time()*1000)}.wav"
            )

        if self.model_path is None or not Path(self.piper_binary).exists() and \
                subprocess.run(["which", self.piper_binary],
                               capture_output=True).returncode != 0:
            return {
                "output_path": None,
                "latency_ms":  0.0,
                "success":     False,
                "error":       "Piper binary or model not found. "
                               "See: https://github.com/rhasspy/piper",
            }

        cmd = [
            self.piper_binary,
            "--model",  self.model_path,
            "--output_file", output_path,
        ]

        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
                timeout=5.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            if result.returncode == 0:
                return {
                    "output_path": output_path,
                    "latency_ms":  round(latency_ms, 2),
                    "success":     True,
                }
            else:
                return {
                    "output_path": None,
                    "latency_ms":  round(latency_ms, 2),
                    "success":     False,
                    "error":       result.stderr.decode(),
                }
        except subprocess.TimeoutExpired:
            return {
                "output_path": None,
                "latency_ms":  5000.0,
                "success":     False,
                "error":       "Piper TTS timed out (>5s)",
            }

    def timed_synthesise(
        self,
        text: str = "I need water",
        runs: int = 10,
    ) -> Dict:
        """Run TTS multiple times and report mean latency."""
        times   = []
        success = False
        for _ in range(runs):
            res = self.synthesise(text)
            if res["success"]:
                times.append(res["latency_ms"])
                success = True

        import statistics
        return {
            "tts_ms":  round(statistics.mean(times), 2) if times else None,
            "success": success,
            "runs":    len(times),
        }
