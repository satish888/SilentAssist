"""
landmark_extraction.py
Algorithm 1: Landmark Extraction and Normalisation
SilentAssist — Silent Speech Reconstruction Pipeline

Reference:
    Namballa, S. (2027). SilentAssist: Silent Speech Reconstruction for
    Non-Verbal Quadriplegics Using Vision Transformers and LLM-Augmented
    Language Prior. ANTIC 2027, Springer CCIS.
"""

import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from typing import List, Optional, Tuple

# 57 landmark indices: lip (outer + inner), jaw, cheek keypoints
# Selected from MediaPipe 478-point face mesh
LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
             375, 321, 405, 314, 17, 84, 181, 91, 146]
LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
             324, 318, 402, 317, 14, 87, 178, 88, 95]
JAW_LINE  = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378,
             379, 365, 397, 288, 361, 323, 454]

LANDMARK_INDICES = list(dict.fromkeys(LIP_OUTER + LIP_INNER + JAW_LINE))[:57]
NOSE_TIP_IDX     = 1      # MediaPipe nose tip
LEFT_EYE_IDX     = 33     # outer canthus
RIGHT_EYE_IDX    = 263    # outer canthus

LANDMARK_DIM = 57 * 3     # 171


def extract_landmarks_from_video(
    video_path: str,
    fps_target: int = 30,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract and normalise facial landmarks from a video file.

    Args:
        video_path:  Path to the input video file.
        fps_target:  Target sampling rate (frames per second).
        max_frames:  Cap on number of frames (None = no cap).

    Returns:
        landmarks:   Float32 array of shape (T, 171) — normalised.
        valid_idx:   Frame indices where MediaPipe succeeded.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or fps_target
    skip      = max(1, round(src_fps / fps_target))

    landmarks_list: List[np.ndarray] = []
    valid_idx:      List[int]        = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            result = _process_frame(face_mesh, frame)
            if result is not None:
                landmarks_list.append(result)
                valid_idx.append(frame_idx)
            if max_frames and len(landmarks_list) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    face_mesh.close()

    if not landmarks_list:
        return np.zeros((0, LANDMARK_DIM), dtype=np.float32), []

    return np.stack(landmarks_list, axis=0), valid_idx


def _process_frame(face_mesh, frame: np.ndarray) -> Optional[np.ndarray]:
    """Run MediaPipe on one BGR frame; return normalised 171-d vector or None."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm   = results.multi_face_landmarks[0].landmark
    pts  = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in range(len(lm))],
                    dtype=np.float32)       # (478, 3)

    # Select 57 landmarks
    selected = pts[LANDMARK_INDICES]        # (57, 3)
    x_t      = selected.flatten()          # (171,)

    # Translation normalisation — subtract nose tip
    nose = pts[NOSE_TIP_IDX]
    x_t  = x_t - np.tile(nose, 57)

    # Scale normalisation — divide by inter-ocular distance
    left_eye  = pts[LEFT_EYE_IDX]
    right_eye = pts[RIGHT_EYE_IDX]
    d_t       = np.linalg.norm(left_eye - right_eye) + 1e-6
    x_t       = x_t / d_t

    return x_t.astype(np.float32)


def extract_landmarks_from_frames(
    frames: List[np.ndarray],
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract landmarks from a pre-loaded list of BGR frames.
    Useful when video is already decoded in memory.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    landmarks_list: List[np.ndarray] = []
    valid_idx:      List[int]        = []

    for idx, frame in enumerate(frames):
        result = _process_frame(face_mesh, frame)
        if result is not None:
            landmarks_list.append(result)
            valid_idx.append(idx)

    face_mesh.close()

    if not landmarks_list:
        return np.zeros((0, LANDMARK_DIM), dtype=np.float32), []
    return np.stack(landmarks_list, axis=0), valid_idx
