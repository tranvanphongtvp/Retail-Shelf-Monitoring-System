from pathlib import Path

import torch

# backend/api/config.py -> project root (checkpoints/, faiss_index/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================== CONFIG ==============================

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DETECTOR_CKPT = str(PROJECT_ROOT / "checkpoints" / "retinanet_detector.pth")
    DET_IMG_SIZE = 640
    SCORE_THRESHOLD = 0.35
    EM_MERGER_DIST_THRESH = 0.3

    EMBED_CKPT = str(PROJECT_ROOT / "checkpoints" / "embedding_mobilenetv3.pth")
    FAISS_INDEX_PATH = str(PROJECT_ROOT / "faiss_index" / "shape_gallery.index")
    FAISS_LABELS_PATH = str(PROJECT_ROOT / "faiss_index" / "gallery_labels.npy")
    RECOGNIZE_THRESHOLD = 0.85
    TOP_K = 5


cfg = Config()
