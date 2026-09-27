"""
debug_recognition.py

Script chan doan loi "cung 1 san pham nhung ra nhieu nhan khac nhau".
Chay detect + recognize tren anh ke hang, in ra top-5 (label, score) cho moi crop
de xac dinh nguyen nhan goc re.

Cach dung (tu thu muc goc du an):
    .venv\\Scripts\\python.exe src/debug_recognition.py --image data/000b05c7-2525-4dc7-9235-467af99969c8.jpg

Hoac chay tat ca anh trong thu muc:
    .venv\\Scripts\\python.exe src/debug_recognition.py --image_dir data/
"""

import sys
import argparse
import os
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import faiss

# Them src vao path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models import load_embedding_model, build_detector, load_detector_checkpoint, em_merger


# ============================== CONFIG ==============================

class DebugConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DETECTOR_CKPT = str(_ROOT / "checkpoints" / "retinanet_detector.pth")
    DET_IMG_SIZE = 640
    SCORE_THRESHOLD = 0.35
    EM_MERGER_DIST_THRESH = 0.3

    EMBED_CKPT = str(_ROOT / "checkpoints" / "embedding_mobilenetv3.pth")
    FAISS_INDEX_PATH = str(_ROOT / "faiss_index" / "shape_gallery.index")
    FAISS_LABELS_PATH = str(_ROOT / "faiss_index" / "gallery_labels.npy")

    # Debug dung TOP_K = 5 de phan tich
    TOP_K = 5


# ============================== DETECT ==============================

class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        model = build_detector(num_classes=1, pretrained=False)
        self.model = load_detector_checkpoint(model, cfg.DETECTOR_CKPT, device=cfg.device)
        self.model.eval().to(cfg.device)

    def _preprocess(self, image_rgb):
        h0, w0 = image_rgb.shape[:2]
        scale = self.cfg.DET_IMG_SIZE / max(h0, w0)
        new_h, new_w = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(image_rgb, (new_w, new_h))
        canvas = np.zeros((self.cfg.DET_IMG_SIZE, self.cfg.DET_IMG_SIZE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        img_t = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        return img_t, scale

    @torch.no_grad()
    def detect(self, image_rgb):
        img_t, scale = self._preprocess(image_rgb)
        img_t = img_t.unsqueeze(0).to(self.cfg.device)
        det = self.model(img_t)[0]
        keep = det["scores"] > self.cfg.SCORE_THRESHOLD
        boxes, hard_scores = det["boxes"][keep], det["scores"][keep]
        if len(boxes) == 0:
            return torch.zeros((0, 4)), torch.zeros((0,))
        merged_boxes, merged_scores = em_merger(
            boxes.cpu(), hard_scores.cpu(), dist_thresh=self.cfg.EM_MERGER_DIST_THRESH
        )
        merged_boxes = merged_boxes / scale
        return merged_boxes, merged_scores


# ============================== RECOGNIZE (DEBUG) ==============================

class DebugRecognizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = load_embedding_model(cfg.EMBED_CKPT, device=cfg.device)
        self.index = faiss.read_index(cfg.FAISS_INDEX_PATH)
        self.labels = np.load(cfg.FAISS_LABELS_PATH, allow_pickle=True)

    def _preprocess(self, crop_rgb):
        img = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    @torch.no_grad()
    def recognize_top_k(self, crop_rgb, top_k=5):
        """Return list[(label, score)] top-K."""
        x = self._preprocess(crop_rgb).unsqueeze(0).to(self.cfg.device)
        emb = self.model(x).cpu().numpy().astype(np.float32)
        scores, indices = self.index.search(emb, top_k)
        results = []
        for i in range(top_k):
            label = str(self.labels[indices[0][i]])
            score = float(scores[0][i])
            results.append((label, score))
        return results


# ============================== ANALYSIS ==============================

def analyze_top_k(results):
    """Classify top-5 results for diagnosis."""
    if not results:
        return "NO_RESULT"

    top1_label, top1_score = results[0]
    unique_labels = set(label for label, _ in results)
    score_range = results[0][1] - results[-1][1]

    if top1_score < 0.5:
        return "VERY_LOW_CONFIDENCE"
    elif top1_score < 0.7:
        return "LOW_CONFIDENCE"
    elif len(unique_labels) >= 4 and score_range < 0.15:
        return "AMBIGUOUS_MANY_LABELS"
    elif len(unique_labels) >= 3 and score_range < 0.10:
        return "AMBIGUOUS_TIGHT_CLUSTER"
    elif len(unique_labels) == 1:
        return "STRONG_CONSENSUS"
    elif results[0][1] - results[1][1] > 0.1:
        return "CLEAR_TOP1"
    else:
        return "MODERATE"


def process_image(image_path, detector, recognizer, cfg):
    """Run detect + recognize on 1 image, print debug results."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  [ERROR] Cannot read image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    boxes, scores = detector.detect(image_rgb)
    total_detected = len(boxes)
    print(f"\n  Image: {os.path.basename(image_path)} ({img_w}x{img_h})")
    print(f"  Total boxes detected: {total_detected}")
    print(f"  {'='*80}")

    diagnostics = Counter()
    all_top1_labels = []
    score_distribution = []

    for idx, (box, det_score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_w), min(y2, img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_rgb[y1:y2, x1:x2]
        top_k = recognizer.recognize_top_k(crop, top_k=cfg.TOP_K)
        diagnosis = analyze_top_k(top_k)
        diagnostics[diagnosis] += 1
        all_top1_labels.append(top_k[0][0])
        score_distribution.append(top_k[0][1])

        print(f"\n  Box #{idx+1}: [{x1},{y1},{x2},{y2}] det_score={det_score:.3f}")
        print(f"  Diagnosis: {diagnosis}")
        for rank, (label, score) in enumerate(top_k):
            marker = "  << TOP-1" if rank == 0 else ""
            print(f"    #{rank+1}: {label:>20s}  score={score:.4f}{marker}")

    # Summary report
    print(f"\n  {'='*80}")
    print(f"  SUMMARY REPORT")
    print(f"  {'='*80}")
    print(f"  Total boxes: {total_detected}")
    print(f"  Unique labels (top-1): {len(set(all_top1_labels))}")

    if score_distribution:
        print(f"  Score top-1 -- min: {min(score_distribution):.4f}, "
              f"max: {max(score_distribution):.4f}, "
              f"mean: {np.mean(score_distribution):.4f}")

    print(f"\n  Diagnosis breakdown:")
    for diag, count in diagnostics.most_common():
        print(f"    {diag}: {count} boxes")

    if all_top1_labels:
        label_counts = Counter(all_top1_labels)
        print(f"\n  Label distribution (top-1):")
        for label, count in label_counts.most_common(15):
            print(f"    {label}: {count} times")

    # Conclusion
    print(f"\n  CONCLUSION:")
    ambiguous_count = diagnostics.get("AMBIGUOUS_MANY_LABELS", 0) + diagnostics.get("AMBIGUOUS_TIGHT_CLUSTER", 0)
    low_conf_count = diagnostics.get("VERY_LOW_CONFIDENCE", 0) + diagnostics.get("LOW_CONFIDENCE", 0)

    if ambiguous_count > total_detected * 0.3:
        print("    -> Many boxes have top-5 with many different labels & close scores.")
        print("    -> ROOT CAUSE: Gallery has noisy labels OR embedding model not discriminative enough.")
        print("    -> RECOMMENDATION: Run audit_gallery.py to check gallery quality.")
    elif low_conf_count > total_detected * 0.3:
        print("    -> Many boxes have very low scores.")
        print("    -> CAUSE: Products not in gallery, or poor crop quality.")
    elif len(set(all_top1_labels)) > total_detected * 0.7:
        print("    -> Too many unique labels compared to box count -- same product may get different labels.")
        print("    -> RECOMMENDATION: Increase RECOGNIZE_THRESHOLD and use majority voting.")
    else:
        print("    -> Recognition results look reasonable. Check further if needed.")


def main():
    parser = argparse.ArgumentParser(description="Debug recognition: print top-5 for each crop")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--max_images", type=int, default=3, help="Max images when using --image_dir")
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        args.image_dir = str(_ROOT / "data")
        print("No image specified, defaulting to data/ directory")

    cfg = DebugConfig()
    print(f"Device: {cfg.device}")
    print(f"FAISS index: {cfg.FAISS_INDEX_PATH}")

    # Check FAISS index type
    index = faiss.read_index(cfg.FAISS_INDEX_PATH)
    print(f"FAISS index type: {type(index).__name__}")
    print(f"FAISS index ntotal: {index.ntotal}")
    labels = np.load(cfg.FAISS_LABELS_PATH, allow_pickle=True)
    print(f"Unique labels in gallery: {len(set(labels))}")

    print("\nLoading models...")
    detector = Detector(cfg)
    recognizer = DebugRecognizer(cfg)
    print("Models loaded.\n")

    images = []
    if args.image:
        images.append(args.image)
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for f in sorted(os.listdir(args.image_dir)):
            if Path(f).suffix.lower() in exts:
                images.append(os.path.join(args.image_dir, f))
        images = images[:args.max_images]

    print(f"Will analyze {len(images)} images...\n")
    for img_path in images:
        process_image(img_path, detector, recognizer, cfg)
        print(f"\n{'#'*90}\n")


if __name__ == "__main__":
    main()
