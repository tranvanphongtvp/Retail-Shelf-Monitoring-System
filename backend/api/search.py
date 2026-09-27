import cv2
import numpy as np
import torch
from collections import Counter

from models import load_embedding_model
from .config import Config


# ============================== RECOGNIZER ==============================

class ProductRecognizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = load_embedding_model(cfg.EMBED_CKPT, device=cfg.device)
        import faiss
        self.index = faiss.read_index(cfg.FAISS_INDEX_PATH)
        self.labels = np.load(cfg.FAISS_LABELS_PATH, allow_pickle=True)

    def _preprocess(self, crop_rgb):
        img = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    @torch.no_grad()
    def recognize(self, crop_rgb):
        """
        Majority voting trên top-K kết quả từ FAISS.

        Thay vì chỉ lấy top-1 (dễ bị nhiễu do gallery có label sai),
        lấy top-K rồi:
          1. Lọc chỉ giữ các kết quả có score >= RECOGNIZE_THRESHOLD
          2. Majority vote: label nào xuất hiện nhiều nhất trong top-K
          3. Reliability = score cao nhất của label thắng vote

        Nếu không có kết quả nào đạt threshold → trả về top-1 với score
        gốc (sẽ bị loại bởi main.py vì < threshold).
        """
        x = self._preprocess(crop_rgb).unsqueeze(0).to(self.cfg.device)
        emb = self.model(x).cpu().numpy().astype(np.float32)

        top_k = self.cfg.TOP_K
        scores, indices = self.index.search(emb, top_k)
        scores_flat = scores[0]
        indices_flat = indices[0]

        # Lấy label và score cho từng kết quả top-K
        top_labels = [str(self.labels[idx]) for idx in indices_flat]
        top_scores = [float(s) for s in scores_flat]

        # Lọc chỉ giữ các kết quả đạt threshold
        valid = [(label, score) for label, score in zip(top_labels, top_scores)
                 if score >= self.cfg.RECOGNIZE_THRESHOLD]

        if not valid:
            # Không có kết quả nào đạt threshold → trả về top-1 gốc
            # (sẽ bị main.py loại vì reliability < RECOGNIZE_THRESHOLD)
            return top_labels[0], top_scores[0]

        # Majority vote: đếm label xuất hiện nhiều nhất
        valid_labels = [label for label, _ in valid]
        label_counts = Counter(valid_labels)
        best_label = label_counts.most_common(1)[0][0]

        # Reliability = score cao nhất của label thắng vote
        best_score = max(score for label, score in valid if label == best_label)

        return best_label, best_score