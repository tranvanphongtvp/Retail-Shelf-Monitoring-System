import cv2
import numpy as np
import torch

from models import build_detector, load_detector_checkpoint, em_merger
from .config import Config


# ============================== DETECTOR ==============================

class ProductDetector:
    def __init__(self, cfg: Config):
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