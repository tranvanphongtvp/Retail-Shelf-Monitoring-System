"""
pipeline.py

Pipeline đầy đủ theo Algorithm 1 của bài báo "Shelf Management":

    1. detectShelfRows(image)      -> danh sách y-coordinate của các hàng kệ
    2. DetectProducts(image)       -> bounding box ứng viên (RetinaNet + EM-Merger,
                                       KHÔNG dùng Soft-IoU -- xem ghi chú bên dưới)
    3. Lọc bounding box "giả" (price tag / tờ khuyến mãi) dựa trên độ chồng lấn
       với các đường shelf row -- 1 cái tag thường NẰM TRÊN đường kẻ ngang giữa
       2 hàng kệ, nên phần lớn diện tích của nó chồng lên dải mỏng quanh đường
       shelf row.
    4. Với mỗi box còn lại: crop -> extractEmbeddings -> search(gallery) -> EAN + reliability
    5. assignLocation: row (so với shelf rows), column (trái->phải trong hàng),
       subrow (chồng chất trong cùng 1 cột)
"""

import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import faiss

from models import (
    load_embedding_model,
    build_detector,
    load_detector_checkpoint,
    em_merger,
)


# ============================== CONFIG ==============================

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- bước 1: shelf row ---
    SHELF_ROW_CKPT = "checkpoints/shelf_row_net_best.pth"
    IMG_H, IMG_W = 512, 384
    HOUGH_H, HOUGH_W = 64, 48
    NTHETA = 11
    THETA_WINDOW_DEG = 10
    NRHO = 100

    # --- bước 2: detection (không còn Soft-IoU) ---
    DETECTOR_CKPT = "checkpoints/retinanet_detector.pth"
    DET_IMG_SIZE = 640
    SCORE_THRESHOLD = 0.35
    EM_MERGER_DIST_THRESH = 0.3
    SHELF_TAG_THRESHOLD = 0.6      # tỉ lệ diện tích box chồng lên dải shelf row để coi là "tag", không phải sản phẩm
    SHELF_BAND_FRAC = 0.015        # bề dày dải quanh mỗi đường shelf row, tính theo % chiều cao ảnh

    # --- bước 3: recognition ---
    EMBED_CKPT = "checkpoints/embedding_mobilenetv3.pth"
    FAISS_INDEX_PATH = "faiss_index/shape_gallery.index"
    FAISS_LABELS_PATH = "faiss_index/gallery_labels.npy"
    RECOGNIZE_THRESHOLD = 0.7
    TOP_K = 1


# ======================= BƯỚC 1: SHELF ROW MODEL =======================

def build_hough_geometry(H, W, thetas_deg, nrho):
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = xs.astype(np.float32).ravel()
    ys = ys.astype(np.float32).ravel()
    thetas = np.deg2rad(thetas_deg)
    ntheta = len(thetas)
    rhos = np.stack([xs * np.cos(t) + ys * np.sin(t) for t in thetas], axis=0)
    rho_min, rho_max = rhos.min(), rhos.max()
    bins = np.linspace(rho_min, rho_max, nrho + 1)
    vote = np.zeros((ntheta, nrho, H * W), dtype=np.float32)
    for ti in range(ntheta):
        idx = np.clip(np.digitize(rhos[ti], bins) - 1, 0, nrho - 1)
        vote[ti, idx, np.arange(H * W)] = 1.0
    vote = vote.reshape(ntheta * nrho, H * W)
    counts = vote.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1
    vote = vote / counts
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_to_y_norm = np.clip(bin_centers, 0, H - 1) / (H - 1)
    return torch.from_numpy(vote), bins, bin_to_y_norm


class ResNetFPNBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.body = resnet_fpn_backbone('resnet50', pretrained=pretrained, trainable_layers=3)

    def forward(self, x):
        return self.body(x)


class FeatureFuser(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.reduce = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feats):
        size = feats['0'].shape[-2:]
        fused = feats['0']
        for k in ['1', '2', '3']:
            fused = fused + F.interpolate(feats[k], size=size, mode='bilinear', align_corners=False)
        return self.reduce(fused)


class DeepHoughTransform(nn.Module):
    def __init__(self, in_channels, vote_matrix, ntheta, nrho, hough_h, hough_w, mid_channels=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((hough_h, hough_w))
        self.reduce = nn.Conv2d(in_channels, mid_channels, 1)
        self.register_buffer('vote_matrix', vote_matrix)
        self.ntheta, self.nrho = ntheta, nrho
        self.param_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, 1, 1),
        )

    def forward(self, feat):
        feat = self.reduce(self.pool(feat))
        B, C, H, W = feat.shape
        flat = feat.reshape(B, C, H * W)
        hough = torch.matmul(flat, self.vote_matrix.t())
        hough = hough.view(B, C, self.ntheta, self.nrho)
        hough = self.param_conv(hough)
        collapsed = hough.max(dim=2).values
        return self.head(collapsed).squeeze(1)


class ShelfRowNet(nn.Module):
    def __init__(self, vote_matrix, ntheta, nrho, hough_h, hough_w, pretrained=False):
        super().__init__()
        self.backbone = ResNetFPNBackbone(pretrained)
        self.fuser = FeatureFuser(256)
        self.dht = DeepHoughTransform(256, vote_matrix, ntheta, nrho, hough_h, hough_w)

    def forward(self, x):
        return self.dht(self.fuser(self.backbone(x)))


class ShelfRowDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        thetas_deg = np.linspace(90 - cfg.THETA_WINDOW_DEG, 90 + cfg.THETA_WINDOW_DEG, cfg.NTHETA)
        vote_matrix, _, self.bin_to_y_norm = build_hough_geometry(
            cfg.HOUGH_H, cfg.HOUGH_W, thetas_deg, cfg.NRHO
        )
        self.model = ShelfRowNet(vote_matrix, cfg.NTHETA, cfg.NRHO, cfg.HOUGH_H, cfg.HOUGH_W)
        self.model.load_state_dict(torch.load(cfg.SHELF_ROW_CKPT, map_location=cfg.device))
        self.model.eval().to(cfg.device)

    def _preprocess(self, image_rgb):
        img = cv2.resize(image_rgb, (self.cfg.IMG_W, self.cfg.IMG_H)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    @torch.no_grad()
    def detect(self, image_rgb, min_dist=3, threshold=0.2):
        """Trả về list các y_norm (0..1) đã sort tăng dần -- áp dụng trực tiếp lên ảnh GỐC
        vì resize không giữ tỉ lệ khung hình chỉ co giãn đều theo trục dọc, nên vị trí
        y tương đối (%) không đổi."""
        x = self._preprocess(image_rgb).unsqueeze(0).to(self.cfg.device)
        logits = self.model(x)[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        cand = [i for i in range(1, len(probs) - 1)
                if probs[i] > threshold and probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1]]
        cand = sorted(cand, key=lambda i: -probs[i])
        kept = []
        for c in cand:
            if all(abs(c - k) >= min_dist for k in kept):
                kept.append(c)
        return sorted(self.bin_to_y_norm[k] for k in kept)


# ======================= BƯỚC 2: DETECTION (RetinaNet, không Soft-IoU) =======================

class ProductDetector:
    """
    Dùng build_detector/load_detector_checkpoint/em_merger từ models.py.
    Khác bản trước: KHÔNG có SoftIoUHead, không cần lấy feature FPN riêng để
    ROIAlign -- chỉ dùng score có sẵn từ classification_head (hard score) của
    RetinaNet, đưa thẳng vào em_merger để gộp box trùng.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        model = build_detector(num_classes=1, pretrained=False)
        self.model = load_detector_checkpoint(model, cfg.DETECTOR_CKPT, device=cfg.device)
        self.model.eval().to(cfg.device)

    def _preprocess(self, image_rgb):
        """Resize giữ tỉ lệ (letterbox góc trên-trái), trả về tensor + scale để quy đổi ngược box."""
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
        """Trả về boxes (x1,y1,x2,y2) theo tọa độ ẢNH GỐC + scores, sau EM-Merger (chỉ hard score)."""
        img_t, scale = self._preprocess(image_rgb)
        img_t = img_t.unsqueeze(0).to(self.cfg.device)

        det = self.model(img_t)[0]   # RetinaNet tự áp NMS chuẩn bên trong, dựa trên hard score

        keep = det['scores'] > self.cfg.SCORE_THRESHOLD
        boxes, hard_scores = det['boxes'][keep], det['scores'][keep]
        if len(boxes) == 0:
            return torch.zeros((0, 4)), torch.zeros((0,))

        merged_boxes, merged_scores = em_merger(
            boxes.cpu(), hard_scores.cpu(),
            dist_thresh=self.cfg.EM_MERGER_DIST_THRESH
        )
        # quy đổi box từ tọa độ canvas (đã letterbox+scale) về tọa độ ảnh gốc
        merged_boxes = merged_boxes / scale
        return merged_boxes, merged_scores


# =================== LỌC BOX GIẢ (PRICE TAG) DỰA TRÊN SHELF ROW ===================

def filter_tag_boxes(boxes, shelf_rows_y_norm, img_h, threshold, band_frac):
    """
    Với mỗi shelf row (đường ngang tại y = y_norm * img_h), coi nó có "bề dày" là
    band_frac * img_h (dải mỏng phía trên/dưới đường kẻ, vì tag/nhãn giá thường
    dán sát mép kệ ngay trên đường phân cách). Box nào có tỉ lệ diện tích chồng lên
    dải đó > threshold thì coi là false positive (tag/vật không phải sản phẩm) -> loại.
    Trả về mask (True = giữ lại, là sản phẩm thật).
    """
    if len(boxes) == 0:
        return torch.zeros((0,), dtype=torch.bool)

    band_half = band_frac * img_h
    keep_mask = torch.ones(len(boxes), dtype=torch.bool)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        if box_area <= 0:
            keep_mask[i] = False
            continue
        overlap_total = 0.0
        for y_norm in shelf_rows_y_norm:
            band_y1 = y_norm * img_h - band_half
            band_y2 = y_norm * img_h + band_half
            inter_y1, inter_y2 = max(y1, band_y1), min(y2, band_y2)
            if inter_y2 > inter_y1:
                overlap_total += (x2 - x1) * (inter_y2 - inter_y1)
        if overlap_total / box_area > threshold:
            keep_mask[i] = False
    return keep_mask


# ======================= BƯỚC 3: RECOGNITION =======================

class ProductRecognizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = load_embedding_model(cfg.EMBED_CKPT, device=cfg.device)
        self.index = faiss.read_index(cfg.FAISS_INDEX_PATH)
        self.labels = np.load(cfg.FAISS_LABELS_PATH, allow_pickle=True)

    def _preprocess(self, crop_rgb):
        img = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    @torch.no_grad()
    def recognize(self, crop_rgb):
        x = self._preprocess(crop_rgb).unsqueeze(0).to(self.cfg.device)
        emb = self.model(x).cpu().numpy().astype(np.float32)
        scores, indices = self.index.search(emb, self.cfg.TOP_K)
        best_label = self.labels[indices[0][0]]
        best_score = float(scores[0][0])
        return best_label, best_score


# ======================= PRODUCT LOCALIZATION (Section 3.3) =======================

def assign_row(box_center_y, shelf_rows_y_sorted_pixel):
    """shelf_rows_y_sorted_pixel: y (pixel, tăng dần, trên -> dưới). Row 1 = trên cùng."""
    row = 1
    for i, y in enumerate(shelf_rows_y_sorted_pixel):
        if box_center_y > y:
            row = i + 2
        else:
            break
    return row


def assign_columns_and_subrows(boxes_in_row):
    """
    boxes_in_row: list dict {"box": (x1,y1,x2,y2), ...}
    Column: sort theo center_x trái->phải.
    Subrow: đúng thủ tục AssignSubRow của paper -- box A "đè lên" box B nếu
    center_x của A < x2 của B VÀ center_y của A < y1 của B (A nằm phía trên B).
    """
    def center(b):
        x1, y1, x2, y2 = b["box"]
        return (x1 + x2) / 2, (y1 + y2) / 2

    boxes_sorted = sorted(boxes_in_row, key=lambda b: center(b)[0])
    for i, b in enumerate(boxes_sorted):
        b["column"] = i + 1

    for b in boxes_sorted:
        b["subrow"] = 1
        cx, cy = center(b)
        for other in boxes_sorted:
            if other is b:
                continue
            _, _, ox2, oy1 = other["box"]
            if ox2 > cx and oy1 > cy:
                b["subrow"] += 1
    return boxes_sorted


# ============================== MAIN PIPELINE ==============================

def process_image(image_path, cfg: Config):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    print("Đang detect shelf rows ...")
    shelf_detector = ShelfRowDetector(cfg)
    shelf_rows_y_norm = shelf_detector.detect(image_rgb)   # list, sorted, top->bottom
    print(f"  -> tìm thấy {len(shelf_rows_y_norm)} hàng kệ: {[round(y, 3) for y in shelf_rows_y_norm]}")

    print("Đang detect sản phẩm (RetinaNet + EM-Merger, hard score) ...")
    product_detector = ProductDetector(cfg)
    boxes, scores = product_detector.detect(image_rgb)
    print(f"  -> {len(boxes)} box trước khi lọc tag")

    keep_mask = filter_tag_boxes(boxes, shelf_rows_y_norm, img_h,
                                  cfg.SHELF_TAG_THRESHOLD, cfg.SHELF_BAND_FRAC)
    boxes, scores = boxes[keep_mask], scores[keep_mask]
    print(f"  -> {len(boxes)} box sau khi loại tag/nhãn giá dựa trên shelf row")

    print("Đang nhận diện từng sản phẩm ...")
    recognizer = ProductRecognizer(cfg)
    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_w), min(y2, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_rgb[y1:y2, x1:x2]
        label, reliability = recognizer.recognize(crop)

        if reliability > cfg.RECOGNIZE_THRESHOLD:
            detections.append({
                "box": (x1, y1, x2, y2),
                "det_score": float(score),
                "EAN": str(label),
                "reliability": reliability,
            })

    print(f"  -> {len(detections)} sản phẩm nhận diện thành công (reliability > {cfg.RECOGNIZE_THRESHOLD})")

    print("Đang gán vị trí (row/column/subrow) ...")
    shelf_rows_y_pixel = [y * img_h for y in shelf_rows_y_norm]
    rows_map = {}
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cy = (y1 + y2) / 2
        row = assign_row(cy, shelf_rows_y_pixel)
        d["row"] = row
        rows_map.setdefault(row, []).append(d)

    final_detections = []
    for row, boxes_in_row in rows_map.items():
        placed = assign_columns_and_subrows(boxes_in_row)
        final_detections.extend(placed)

    return final_detections, shelf_rows_y_norm


def print_results(detections):
    print("\n=== KẾT QUẢ ===")
    for d in detections:
        print(f"POS: Row {d['row']}, Col {d['column']}, SubRow {d['subrow']} "
              f"| EAN: {d['EAN']} | reliability={d['reliability']:.3f} | box={d['box']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    cfg = Config()
    detections, shelf_rows = process_image(args.image, cfg)
    print_results(detections)


if __name__ == "__main__":
    main()