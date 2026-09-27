import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

EMBED_DIM = 256

class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, pretrained=False):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.fc(feat)
        return F.normalize(emb, p=2, dim=1)

def load_embedding_model(ckpt_path, device='cpu'):
    model = EmbeddingNet(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model.to(device)


# ============================== DETECTION (RetinaNet) ==============================
#EM-Merger vẫn giữ lại để xử lý sản phẩm xếp sát nhau, nhưng chỉ weight theo hard score.

def build_detector(num_classes=1, pretrained=True):
    """RetinaNet v1 — khớp notebook fine-tune và checkpoints/retinanet_detector.pth."""
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = retinanet_resnet50_fpn(weights=weights)
    in_features = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_features, num_anchors, num_classes=num_classes
    )
    return model


def load_detector_checkpoint(model, ckpt_path, device="cpu"):
    """
    Checkpoint cũ có thể chứa key của iou_tower (từ lúc còn Soft-IoU) -- dùng
    strict=False để bỏ qua có chủ đích, đồng thời in ra để tự xác nhận không có
    key quan trọng (backbone/cls/box) nào bị thiếu do lệch kiến trúc.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected_non_iou = [k for k in unexpected if "iou_tower" not in k]
    missing_important = [k for k in missing if "iou_tower" not in k]

    print(f"Missing keys (expected empty): {missing_important}")
    print(f"Unexpected keys outside iou_tower (expected empty): {unexpected_non_iou}")
    if missing_important or unexpected_non_iou:
        print("WARNING: keys mismatch outside iou_tower scope -- check architecture!")

    return model


def em_merger(boxes, hard_scores, dist_thresh=0.3):
    """Gộp box trùng của cùng 1 sản phẩm xếp sát nhau, weight hoàn toàn theo hard_scores."""
    if len(boxes) == 0:
        return boxes, hard_scores

    conf = hard_scores
    centers = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], dim=1)
    sizes = torch.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], dim=1)

    order = torch.argsort(conf, descending=True)
    cluster_id = -torch.ones(len(boxes), dtype=torch.long)
    next_id = 0
    for i in order.tolist():
        assigned = False
        for c in range(next_id):
            members = (cluster_id == c).nonzero(as_tuple=True)[0]
            c_center = (centers[members] * conf[members, None]).sum(0) / conf[members].sum()
            norm_dist = torch.norm((centers[i] - c_center) / sizes[i].clamp(min=1e-3))
            if norm_dist < dist_thresh:
                cluster_id[i] = c
                assigned = True
                break
        if not assigned:
            cluster_id[i] = next_id
            next_id += 1

    merged_boxes, merged_scores = [], []
    for c in range(next_id):
        members = (cluster_id == c).nonzero(as_tuple=True)[0]
        w = conf[members]
        merged_boxes.append((boxes[members] * w[:, None]).sum(0) / w.sum())
        merged_scores.append(w.max())
    return torch.stack(merged_boxes), torch.stack(merged_scores)