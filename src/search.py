"""
search.py

Nhận 1 ảnh sản phẩm mới (đã crop, giống ảnh trong SHAPE), trích xuất embedding
bằng EmbeddingNet đã train, rồi search trong FAISS index để tìm SKU gần nhất.
Tương ứng đúng Algorithm 1 dòng 11-14 trong bài báo:

    cropped   <- image[b_box]
    embedding <- extractEmbeddings(cropped)
    EANs_list, reliability_list <- search(embedding, gallery)
    if reliability_list[0] > recognize_threshold: ... (nhận diện thành công)

Cách dùng:
    python search.py \
        --image path/to/product.jpg \
        --checkpoint checkpoints/embedding_mobilenetv3.pth \
        --index faiss_index/shape_gallery.index \
        --labels faiss_index/gallery_labels.npy \
        --top_k 5 \
        --threshold 0.7
"""

import argparse
import numpy as np
import torch
import faiss
from PIL import Image
from torchvision import transforms

from models import load_embedding_model, EMBED_DIM  # từ src/models.py đã viết trước đó

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_embedding(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    img_t = EVAL_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_t)  # đã L2-normalize sẵn trong forward() của EmbeddingNet
    return emb.cpu().numpy().astype(np.float32)


def search(embedding, index, labels, top_k=5):
    """
    index dùng IndexFlatIP -> score trả về chính là cosine similarity
    (vì embedding đã L2-normalize), giá trị trong [-1, 1], càng gần 1 càng giống.
    """
    scores, indices = index.search(embedding, top_k)
    scores, indices = scores[0], indices[0]
    matched_labels = labels[indices]
    return list(zip(matched_labels.tolist(), scores.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                         help="Đường dẫn ảnh sản phẩm cần nhận diện (đã crop)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/embedding_mobilenetv3.pth")
    parser.add_argument("--index", type=str, default="faiss_index/shape_gallery.index")
    parser.add_argument("--labels", type=str, default="faiss_index/gallery_labels.npy")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7,
                         help="recognize_threshold: dưới ngưỡng này coi là 'không nhận diện được' "
                              "(sản phẩm lạ / chưa có trong gallery), giống Algorithm 1 của paper.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Đang load model từ {args.checkpoint} ...")
    model = load_embedding_model(args.checkpoint, device=device)

    print(f"Đang load FAISS index từ {args.index} ...")
    index = faiss.read_index(args.index)
    labels = np.load(args.labels, allow_pickle=True)

    embedding = compute_embedding(model, args.image, device)
    results = search(embedding, index, labels, top_k=args.top_k)

    print(f"\nKết quả top-{args.top_k} cho ảnh: {args.image}")
    for rank, (label, score) in enumerate(results, start=1):
        print(f"  #{rank}: {label}  (score={score:.4f})")

    best_label, best_score = results[0]
    if best_score > args.threshold:
        print(f"\n=> Nhận diện thành công: {best_label} (reliability={best_score:.4f} > threshold={args.threshold})")
    else:
        print(f"\n=> KHÔNG đạt ngưỡng tin cậy (best={best_score:.4f} <= threshold={args.threshold})."
              f" Có thể là sản phẩm lạ hoặc chưa có trong gallery.")


if __name__ == "__main__":
    main()