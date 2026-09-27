"""
build_faiss_index.py

Build FAISS index từ gallery_embeddings.npz (đã trích xuất trên Kaggle ở bước
product recognition), dùng cho bước tra cứu sản phẩm (product recognition)
theo đúng cách bài báo Shelf Management chọn: FlatIP (exhaustive search,
cosine similarity vì vector đã L2-normalize) -- xem Table 6 của paper.

Cách dùng:
    python build_faiss_index.py \
        --gallery embeddings/gallery_embeddings.npz \
        --out_dir faiss_index

Sau khi chạy xong, faiss_index/ sẽ có:
    - shape_gallery.index   (FAISS index, dùng để search)
    - gallery_labels.npy    (nhãn SKU tương ứng với từng vector trong index)
    - gallery_paths.npy     (đường dẫn ảnh gốc, nếu có trong file .npz, để tra cứu ngược)
"""

import argparse
import os
import numpy as np
import faiss


def load_gallery(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["label_names"]
    paths = data["image_paths"] if "image_paths" in data.files else None
    return embeddings, labels, paths


def build_index(embeddings, index_type="flatip", nlist=4096):
    """
    index_type:
        - "flatip": exhaustive search, chính xác 100%, chậm hơn nhưng gallery
          cỡ vài chục nghìn ảnh vẫn rất nhanh (paper đo 0.41ms/query ở Table 6).
        - "ivfflat": nhanh hơn nữa khi gallery cực lớn, đánh đổi ~1-10% accuracy
          (xem Table 6 paper, nlist=4096/nprobe=4 mất ~19% accuracy, không khuyến
          khích trừ khi gallery của bạn quá lớn để dùng FlatIP).
    """
    dim = embeddings.shape[1]
    if index_type == "flatip":
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    elif index_type == "ivfflat":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
    else:
        raise ValueError(f"index_type không hỗ trợ: {index_type}")
    return index


def sanity_check(index, embeddings, labels, k=5, n_queries=20):
    """
    Kiểm tra nhanh: search chính các vector đã add vào index, top-1 phải luôn
    khớp với chính nó (self-match). Nếu không khớp -> có gì đó sai khi build index.
    """
    n = min(n_queries, embeddings.shape[0])
    sample_idx = np.random.choice(embeddings.shape[0], n, replace=False)
    query = embeddings[sample_idx]

    scores, indices = index.search(query, k)
    self_match = indices[:, 0] == sample_idx
    print(f"Sanity check: {self_match.sum()}/{n} self-match ở top-1 "
          f"(kỳ vọng gần {n}/{n} nếu index build đúng).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=str, default="embeddings/gallery_embeddings.npz",
                         help="Đường dẫn tới file gallery_embeddings.npz")
    parser.add_argument("--out_dir", type=str, default="faiss_index",
                         help="Thư mục lưu index + labels")
    parser.add_argument("--index_type", type=str, default="flatip",
                         choices=["flatip", "ivfflat"])
    parser.add_argument("--nlist", type=int, default=4096,
                         help="Chỉ dùng khi index_type=ivfflat")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Đang load gallery từ {args.gallery} ...")
    embeddings, labels, paths = load_gallery(args.gallery)
    print(f"gallery: {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")

    print(f"Đang build FAISS index (type={args.index_type}) ...")
    index = build_index(embeddings, index_type=args.index_type, nlist=args.nlist)

    sanity_check(index, embeddings, labels)

    index_path = os.path.join(args.out_dir, "shape_gallery.index")
    labels_path = os.path.join(args.out_dir, "gallery_labels.npy")
    faiss.write_index(index, index_path)
    np.save(labels_path, labels)
    print(f"Đã lưu index tại: {index_path}")
    print(f"Đã lưu labels tại: {labels_path}")

    if paths is not None:
        paths_path = os.path.join(args.out_dir, "gallery_paths.npy")
        np.save(paths_path, paths)
        print(f"Đã lưu đường dẫn ảnh gốc tại: {paths_path}")


if __name__ == "__main__":
    main()