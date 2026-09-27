"""
clean_gallery.py

Thực hiện Bước 4: Làm sạch gallery (Loại bỏ outliers).
1. Đọc gallery_embeddings.npz gốc
2. Tính intra-class similarity, tìm ra các crop có similarity khác biệt (outliers) 
3. Lọc bỏ các outliers này khỏi gallery
4. Build lại FAISS index với dữ liệu đã làm sạch.
"""

import os
import argparse
import numpy as np
import faiss
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def load_gallery(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["label_names"]
    paths = data["image_paths"] if "image_paths" in data.files else None
    return embeddings, labels, paths


def find_outliers(embeddings, labels):
    unique_labels = np.unique(labels)
    all_outlier_indices = []

    print(f"Analyzing {len(unique_labels)} labels to find outliers...")
    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]
        class_embs = embeddings[mask]
        n_samples = len(class_embs)

        if n_samples < 2:
            continue

        sim_matrix = class_embs @ class_embs.T
        n = len(sim_matrix)
        mask_upper = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[mask_upper]

        if len(pairwise_sims) == 0:
            continue

        intra_mean = float(pairwise_sims.mean())
        intra_std = float(pairwise_sims.std())

        avg_sim_per_sample = (sim_matrix.sum(axis=1) - 1.0) / (n - 1)
        threshold = intra_mean - 2 * intra_std
        
        outlier_mask = avg_sim_per_sample < threshold
        
        if outlier_mask.sum() > 0:
            class_outlier_indices = indices[outlier_mask]
            all_outlier_indices.extend(class_outlier_indices.tolist())

    return set(all_outlier_indices)


def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery_in", type=str, default=str(_ROOT / "embeddings" / "gallery_embeddings.npz"))
    parser.add_argument("--gallery_out", type=str, default=str(_ROOT / "embeddings" / "clean_gallery_embeddings.npz"))
    parser.add_argument("--index_dir", type=str, default=str(_ROOT / "faiss_index"))
    args = parser.parse_args()

    # 1. Load original
    print(f"Loading original gallery from: {args.gallery_in}")
    embeddings, labels, paths = load_gallery(args.gallery_in)
    print(f"Initial size: {len(labels)} vectors")

    # 2. Find outliers
    outliers = find_outliers(embeddings, labels)
    print(f"Detected {len(outliers)} outliers ({len(outliers)/len(labels)*100:.1f}%)")

    # 3. Filter
    keep_indices = [i for i in range(len(labels)) if i not in outliers]
    keep_indices = np.array(keep_indices)

    clean_embeddings = embeddings[keep_indices]
    clean_labels = labels[keep_indices]
    clean_paths = paths[keep_indices] if paths is not None else None

    print(f"Size after cleaning: {len(clean_labels)} vectors")

    # 4. Save clean npz
    os.makedirs(os.path.dirname(args.gallery_out), exist_ok=True)
    save_dict = {
        "embeddings": clean_embeddings,
        "label_names": clean_labels
    }
    if clean_paths is not None:
        save_dict["image_paths"] = clean_paths
    
    np.savez(args.gallery_out, **save_dict)
    print(f"Saved clean gallery npz to: {args.gallery_out}")

    # 5. Build and save new FAISS index
    os.makedirs(args.index_dir, exist_ok=True)
    print("Rebuilding FAISS index...")
    index = build_index(clean_embeddings)

    index_path = os.path.join(args.index_dir, "shape_gallery.index")
    labels_path = os.path.join(args.index_dir, "gallery_labels.npy")
    
    faiss.write_index(index, index_path)
    np.save(labels_path, clean_labels)
    
    if clean_paths is not None:
        paths_path = os.path.join(args.index_dir, "gallery_paths.npy")
        np.save(paths_path, clean_paths)

    print(f"Updated FAISS index at: {index_path}")
    print("STEP 4 COMPLETED!")


if __name__ == "__main__":
    main()
