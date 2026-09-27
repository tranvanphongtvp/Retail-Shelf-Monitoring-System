"""
audit_gallery.py

Kiem tra chat luong gallery embeddings + labels.
Phat hien label nhieu, mau outlier, va do tach biet giua cac class.

Cach dung:
    .venv\\Scripts\\python.exe src/audit_gallery.py
    .venv\\Scripts\\python.exe src/audit_gallery.py --gallery embeddings/gallery_embeddings.npz --top_problems 30
"""

import argparse
import numpy as np
from pathlib import Path
from collections import Counter

_ROOT = Path(__file__).resolve().parents[1]


def load_gallery(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["label_names"]
    paths = data["image_paths"] if "image_paths" in data.files else None
    return embeddings, labels, paths


def compute_class_stats(embeddings, labels):
    """Tinh intra-class similarity va phat hien outlier cho tung label."""
    unique_labels = np.unique(labels)
    stats = []

    for label in unique_labels:
        mask = labels == label
        class_embs = embeddings[mask]
        n_samples = len(class_embs)

        if n_samples < 2:
            stats.append({
                "label": label,
                "n_samples": n_samples,
                "intra_sim_mean": 1.0,
                "intra_sim_min": 1.0,
                "intra_sim_std": 0.0,
                "outlier_indices": [],
                "outlier_scores": [],
            })
            continue

        # Cosine similarity matrix (embeddings da L2-normalize -> dot product = cosine)
        sim_matrix = class_embs @ class_embs.T

        # Intra-class: trung binh sim giua tat ca cac cap (loai diagonal)
        n = len(sim_matrix)
        mask_upper = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[mask_upper]

        intra_mean = float(pairwise_sims.mean()) if len(pairwise_sims) > 0 else 1.0
        intra_min = float(pairwise_sims.min()) if len(pairwise_sims) > 0 else 1.0
        intra_std = float(pairwise_sims.std()) if len(pairwise_sims) > 0 else 0.0

        # Tim outlier: mau nao co trung binh sim voi cac mau khac cung class thap nhat
        avg_sim_per_sample = (sim_matrix.sum(axis=1) - 1.0) / (n - 1)  # tru di sim voi chinh no
        threshold = intra_mean - 2 * intra_std  # 2 sigma rule
        outlier_mask = avg_sim_per_sample < threshold
        outlier_indices = np.where(mask)[0][outlier_mask].tolist()
        outlier_scores = avg_sim_per_sample[outlier_mask].tolist()

        stats.append({
            "label": label,
            "n_samples": n_samples,
            "intra_sim_mean": intra_mean,
            "intra_sim_min": intra_min,
            "intra_sim_std": intra_std,
            "outlier_indices": outlier_indices,
            "outlier_scores": outlier_scores,
        })

    return stats


def find_confused_pairs(embeddings, labels, top_k=20):
    """Tim cac cap label khac nhau nhung co embedding gan nhau nhat.
    Dung FAISS de tim nearest centroid thay vi tinh full NxN matrix."""
    import faiss

    unique_labels = np.unique(labels)

    # Tinh centroid cho moi class
    centroids = []
    centroid_labels = []
    for label in unique_labels:
        mask = labels == label
        class_embs = embeddings[mask]
        centroid = class_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
        centroid_labels.append(label)

    centroids = np.array(centroids, dtype=np.float32)
    centroid_labels = np.array(centroid_labels)

    # Dung FAISS IndexFlatIP de tim K nearest centroids cho moi centroid
    dim = centroids.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(centroids)

    search_k = min(top_k + 1, len(centroids))  # +1 vi top-1 la chinh no
    scores, indices = index.search(centroids, search_k)

    # Thu thap cac cap (bo self-match)
    seen = set()
    pairs = []
    for i in range(len(centroids)):
        for j_rank in range(search_k):
            j = indices[i][j_rank]
            if j == i:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key not in seen:
                seen.add(pair_key)
                pairs.append((centroid_labels[i], centroid_labels[j], float(scores[i][j_rank])))

    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Audit gallery quality")
    parser.add_argument("--gallery", type=str,
                        default=str(_ROOT / "embeddings" / "gallery_embeddings.npz"))
    parser.add_argument("--top_problems", type=int, default=20,
                        help="Number of top problematic labels to show")
    parser.add_argument("--top_confused", type=int, default=20,
                        help="Number of top confused label pairs to show")
    args = parser.parse_args()

    print(f"Loading gallery from {args.gallery} ...")
    embeddings, labels, paths = load_gallery(args.gallery)
    print(f"Gallery: {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")
    print(f"Unique labels: {len(np.unique(labels))}")

    # --- Basic stats ---
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)

    label_counts = Counter(labels.tolist())
    counts_values = list(label_counts.values())
    print(f"Total vectors: {len(labels)}")
    print(f"Unique labels: {len(label_counts)}")
    print(f"Samples per label -- min: {min(counts_values)}, max: {max(counts_values)}, "
          f"mean: {np.mean(counts_values):.1f}, median: {np.median(counts_values):.1f}")

    # Distribution of sample counts
    print("\nSample count distribution:")
    brackets = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 50), (51, 1000)]
    for lo, hi in brackets:
        count = sum(1 for v in counts_values if lo <= v <= hi)
        if count > 0:
            label_str = f"{lo}" if lo == hi else f"{lo}-{hi}"
            print(f"  {label_str} samples: {count} labels ({count/len(label_counts)*100:.1f}%)")

    # --- Embedding norm check ---
    print("\n" + "=" * 80)
    print("EMBEDDING NORM CHECK")
    print("=" * 80)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Norm -- min: {norms.min():.4f}, max: {norms.max():.4f}, "
          f"mean: {norms.mean():.4f}, std: {norms.std():.6f}")
    if norms.std() < 0.01 and abs(norms.mean() - 1.0) < 0.05:
        print("-> Embeddings are L2-normalized (OK)")
    else:
        print("-> WARNING: Embeddings may NOT be properly L2-normalized!")

    # --- Intra-class analysis ---
    print("\n" + "=" * 80)
    print("INTRA-CLASS SIMILARITY ANALYSIS")
    print("=" * 80)
    print("Computing per-class stats (this may take a while)...")

    stats = compute_class_stats(embeddings, labels)

    # Sort by intra_sim_mean ascending (worst classes first)
    stats_multi = [s for s in stats if s["n_samples"] >= 2]
    stats_multi.sort(key=lambda x: x["intra_sim_mean"])

    all_intra_means = [s["intra_sim_mean"] for s in stats_multi]
    if all_intra_means:
        print(f"\nIntra-class similarity (classes with >= 2 samples):")
        print(f"  Overall mean: {np.mean(all_intra_means):.4f}")
        print(f"  Overall min:  {np.min(all_intra_means):.4f}")
        print(f"  Overall max:  {np.max(all_intra_means):.4f}")
        print(f"  Overall std:  {np.std(all_intra_means):.4f}")

    # Labels with lowest intra-class similarity (most inconsistent)
    print(f"\nTop {args.top_problems} WORST labels (lowest intra-class similarity):")
    print(f"  {'Label':>15s}  {'N':>4s}  {'Sim Mean':>8s}  {'Sim Min':>8s}  {'Outliers':>8s}")
    print(f"  {'-'*15}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for s in stats_multi[:args.top_problems]:
        print(f"  {s['label']:>15s}  {s['n_samples']:>4d}  {s['intra_sim_mean']:>8.4f}  "
              f"{s['intra_sim_min']:>8.4f}  {len(s['outlier_indices']):>8d}")

    # Total outliers
    total_outliers = sum(len(s["outlier_indices"]) for s in stats)
    print(f"\nTotal outlier samples detected: {total_outliers} / {len(labels)} "
          f"({total_outliers/len(labels)*100:.1f}%)")

    # --- Confused pairs ---
    print("\n" + "=" * 80)
    print("MOST CONFUSED LABEL PAIRS (highest inter-class similarity)")
    print("=" * 80)
    print("Computing centroid similarities...")

    confused = find_confused_pairs(embeddings, labels, top_k=args.top_confused)
    print(f"\nTop {args.top_confused} most confused pairs:")
    print(f"  {'Label A':>15s}  {'Label B':>15s}  {'Centroid Sim':>12s}  {'Samples A':>9s}  {'Samples B':>9s}")
    print(f"  {'-'*15}  {'-'*15}  {'-'*12}  {'-'*9}  {'-'*9}")
    for label_a, label_b, sim in confused:
        na = int(np.sum(labels == label_a))
        nb = int(np.sum(labels == label_b))
        print(f"  {str(label_a):>15s}  {str(label_b):>15s}  {sim:>12.4f}  {na:>9d}  {nb:>9d}")

    # --- Summary and recommendation ---
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    # Check if many classes have very low intra-class similarity
    low_intra = [s for s in stats_multi if s["intra_sim_mean"] < 0.7]
    very_low_intra = [s for s in stats_multi if s["intra_sim_mean"] < 0.5]

    print(f"\nClasses with intra-class sim < 0.7: {len(low_intra)} / {len(stats_multi)} "
          f"({len(low_intra)/max(len(stats_multi),1)*100:.1f}%)")
    print(f"Classes with intra-class sim < 0.5: {len(very_low_intra)} / {len(stats_multi)} "
          f"({len(very_low_intra)/max(len(stats_multi),1)*100:.1f}%)")

    # Check confused pairs
    very_confused = [p for p in confused if p[2] > 0.95]
    print(f"Label pairs with centroid sim > 0.95: {len(very_confused)}")

    single_sample_labels = sum(1 for v in counts_values if v == 1)
    print(f"Labels with only 1 sample: {single_sample_labels} / {len(label_counts)} "
          f"({single_sample_labels/len(label_counts)*100:.1f}%)")

    print("\n--- Recommendations ---")
    if len(very_low_intra) > len(stats_multi) * 0.1:
        print("[HIGH] Many classes have very low internal consistency (<0.5).")
        print("       -> These labels likely contain WRONG images. Clean them first.")
    if len(very_confused) > 10:
        print("[HIGH] Many label pairs are nearly identical (centroid sim > 0.95).")
        print("       -> These may be duplicate labels for the same product. Merge them.")
    if single_sample_labels > len(label_counts) * 0.3:
        print("[MEDIUM] Many labels have only 1 sample.")
        print("         -> Add more samples per label for better recognition.")
    if total_outliers > len(labels) * 0.05:
        print(f"[MEDIUM] {total_outliers} outlier samples detected ({total_outliers/len(labels)*100:.1f}%).")
        print("         -> Remove these outliers and rebuild FAISS index.")
    if np.mean(all_intra_means) < 0.8:
        print("[LOW] Average intra-class similarity is below 0.8.")
        print("      -> Consider re-training the embedding model with harder mining.")

    print("\nDone.")


if __name__ == "__main__":
    main()
