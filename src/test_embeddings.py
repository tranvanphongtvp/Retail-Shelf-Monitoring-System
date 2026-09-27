import numpy as np

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["label_names"]

def evaluate_topk(gallery_emb, gallery_labels, test_emb, test_labels, ks=(1, 5, 10)):
    """
    Brute-force cosine similarity (vector đã L2-normalize -> dot product = cosine).
    Với mỗi ảnh test, tìm top-k ảnh gallery gần nhất, kiểm tra nhãn có khớp không.
    """
    sims = test_emb @ gallery_emb.T          # (n_test, n_gallery)
    order = np.argsort(-sims, axis=1)        # sắp giảm dần theo similarity

    results = {}
    for k in ks:
        correct = 0
        for i in range(len(test_labels)):
            top_k_idx = order[i, :k]
            top_k_labels = gallery_labels[top_k_idx]
            if test_labels[i] in top_k_labels:
                correct += 1
        results[f"top-{k}"] = correct / len(test_labels)
    return results

if __name__ == "__main__":
    GALLERY_PATH = "embeddings/gallery_embeddings.npz"
    TEST_PATH    = "embeddings/test_embeddings.npz"

    gallery_emb, gallery_labels = load_npz(GALLERY_PATH)
    test_emb, test_labels       = load_npz(TEST_PATH)

    print(f"gallery: {gallery_emb.shape} | test: {test_emb.shape}")
    acc = evaluate_topk(gallery_emb, gallery_labels, test_emb, test_labels, ks=(1, 5, 10))
    for k, v in acc.items():
        print(f"{k}: {v:.4f}")