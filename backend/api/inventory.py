from collections import Counter


# ============================== INVENTORY / COUNTS ==============================

def summarize_counts(products) -> dict:
    return dict(Counter(p.EAN for p in products))