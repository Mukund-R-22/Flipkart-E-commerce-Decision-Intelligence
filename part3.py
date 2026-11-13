"""Part 3 - Segmentation and dimensionality reduction. Run: python part3.py"""
import numpy as np

from common import load_data, sample_sd_cols, write_csv, write_results

data = load_data()
X5 = data["X"][:, :5]
outcome = data["outcome"]
value = data["value"]
n = len(X5)
Z = (X5 - X5.sum(axis=0) / n) / sample_sd_cols(X5)      # standardised 800 x 5
out = []


def add(k, v):
    out.append((k, v))


# ---- 3.1 K-means from scratch
def kmeans(Z, K):
    step = len(Z) // K
    cent = Z[[j * step for j in range(K)]].copy()
    labels = None
    passes = 0
    while True:
        d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)              # ties -> lower-numbered centroid
        passes += 1
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            members = Z[labels == k]
            if len(members) > 0:                # empty cluster keeps its centroid
                cent[k] = members.sum(axis=0) / len(members)
    wcss = float(((Z - cent[labels]) ** 2).sum())
    return labels, cent, wcss, passes


results = {}
for K in range(2, 7):
    results[K] = kmeans(Z, K)
    add("wcss_k%d" % K, results[K][2])
    print("K=%d passes=%d" % (K, results[K][3]))

# ---- 3.2 elbow rule
W = {K: results[K][2] for K in results}
elbow = None
for K in (3, 4, 5):
    if W[K] - W[K + 1] < 0.5 * (W[K - 1] - W[K]):
        elbow = K
        break
add("elbow_k", elbow if elbow is not None else 0)   # 0 = rule never satisfied

# ---- 3.3 profile K = 3 (A, B, C by ascending mean X3)
labels3 = results[3][0]
x3_means = [float(X5[labels3 == k, 2].sum() / np.sum(labels3 == k)) for k in range(3)]
rank = sorted(range(3), key=lambda k: x3_means[k])
seg_names = ["A", "B", "C"]
seg_of_row = np.empty(n, dtype=int)
seg_rows = []
for name, k in zip(seg_names, rank):
    mask = labels3 == k
    seg_of_row[mask] = seg_names.index(name)
    size = int(mask.sum())
    add("size_segment_%s" % name, size)
    seg_rows.append([name, size] + [float(X5[mask, j].sum() / size) for j in range(5)]
                    + [float(outcome[mask].sum()) / size, float(value[mask].sum() / size)])
write_csv("part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
          seg_rows)

# ---- 3.4 hierarchical clustering on the supplied table
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def linkage_dist(a, b, rule):
    ds = [D[i, j] for i in a for j in b]
    if rule == "single":
        return min(ds)
    if rule == "complete":
        return max(ds)
    return sum(ds) / len(ds)


def label(c):
    return "+".join(names[i] for i in sorted(c))


def agglomerate(rule):
    clusters = [[i] for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage_dist(clusters[i], clusters[j], rule)
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        a, b = clusters[i], clusters[j]
        if min(b) < min(a):
            a, b = b, a
        merges.append((label(a), label(b), d))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [sorted(a + b)]
    return merges


hrows = []
for rule in ("single", "complete", "average"):
    merges = agglomerate(rule)
    for m_no, (a, b, h) in enumerate(merges, start=1):
        hrows.append([rule, m_no, a, b, h])
    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    g = max(range(4), key=lambda i: gaps[i])         # gap between merge g+1 and g+2
    add("%s_clusters_at_cut" % rule, 6 - (g + 1))
    print(rule, [(m[0], m[1], round(m[2], 4)) for m in merges])
write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hrows)

# ---- 3.5 PCA in seven steps
mean_vec = Z.sum(axis=0) / n                 # (1) ~ zeros after standardising
Xc = Z - mean_vec                            # (2) centre
S = (Xc.T @ Xc) / (n - 1)                    # (3) covariance
vals, vecs = np.linalg.eigh(S)               # (4) eigen-decomposition
idx = np.argsort(-vals, kind="stable")       # (5) largest first
vals, vecs = vals[idx], vecs[:, idx]
for j in range(5):                           # sign convention
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
Wtop = vecs[:, :2]                           # (6) top two
proj = Xc @ Wtop                             # (7) project
total = float(vals.sum())
add("eigenvalue_sum", total)
for j in range(5):
    add("eigenvalue_%d" % (j + 1), float(vals[j]))
for j in range(5):
    add("explained_ratio_%d" % (j + 1), float(vals[j]) / total)
cum = np.cumsum(vals) / total
for j in range(5):
    add("cumulative_ratio_%d" % (j + 1), float(cum[j]))
add("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1)
for pc in (0, 1):
    for j in range(5):
        add("pc%d_loading_X%d" % (pc + 1, j + 1), float(vecs[j, pc]))
for r in range(3):
    add("row%d_pc1" % (r + 1), float(proj[r, 0]))
    add("row%d_pc2" % (r + 1), float(proj[r, 1]))
for si, name in enumerate(seg_names):
    mask = seg_of_row == si
    add("mean_pc1_segment_%s" % name, float(proj[mask, 0].sum() / mask.sum()))

write_results("part3_results.txt", out)
print("mean vector (should be ~0):", np.round(mean_vec, 12))
print("part3 done:", len(out), "keys")
