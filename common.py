"""Shared helpers for all four Parts. Importing this module runs nothing."""
import csv
import math
import os

import numpy as np

DATA_PATH = os.path.join("data", "records.csv")
OUT_DIR = "outputs"


# ---------------------------------------------------------------- data + output
def load_data():
    """Return dict with X (800x7 float), outcome (int), value (float)."""
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.reader(f))[1:]
    X = np.array([[float(r[i]) for i in range(1, 8)] for r in rows])
    outcome = np.array([int(r[8]) for r in rows])
    value = np.array([float(r[9]) for r in rows])
    return {"X": X, "outcome": outcome, "value": value}


def fmt(v):
    """bool -> yes/no, int -> integer, float -> 4 decimals, text as given."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = "%.4f" % float(v)
        return "0.0000" if s == "-0.0000" else s
    return str(v)


def write_results(name, pairs):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for k, v in pairs:
            f.write("%s = %s\n" % (k, fmt(v)))


def write_csv(name, header, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(v) for v in r])


# ---------------------------------------------------------------- statistics
def mean(a):
    return float(np.sum(a) / len(a))


def sample_sd(a):
    m = mean(a)
    return math.sqrt(float(np.sum((a - m) ** 2)) / (len(a) - 1))


def sample_sd_cols(M):
    m = M.sum(axis=0) / M.shape[0]
    return np.sqrt(((M - m) ** 2).sum(axis=0) / (M.shape[0] - 1))


# ---------------------------------------------------------------- Task 2.1
def stratified_split(outcome):
    """Per class, first floor(7*count/10) rows (file order) -> train. Returns sorted index arrays."""
    train = []
    for cls in (0, 1):
        idx = [i for i in range(len(outcome)) if outcome[i] == cls]
        train.extend(idx[: (7 * len(idx)) // 10])
    train_set = set(train)
    train_idx = np.array(sorted(train_set))
    test_idx = np.array([i for i in range(len(outcome)) if i not in train_set])
    return train_idx, test_idx


def prepare_split(data):
    """Split + standardise with TRAINING mean and sample SD only."""
    X, y = data["X"], data["outcome"]
    tr, te = stratified_split(y)
    mu = X[tr].sum(axis=0) / len(tr)
    sd = sample_sd_cols(X[tr])
    return {
        "tr": tr, "te": te,
        "Xtr": (X[tr] - mu) / sd, "Xte": (X[te] - mu) / sd,
        "ytr": y[tr], "yte": y[te],
        "Xtr_raw": X[tr], "Xte_raw": X[te],
    }


# ---------------------------------------------------------------- logistic regression
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y))


def train_logistic(X, y, eta, epochs, record=()):
    """Batch gradient descent from zero. Returns w, b, {epoch: loss} (epoch 0 = before any update)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(X @ w + b))
    for ep in range(1, epochs + 1):
        p = sigmoid(X @ w + b)
        err = p - y
        w = w - eta * (X.T @ err) / n
        b = b - eta * float(np.sum(err)) / n
        if ep in record:
            losses[ep] = bce(y, sigmoid(X @ w + b))
    return w, b, losses


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


# ---------------------------------------------------------------- metrics
def confusion(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    acc = _div(tp + tn, tp + fp + tn + fn)
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    f1 = _div(2 * prec * rec, prec + rec)
    spec = _div(tn, tn + fp)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "specificity": spec}


# ---------------------------------------------------------------- KNN
def pairwise_dist(Q, R):
    """Euclidean distances, shape (len(Q), len(R))."""
    diff = Q[:, None, :] - R[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def knn_order(Q, R):
    """For each query, reference indices ranked by (distance, file position): stable sort."""
    return np.argsort(pairwise_dist(Q, R), axis=1, kind="stable")


def knn_share(order, ref_y, K):
    """Share of positives among the K nearest neighbours of each query."""
    return ref_y[order[:, :K]].sum(axis=1) / K


def knn_predict(order, ref_y, K):
    """Positive when at least (K+1)/2 of the K neighbours are positive (K odd)."""
    return (ref_y[order[:, :K]].sum(axis=1) >= (K + 1) // 2).astype(int)


KNN_KS = list(range(1, 22, 2))


def knn_cross_validate(Xtr, ytr):
    """5-fold CV (fold = position mod 5) over K = 1,3,...,21. Returns (list of (K, mean F1), best K)."""
    pos = np.arange(len(ytr))
    fold = pos % 5
    f1s = {K: [] for K in KNN_KS}
    for f in range(5):
        held = fold == f
        ref = ~held
        order = knn_order(Xtr[held], Xtr[ref])
        ref_y = ytr[ref]
        for K in KNN_KS:
            pred = knn_predict(order, ref_y, K)
            f1s[K].append(metrics(*confusion(ytr[held], pred))["f1"])
    table = [(K, sum(f1s[K]) / 5.0) for K in KNN_KS]
    best_k, best = None, -1.0
    for K, s in table:          # ascending K, strict > keeps the smaller K on ties
        if s > best:
            best, best_k = s, K
    return table, best_k


def cost(fp, fn):
    return 50 * fp + 400 * fn
