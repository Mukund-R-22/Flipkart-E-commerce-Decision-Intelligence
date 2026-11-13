"""Part 4 - Model-choice and architecture memo. Run: python part4.py
Re-uses the split, logistic model and KNN of Part 2 through common.py (same code, same numbers)."""
import math

import numpy as np

from common import (confusion, knn_cross_validate, knn_order, knn_predict, knn_share, load_data,
                    metrics, predict_proba, prepare_split, train_logistic, write_csv, write_results)

data = load_data()
s = prepare_split(data)
Xtr, Xte, ytr, yte = s["Xtr"], s["Xte"], s["ytr"], s["yte"]
Rtr, Rte = s["Xtr_raw"], s["Xte_raw"]          # original units
out = []


def add(k, v):
    out.append((k, v))


# ---- 4.1 purity of four candidate splits (original units, training rows)
def entropy(pos, n):
    if n == 0:
        return 0.0
    e = 0.0
    for q in (pos / n, 1 - pos / n):
        if q > 0:
            e -= q * math.log2(q)
    return e


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1 - p ** 2 - (1 - p) ** 2


splits = {
    "S1": lambda R: R[:, 5] == 1,        # X6 = 1
    "S2": lambda R: R[:, 6] == 1,        # X7 = 1
    "S3": lambda R: R[:, 2] > 0.5,       # X3 > 0.5
    "S4": lambda R: R[:, 0] <= 24,       # X1 <= 24
}
n_tr, pos_tr = len(ytr), int(ytr.sum())
root_e, root_g = entropy(pos_tr, n_tr), gini(pos_tr, n_tr)
add("root_entropy", root_e)
add("root_gini", root_g)
rows41, ig, gg = [], {}, {}
for name, rule in splits.items():
    yes = rule(Rtr)
    ny, py = int(yes.sum()), int(ytr[yes].sum())
    nn, pn = n_tr - ny, pos_tr - py
    ig[name] = root_e - (ny / n_tr * entropy(py, ny) + nn / n_tr * entropy(pn, nn))
    gg[name] = root_g - (ny / n_tr * gini(py, ny) + nn / n_tr * gini(pn, nn))
    rows41.append([name, ny, py, nn, pn, ig[name], gg[name]])
write_csv("part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], rows41)
best_ig = max(splits, key=lambda k: ig[k])       # first maximum wins on exact ties
best_gg = max(splits, key=lambda k: gg[k])
add("best_split_by_info_gain", best_ig)
add("best_split_by_gini_gain", best_gg)

# ---- 4.2 stump + five predictions on the 240 test rows
yes_tr = splits[best_ig](Rtr)
rate_yes = float(ytr[yes_tr].sum()) / int(yes_tr.sum())
rate_no = float(ytr[~yes_tr].sum()) / int((~yes_tr).sum())
add("stump_rate_yes", rate_yes)
add("stump_rate_no", rate_no)

w, b, _ = train_logistic(Xtr, ytr, 0.1, 1000)
p_log = predict_proba(Xte, w, b)
_, best_k = knn_cross_validate(Xtr, ytr)
order = knn_order(Xte, Xtr)
p_knn = knn_share(order, ytr, best_k)
pred_knn = knn_predict(order, ytr, best_k)
p_stump = np.where(splits[best_ig](Rte), rate_yes, rate_no)

pred_log = (p_log >= 0.3).astype(int)
pred_stump = (p_stump >= 0.3).astype(int)
pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(int)
pred_soft = (((p_log + p_knn + p_stump) / 3.0) >= 0.3).astype(int)
vrows = []
for name, pred in (("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                   ("hard_vote", pred_hard), ("soft_vote", pred_soft)):
    tp, fp, tn, fn = confusion(yte, pred)
    m = metrics(tp, fp, tn, fn)
    vrows.append([name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]])
write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)

# ---- 4.3 AdaBoost, two rounds by hand
y = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=float)
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=float)
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=float)
w_t = np.full(8, 1.0 / 8)
arows, alphas = [], []
for rnd, h in ((1, h1), (2, h2)):
    eps = float(w_t[h != y].sum())
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = w_t * np.exp(-alpha * y * h)
    new = unnorm / unnorm.sum()
    for i in range(8):
        arows.append([rnd, i + 1, float(w_t[i]), float(unnorm[i]), float(new[i])])
    add("round%d_error" % rnd, eps)
    add("round%d_alpha" % rnd, alpha)
    alphas.append(alpha)
    w_t = new
write_csv("part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], arows)
score = alphas[0] * h1 + alphas[1] * h2
add("ensemble_accuracy", float(np.sum(np.sign(score) == y)) / 8)

# ---- 4.4 out-of-bag share
add("oob_fraction", (1 - 1 / 560) ** 560)


# ---- 4.5 network sizing (counts only, nothing is trained)
def dense(i, o):
    return i * o + o


def conv(k, c_in, c_out):
    return k * k * c_in * c_out + c_out


def out_side(inp, f, pad, stride):
    return (inp - f + 2 * pad) // stride + 1


mlp = [dense(784, 16), dense(16, 16), dense(16, 10)]
s1 = out_side(28, 5, 2, 1)       # conv1 -> 28
p1 = out_side(s1, 2, 0, 2)       # pool  -> 14
s2 = out_side(p1, 5, 0, 1)       # conv2 -> 10
p2 = out_side(s2, 2, 0, 2)       # pool  -> 5
flat = p2 * p2 * 16              # 400
cnn = [conv(5, 1, 6), conv(5, 6, 16), dense(flat, 32), dense(32, 10)]
layer_names = ["mlp_hidden1", "mlp_hidden2", "mlp_output", "cnn_conv1", "cnn_conv2", "cnn_dense1", "cnn_dense2"]
write_csv("part4_architecture.csv", ["layer", "parameters"], [[n_, p_] for n_, p_ in zip(layer_names, mlp + cnn)])
add("mlp_total", sum(mlp))
add("cnn_total", sum(cnn))
add("fc_layer_params", dense(128 * 128 * 3, 1024))
add("conv_layer_params", conv(3, 3, 64))
d_in, d_h, d_out = 64, 128, 16
add("rnn_total", d_h * d_in + d_h * d_h + d_out * d_h + d_h + d_out)

write_results("part4_results.txt", out)
print("CNN shapes: conv1 %dx%dx6, pool1 %dx%dx6, conv2 %dx%dx16, pool2 %dx%dx16, flatten %d" % (s1, s1, p1, p1, s2, s2, p2, p2, flat))
print("part4 done:", len(out), "keys")
