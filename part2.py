"""Part 2 - Classification and evaluation. Run: python part2.py"""
import numpy as np

from common import (bce, confusion, cost, knn_cross_validate, knn_order, knn_predict,
                    load_data, metrics, predict_proba, prepare_split, train_logistic,
                    write_csv, write_results)

data = load_data()
s = prepare_split(data)
Xtr, Xte, ytr, yte = s["Xtr"], s["Xte"], s["ytr"], s["yte"]
out = []


def add(k, v):
    out.append((k, v))


# ---- 2.1
add("train_rows", len(ytr))
add("train_positives", int(ytr.sum()))
add("test_rows", len(yte))
add("test_positives", int(yte.sum()))

# ---- 2.2 logistic regression from scratch
w, b, losses = train_logistic(Xtr, ytr, 0.1, 1000, record=(0, 1, 10, 100, 1000))
for ep in (0, 1, 10, 100, 1000):
    add("loss_epoch_%d" % ep, losses[ep])
for j in range(7):
    add("weight_X%d" % (j + 1), float(w[j]))
add("bias", b)
add("test_bce", bce(yte, predict_proba(Xte, w, b)))

# ---- 2.3 learning-rate effect
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
    add("loss_epoch_200_eta_%s" % eta, l[200])

# ---- 2.4 odds ratios
for j in range(7):
    add("odds_ratio_X%d" % (j + 1), float(np.exp(w[j])))

# ---- 2.5 two thresholds
ptest = predict_proba(Xte, w, b)
ptrain = predict_proba(Xtr, w, b)
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, ptest >= tau)
    m = metrics(tp, fp, tn, fn)
    for k, v in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", m["accuracy"]),
                 ("precision", m["precision"]), ("recall", m["recall"]), ("f1", m["f1"]),
                 ("specificity", m["specificity"])):
        add("tau_%s_%s" % (tau, k), v)
add("blind_accuracy", float(np.sum(yte == 0)) / len(yte))

# ---- 2.6 threshold sweep
rows = []
for i in range(1, 10):
    tau = i / 10
    tp, fp, tn, fn = confusion(yte, ptest >= tau)
    rows.append([tau, tp, fp, tn, fn, tp / (tp + fn) if tp + fn else 0.0, fp / (fp + tn) if fp + tn else 0.0])
write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# ---- 2.7 KNN with 5-fold CV
table, best_k = knn_cross_validate(Xtr, ytr)
write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [[K, f] for K, f in table])
add("knn_best_k", best_k)
order = knn_order(Xte, Xtr)
kp = knn_predict(order, ytr, best_k)
tp, fp, tn, fn = confusion(yte, kp)
m = metrics(tp, fp, tn, fn)
for k, v in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", m["accuracy"]),
             ("precision", m["precision"]), ("recall", m["recall"]), ("f1", m["f1"])):
    add("knn_" + k, v)

# ---- 2.8 cost-based threshold (chosen on TRAINING rows only)
best_tau, best_cost = None, None
for k in range(1, 20):
    tau = k / 20
    _, fp_, _, fn_ = confusion(ytr, ptrain >= tau)
    c = cost(fp_, fn_)
    if best_cost is None or c < best_cost:      # strict < keeps the smaller tau on ties
        best_cost, best_tau = c, tau
add("cost_tau", best_tau)
add("cost_train_at_tau", best_cost)
_, fp_, _, fn_ = confusion(yte, ptest >= best_tau)
add("cost_test_at_tau", cost(fp_, fn_))
_, fp_, _, fn_ = confusion(yte, ptest >= 0.5)
add("cost_test_at_0.5", cost(fp_, fn_))
add("cost_test_flag_nobody", 400 * int(yte.sum()))

write_results("part2_results.txt", out)
print("part2 done:", len(out), "keys")
