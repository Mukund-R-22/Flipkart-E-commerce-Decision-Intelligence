"""Part 1 - Estimation, inference and a regression baseline. Run: python part1.py"""
import math

import numpy as np

from common import load_data, mean, sample_sd, write_results

data = load_data()
X = data["X"]
y = data["value"]
x4 = X[:, 3]
n = len(y)
out = []


def add(k, v):
    out.append((k, v))


# ---- 1.1 estimate with an interval
mean_value = mean(y)
sd_value = sample_sd(y)
se_value = sd_value / math.sqrt(n)
add("n", n)
add("mean_value", mean_value)
add("sd_value", sd_value)
add("se_value", se_value)
add("ci95_low", mean_value - 1.96 * se_value)
add("ci95_high", mean_value + 1.96 * se_value)

# ---- 1.2 test the planning claim
z_stat = (mean_value - 8.0) / se_value
add("z_stat", z_stat)
add("reject_h0", abs(z_stat) > 1.96)

# ---- 1.3 sample size for margin E = 0.5
E = 0.5
add("n_required", math.ceil((1.96 * sd_value / E) ** 2))


# ---- 1.4 simple regression, closed form
def simple_ols(x, t):
    mx, mt = mean(x), mean(t)
    cov = float(np.sum((x - mx) * (t - mt))) / (len(x) - 1)
    var = float(np.sum((x - mx) ** 2)) / (len(x) - 1)
    slope = cov / var
    intercept = mt - slope * mx
    fitted = intercept + slope * x
    tss = float(np.sum((t - mt) ** 2))
    ess = float(np.sum((fitted - mt) ** 2))
    rss = float(np.sum((t - fitted) ** 2))
    return slope, intercept, fitted, tss, ess, rss


reg_slope, reg_intercept, fitted, tss, ess, rss = simple_ols(x4, y)
resid = y - fitted
add("reg_slope", reg_slope)
add("reg_intercept", reg_intercept)
add("tss", tss)
add("ess", ess)
add("rss", rss)
add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
add("r2", 1 - rss / tss)
add("rmse", math.sqrt(rss / n))
add("mae", float(np.sum(np.abs(resid))) / n)

# ---- 1.5 gradient descent on standardised X4
mx4, sdx4 = mean(x4), sample_sd(x4)
z = (x4 - mx4) / sdx4
m, c = 0.0, 0.0
eta = 0.1
J = {}
J[0] = float(np.sum((y - (m * z + c)) ** 2)) / n
for step in range(1, 501):
    e = y - (m * z + c)
    dm = -(2.0 / n) * float(np.sum(z * e))
    dc = -(2.0 / n) * float(np.sum(e))
    m -= eta * dm
    c -= eta * dc
    if step in (1, 10, 100, 500):
        J[step] = float(np.sum((y - (m * z + c)) ** 2)) / n
add("gd_J0", J[0])
add("gd_J1", J[1])
add("gd_J10", J[10])
add("gd_J100", J[100])
add("gd_J500", J[500])
gd_slope = m / sdx4
gd_intercept = c - m * mx4 / sdx4
add("gd_slope", gd_slope)
add("gd_intercept", gd_intercept)
add("gd_slope_gap", abs(gd_slope - reg_slope))


# ---- 1.6 funnel check and log remedy
def half_sds(fit, res):
    order = np.argsort(fit, kind="stable")
    lo, hi = order[:400], order[400:]
    return sample_sd(res[lo]), sample_sd(res[hi])


lo_raw, hi_raw = half_sds(fitted, resid)
add("resid_sd_lower_raw", lo_raw)
add("resid_sd_upper_raw", hi_raw)
add("resid_sd_ratio_raw", hi_raw / lo_raw)

ly, lx = np.log(y), np.log(x4)
ll_slope, ll_int, ll_fit, ll_tss, ll_ess, ll_rss = simple_ols(lx, ly)
ll_r2 = 1 - ll_rss / ll_tss
add("loglog_intercept", ll_int)
add("loglog_slope", ll_slope)
add("loglog_r2", ll_r2)
add("loglog_rmse", math.sqrt(ll_rss / n))
lo_log, hi_log = half_sds(ll_fit, ly - ll_fit)
add("resid_sd_lower_log", lo_log)
add("resid_sd_upper_log", hi_log)
add("resid_sd_ratio_log", hi_log / lo_log)

# ---- 1.7 multiple regression, normal equation
A = np.column_stack([np.ones(n), np.log(X[:, 3]), X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(A.T @ A, A.T @ ly)
for i in range(8):
    add("mr_beta_%d" % i, float(beta[i]))
mr_fit = A @ beta
mr_rss = float(np.sum((ly - mr_fit) ** 2))
mr_r2 = 1 - mr_rss / ll_tss
p = 7
add("mr_r2", mr_r2)
add("mr_adj_r2", 1 - (1 - mr_r2) * (n - 1) / (n - p - 1))
add("loglog_adj_r2", 1 - (1 - ll_r2) * (n - 1) / (n - 1 - 1))

write_results("part1_results.txt", out)
print("part1 done:", len(out), "keys")
