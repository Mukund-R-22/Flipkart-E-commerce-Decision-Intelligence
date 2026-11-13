Domain: E-commerce

All data is synthetic (produced by generate_data.py). Nothing here describes real Flipkart customers, products, policies or performance. Money is in Indian Rupees (INR). The value columns are in hundreds of INR, so 8.0 means INR 800.

## How to run

Run these commands from the repository root of a clean clone (Python 3.10 or newer), in this order:

```
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

generate_data.py prints 800 10 170 c6f249ba693f1ebeed318d6d5e18de23 (rows, columns, number of churned = 1 records, MD5 of the file). Each part script writes its files into the outputs folder. common.py is a shared helper module (data loading, split, logistic regression, KNN, metrics); importing it runs nothing.

## Part 1 findings

R1.1 Hypothesis test

H0 (average order value = 8.0, which is INR 800) was rejected. z_stat = 2.8930, which is larger than the critical value 1.96.

| Quantity | Value |
|---|---|
| Records (n) | 800 |
| Sample mean (mean_value) | 8.9382 |
| Sample SD (sd_value) | 9.1723 |
| Standard error (se_value) | 0.3243 |
| 95% interval | 8.3026 to 9.5738 |
| z_stat | 2.8930 |
| Critical value | 1.96 |
| Reject H0 | yes |
| Gap, mean_value - 8.0 | 0.9382 units (about INR 94) |

A Type I error here would mean concluding that the true average order value differs from INR 800 when it really is INR 800, so the team would re-plan its category budgets for no reason.

The gap is statistically clear: the whole 95% interval (INR 830 to INR 957) lies above INR 800. In practice it is moderate. The plan is about 12% too low, which matters when multiplied over many orders, but single orders vary a lot (sd_value is about INR 917), so the average says little about any one order.

R1.2 Sample size

| Item | Value |
|---|---|
| n_required (margin of error 0.5 units = INR 50, 95% confidence) | 1293 |
| Rows available | 800 |
| Shortfall | 493 |

The current data gives a margin of about 0.64 units (INR 64), wider than the INR 50 the team wants. To reach INR 50 the team needs about 60% more records than it has now.

R1.3 Gradient descent versus the closed form

| Method | Slope | Intercept |
|---|---|---|
| Closed form (reg_slope, reg_intercept) | 0.6701 | -3.6598 |
| Gradient descent (gd_slope, gd_intercept) | 0.6701 | -3.6598 |

gd_slope_gap = 0.0000 (the requirement is below 0.001). The squared-error cost of a straight line has a single bowl-shaped minimum, and 500 small steps on the standardised X4 slid to the bottom of the bowl, which is exactly where the closed-form formula lands (gd_J100 and gd_J500 are both 40.3698).

R1.4 Funnel check and log remedy

| Model | SD of residuals, lower half | SD of residuals, upper half | Ratio (upper / lower) |
|---|---|---|---|
| Raw (value on X4) | 2.3155 | 8.6891 | 3.7525 (resid_sd_ratio_raw) |
| Log (ln value on ln X4) | 0.5165 | 0.5369 | 1.0396 (resid_sd_ratio_log) |

The raw model shows a funnel: the residual spread in the upper half of fitted values is almost four times that of the lower half. The log model removes it: the two halves have almost the same spread (ratio close to 1).

R1.5 Multiple regression (ln value on eight columns)

| Term | Coefficient | Absolute value above 0.05? |
|---|---|---|
| Constant (mr_beta_0) | -1.8613 | excluded |
| ln(X4) (mr_beta_1) | 1.1576 | yes |
| X1 tenure (mr_beta_2) | -0.0074 | no |
| X2 complaints (mr_beta_3) | 0.0940 | yes |
| X3 discount share (mr_beta_4) | 0.1067 | yes |
| X5 loyalty score (mr_beta_5) | 0.0028 | no |
| X6 cod_flag (mr_beta_6) | 0.1518 | yes |
| X7 plus_member_flag (mr_beta_7) | 0.0172 | no |

The predictors above the 0.05 limit are ln(X4), X2, X3 and X6. This is a mechanical rule only. The units differ (a log, a count, a 0-to-1 ratio, a 0/1 flag), so it is not a ranking of importance.

| Model | Adjusted R2 |
|---|---|
| Seven predictors plus ln(X4) (mr_adj_r2) | 0.7403 |
| Log-log model with one predictor (loglog_adj_r2) | 0.6561 |

The larger model explains about 8 percentage points more of the variation in ln(value) even after the adjustment for extra predictors, so the extra predictors earn their place. The R2 values of the raw and the log model are on different scales and are not compared.

## Part 2 findings

Split: 560 training rows (119 positive) and 240 test rows (51 positive), standardised with training statistics only.

R2.1 Learning rate

| Learning rate | Training loss after 200 updates | Order |
|---|---|---|
| 1.0 | 0.3703 | lowest |
| 0.1 | 0.3776 | middle |
| 0.01 | 0.5091 | highest |

A learning rate that is too small (0.01) is still far from the minimum after 200 steps. A larger rate gets there faster; 1.0 was still stable on this standardised data, but a rate that is too large can overshoot and bounce instead of settling.

R2.2 Weights and odds ratios

| Predictor | Weight | Odds ratio |
|---|---|---|
| X1 customer_tenure_months | -0.5554 | 0.5739 |
| X2 complaint_count_90d | 0.5505 | 1.7341 |
| X3 discount_order_share | 0.9013 | 2.4629 |
| X4 avg_order_value | -0.1459 | 0.8642 |
| X5 loyalty_score | -0.2162 | 0.8056 |
| X6 cod_flag | 0.2348 | 1.2646 |
| X7 plus_member_flag | -0.0120 | 0.9881 |

The three largest absolute weights are X3 discount share (0.9013), X1 tenure (-0.5554) and X2 complaints (0.5505). The largest odds ratio is 2.4629 for X3: a customer whose discount share is one training standard deviation higher has about 2.5 times the odds of churning, other features fixed. Heavy discount-seekers are the most at risk.

R2.3 Why accuracy alone is weak, and the effect of the threshold

| Rule on the 240 test rows | tp | fp | tn | fn | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Blind (predict 0 for everyone) | 0 | 0 | 189 | 51 | 0.7875 | 0.0000 | 0.0000 | 0.0000 |
| Logistic, threshold 0.5 | 24 | 20 | 169 | 27 | 0.8042 | 0.5455 | 0.4706 | 0.5053 |
| Logistic, threshold 0.3 | 38 | 44 | 145 | 13 | 0.7625 | 0.4634 | 0.7451 | 0.5714 |

blind_accuracy is 0.7875 and the threshold 0.5 accuracy is 0.8042, only about two percentage points better. Only about one customer in five churns, so accuracy rewards ignoring them. Lowering the threshold from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 (24 to 38 churners caught) and lowered precision from 0.5455 to 0.4634 (false alarms rose from 20 to 44).

R2.4 Logistic regression versus KNN

| Model (test rows) | F1 | Recall | Precision |
|---|---|---|---|
| Logistic regression, threshold 0.3 | 0.5714 | 0.7451 | 0.4634 |
| KNN, K = 5 (chosen by cross-validation) | 0.3696 | 0.3333 | 0.4146 |

Logistic regression is clearly better on both F1 and recall.

R2.5 Deployment

| Rule | Cost on training rows (INR) | Cost on test rows (INR) |
|---|---|---|
| Logistic, threshold 0.15 (cost_tau) | 12,150 | 6,050 |
| Logistic, threshold 0.5 | not used | 11,800 |
| Flag nobody | not used | 20,400 |

I would deploy the logistic model with threshold 0.15. It costs INR 6,050 on the test rows, against INR 11,800 at threshold 0.5 and INR 20,400 if nobody is flagged. The threshold was chosen on training rows because the test rows must stay unseen. If they chose the threshold, the test cost would be a tuned, optimistic number rather than an honest estimate.

## Part 3 findings

R3.1 Choosing the number of clusters

| K | WCSS | Drop to next K |
|---|---|---|
| 2 | 2262.6753 | 659.8282 (K=2 to 3) |
| 3 | 1602.8471 | 108.0739 (K=3 to 4) |
| 4 | 1494.7732 | 171.0575 (K=4 to 5) |
| 5 | 1323.7157 | 45.5466 (K=5 to 6) |
| 6 | 1278.1691 | not applicable |

The big drop is from 2 to 3 clusters and the next one is much smaller (108.07 is below half of 659.83), so elbow_k = 3 and the elbow supports K = 3. The drops do not shrink steadily (108 and then 171). The pinned starting rows give one reproducible solution, not necessarily the global optimum.

R3.2 Segments (means in original units; X4 and mean_value are in hundreds of INR; outcome_rate is the share of the segment that churned)

| Segment | Size | Tenure | Complaints | Discount share | Avg order (X4) | Loyalty | Outcome rate | Mean value |
|---|---|---|---|---|---|---|---|---|
| A | 322 | 47.5 | 1.95 | 0.225 | 12.53 | 62.2 | 0.0186 | 3.60 |
| B | 255 | 28.6 | 6.07 | 0.453 | 28.72 | 55.0 | 0.2353 | 17.12 |
| C | 223 | 14.4 | 4.23 | 0.712 | 16.51 | 38.5 | 0.4664 | 7.29 |

| Segment | Business name | Action |
|---|---|---|
| A | Loyal full-price veterans (long tenure, few complaints, few discounts, almost no churn) | Reward loyalty with points or early access; do not spend coupons on them |
| B | High-spend, complaint-prone shoppers (largest orders, most complaints) | Fast-track complaint resolution and add a service call, because each lost customer is valuable |
| C | Bargain-focused newcomers (shortest tenure, heaviest discount use, lowest loyalty) | Send retention coupons early in the 90-day window and run an onboarding offer |

The segment with the highest outcome_rate is C, at 0.4664.

R3.3 Hierarchical clustering cut

| Linkage | Clusters at the cut | Members |
|---|---|---|
| Single | 4 | {E1}, {E4}, {E5}, {E2, E3, E6} |
| Complete | 2 | {E1, E4, E5}, {E2, E3, E6} |
| Average | 2 | {E1, E4, E5}, {E2, E3, E6} |

The two rules disagree because single linkage measures the nearest pair. Its merge heights 2.8, 2.9 and 3.4 form a gradual chain with no large gap near the top, so its largest gap (1.4 to 2.8) comes earlier. Complete linkage measures the farthest pair, so joining the two groups costs 7.9 (E1 to E2), which creates a huge final gap and a clean two-group cut.

R3.4 PCA

The mean vector of the standardised data is all zeros (to rounding), so centring changes nothing.

| Predictor | PC1 loading | PC2 loading |
|---|---|---|
| X1 tenure | 0.5317 | 0.1773 |
| X2 complaints | -0.4100 | 0.5056 |
| X3 discount share | -0.5187 | -0.2648 |
| X4 avg order value | -0.2916 | 0.6934 |
| X5 loyalty score | 0.4418 | 0.4025 |

| Component | Eigenvalue | Explained ratio | Cumulative ratio |
|---|---|---|---|
| 1 | 2.6136 | 0.5227 | 0.5227 |
| 2 | 1.1828 | 0.2366 | 0.7593 |
| 3 | 0.5114 | 0.1023 | 0.8616 |
| 4 | 0.4055 | 0.0811 | 0.9427 |
| 5 | 0.2867 | 0.0573 | 1.0000 |

PC1 is driven by tenure (0.5317) and loyalty (0.4418) on the positive side, and by discount share (-0.5187), complaints (-0.4100) and average order value (-0.2916) on the negative side. It contrasts established, loyal customers with discount-driven, complaining ones. Three components are needed to keep at least 80% of the variance (components_for_80pct = 3, cumulative 0.8616; two components give only 0.7593).

| Segment | Mean PC1 score |
|---|---|
| A | 1.7773 |
| B | -0.7831 |
| C | -1.6709 |

PC1 separates segment A clearly from B and C. B and C are closer together on this axis and differ more on the second component.

## Part 4 findings

R4.1 Which split is purest

Root of the 560 training rows: entropy 0.7462, Gini 0.3347.

| Split | n_yes | positives_yes | n_no | positives_no | Information gain | Gini gain |
|---|---|---|---|---|---|---|
| S1 X6 = 1 | 235 | 64 | 325 | 55 | 0.0110 | 0.0052 |
| S2 X7 = 1 | 281 | 52 | 279 | 67 | 0.0033 | 0.0015 |
| S3 X3 > 0.5 | 209 | 87 | 351 | 32 | 0.1046 | 0.0494 |
| S4 X1 <= 24 | 188 | 86 | 372 | 33 | 0.1252 | 0.0606 |

S4 (tenure of 24 months or less) is best by both information gain and Gini gain. It splits the training rows into a yes side where 86 of 188 customers (46%) churn and a no side where 33 of 372 (9%) churn, so its two sides are the purest. A tree grown until every leaf is pure would memorise noise and individual customers in the training set and fail on new ones (overfitting). One rule that stops a tree from growing too deep is a maximum depth, or a minimum number of records per leaf.

R4.2 Which model to ship

| Model (test rows, threshold 0.3 rule) | tp | fp | tn | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Logistic | 38 | 44 | 145 | 13 | 0.4634 | 0.7451 | 0.5714 |
| KNN (K = 5) | 17 | 24 | 165 | 34 | 0.4146 | 0.3333 | 0.3696 |
| Stump (S4) | 39 | 51 | 138 | 12 | 0.4333 | 0.7647 | 0.5532 |
| Hard vote | 35 | 45 | 144 | 16 | 0.4375 | 0.6863 | 0.5344 |
| Soft vote | 38 | 49 | 140 | 13 | 0.4368 | 0.7451 | 0.5507 |

The stump's rates are 0.4574 for customers in the yes side and 0.0887 for the no side. I would ship the logistic model: it has the highest F1 (0.5714) and the joint highest recall among the models other than the stump. Combining models did not help. Neither vote beat the logistic model alone, because the weak KNN pulls the combination down.

R4.3 AdaBoost by hand

| Round | Weighted error | Alpha |
|---|---|---|
| 1 (rule h1) | 0.2500 | 0.5493 |
| 2 (rule h2) | 0.1667 | 0.8047 |

After round 1 normalisation the two misclassified points (points 3 and 7) have 0.2500 each, a total weight of 0.5000. Round 2 trusts h2 more than h1 because h2 has a lower weighted error (0.1667 against 0.2500), since it gets the two heavily weighted points right. The ensemble accuracy on the eight points is 0.7500.

R4.4 Network sizing

Model A, option 2 (convolutional network), input 28 x 28 x 1:

| Layer | Output shape (height x width x depth) | Parameters |
|---|---|---|
| Convolution 5 x 5, 6 filters, stride 1, padding 2 | 28 x 28 x 6 | 156 |
| ReLU | 28 x 28 x 6 | 0 |
| Max-pool 2 x 2, stride 2 | 14 x 14 x 6 | 0 |
| Convolution 5 x 5, 16 filters, stride 1, padding 0 | 10 x 10 x 16 | 2,416 |
| ReLU | 10 x 10 x 16 | 0 |
| Max-pool 2 x 2, stride 2 | 5 x 5 x 16 | 0 |
| Flatten | 400 | 0 |
| Fully connected 32 neurons, ReLU | 32 | 12,832 |
| Fully connected 10 neurons (output) | 10 | 330 |
| Total (cnn_total) | | 15,734 |

| Comparison | Parameters |
|---|---|
| Fully connected network 784 - 16 - 16 - 10 (mlp_total) | 13,002 |
| Convolutional network, option 2 (cnn_total) | 15,734 |
| Fully connected layer, 128 x 128 x 3 input to 1,024 neurons (fc_layer_params) | 50,332,672 |
| Convolutional layer, 64 filters of 3 x 3 on the same input (conv_layer_params) | 1,792 |

The convolutional layer needs far fewer parameters because the same small filters are reused across the whole image.

Model B (recurrent network, 64-dimensional input, 128-dimensional hidden state, 16 classes):

| Part | Shape | Parameters |
|---|---|---|
| W_xh | 128 x 64 | 8,192 |
| W_hh | 128 x 128 | 16,384 |
| W_hy | 16 x 128 | 2,048 |
| b_h | 128 | 128 |
| b_y | 16 | 16 |
| Total (rnn_total) | | 26,768 |

rnn_total does not grow when the sequences get longer because the same weights are reused at every time step. Only the number of steps changes, not the number of parameters.

| Choice for 10 classes | Selection |
|---|---|
| Hidden-layer activation | ReLU |
| Output activation | Softmax |
| Loss | Categorical cross-entropy |

In a very deep plain network the gradient is multiplied by many small factors on its way back, so it shrinks toward zero and the early layers barely learn (vanishing gradient). A skip connection adds a layer's input to its output, giving the gradient a short direct path back to the early layers.

## Model-choice memo

To: Head of the Retention Team

Recommendation: ship the logistic-regression model and flag a customer when the predicted churn probability is at least 0.15.

On the 240 held-out test customers (51 real churners), this costs INR 6,050, using INR 50 per wasted coupon and INR 400 per lost customer. The usual 0.5 cut-off costs INR 11,800 and flagging nobody costs INR 20,400, so 0.15 cuts cost by about 49% and 70% respectively. The threshold was chosen on training data only (training cost INR 12,150), so the test figure is an honest check.

Logistic regression also leads on quality. At threshold 0.3 its F1 is 0.5714 and recall 0.7451, against nearest-neighbours at F1 0.3696 and recall 0.3333. Voting the models together did not beat it (hard vote F1 0.5344, soft vote 0.5507). A one-question tenure rule (24 months or less) came close (F1 0.5532), but its INR cost was not computed with the cost rule.

Risk: the evidence is one split with only 51 churners in the test set, on synthetic data, so the INR savings could shift noticeably on real customers. Re-check the threshold on fresh data before launch.

Neither the image model (Model A) nor the review-text model (Model B) should be built before this churn model is live, because only the churn model has a measured INR saving.

## Limitations

1. Synthetic data. Every record comes from a generator, so the patterns are cleaner than real customer behaviour, and no conclusion applies to actual Flipkart customers.
2. Single split and few positives. All test results use one fixed 70/30 split with only 51 positive test records, so one or two customers change recall by several points. The best K and the threshold 0.15 are valid for this split only.
3. Assumed costs. The INR 50 and INR 400 costs are rough figures supplied by the brief. A different cost ratio would move the best threshold, and a low threshold sends many false alarms (44 at threshold 0.3, more at 0.15).
4. Model simplicity. Logistic regression assumes a straight-line effect on the log-odds, K-means from fixed starting rows may not reach the global optimum, and the log-log regression predicts a typical (median-like) order value, not the average, unless a correction is applied.
5. Small hierarchical example. The linkage analysis uses a supplied six-cohort table, so the cut depends heavily on a few distances.
