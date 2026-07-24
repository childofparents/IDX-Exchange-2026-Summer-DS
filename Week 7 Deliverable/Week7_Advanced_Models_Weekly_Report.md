# Week 7 Advanced Models Weekly Report

## Evaluation design

The Week 3 preprocessing and Week 6 feature-engineering pipelines were rerun
from the monthly CSV files after shifting the timeline forward one month.
Training uses 2025-06 through 2026-05 (130,060
rows after train-only target handling), and the latest month, 2026-06,
is an untouched test set of 12,857 California single-family residence
sales. Listing-process leakage fields were excluded. Hyperparameters were
selected with complete-month chronological validation ending before the test
month.

## Gradient boosting comparison

The selected XGBoost and LightGBM configurations were chosen by mean validation
MdAPE, with validation R2 as a tie-breaker. On June 2026, the strongest booster
was **XGBoost (depth 8, lr 0.05, 500 trees)**: R2 0.812, MAE
$232,373, RMSE $666,903,
MAPE 25.2%, and MdAPE
11.8%. Its training R2 was
0.935.

XGBoost and LightGBM both learn nonlinear interactions and additive corrections
to prior trees. XGBoost grows regularized trees level-wise and is deliberately
conservative here through row/column subsampling and L2 regularization.
LightGBM's leaf-wise histogram growth is faster and can capture sharp local
interactions, but it can overfit without depth, leaf, and child-size controls.
The reported validation dispersion and train/test gap therefore matter as much
as a small metric difference.

## Training-window finding

No May-only proxy window dominated every metric. The 3-month window had the
lowest MdAPE (11.0%); nine months had the lowest MAE; and eleven months had the
best R2 and RMSE. The final fit uses the assignment's required twelve-month
shifted window, extending the strongest variance/tail-risk proxy to a full
seasonal cycle and the largest recent sample without using any June outcome.
Window length should be rechecked in a multi-month rolling backtest as more
monthly extracts arrive.

## Comparison with models explored in prior weeks

For a fair same-month comparison, the Week 6 Random Forest configuration was
retrained on the shifted W3/W6 matrix. Its June metrics were R2
0.749, MAE
$232,646, MAPE
23.7%, and MdAPE
9.1%. Historical Week 4-6 tables used May,
so they are included for lineage but not treated as an apples-to-apples ranking.

Based on the latest same-month comparison, **XGBoost (depth 8, lr 0.05, 500 trees)** is the
best overall statewide production candidate among the explored model families:
it leads on R2, RMSE, and narrowly on MAE, and its June result closely matches
its rolling validation behavior. The Random Forest remains better for typical
percentage error and should be retained as a benchmark or evaluated as an
ensemble component. This is a deliberate tradeoff, not a claim that XGBoost
wins every metric.

## Generalizability interpretation

The recommendation is evidence-based but conditional. A single June holdout
simulates the next production month and is more honest than a random split, yet
it cannot prove consistent accuracy for every future California home. The model
should be promoted with monthly rolling backtests, county/ZIP and price-band
monitoring, prediction-interval or uncertainty work, and retraining/drift
triggers. Unseen ZIP/district categories are safely routed by the encoder, but
localized sparse markets and luxury homes remain the highest-risk segments.
