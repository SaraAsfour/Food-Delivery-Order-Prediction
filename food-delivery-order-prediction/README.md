# User Behavior Prediction in Food Delivery Applications


Predicting whether a food-delivery app session ends in an order (`order_placed = 1`) from session-level behavior: timestamps, cart activity, promotions and device information.


## Results

| Model | AUC | F1 |
|---|---|---|
| Logistic Regression | 0.9624 | 0.4284 |
| K-Nearest Neighbors | 0.9223 | 0.3902 |
| Decision Tree | 0.9407 | 0.4404 |
| Random Forest | 0.9668 | 0.4558 |
| **LightGBM (final)** | **0.9730** | **0.5180** |

The final LightGBM model reaches an **out-of-fold AUC of 0.9730** and a best F1 of 0.518 at a decision threshold of 0.24.

## Dataset

- 297,236 training sessions and 99,639 test sessions, with 17 raw features
- Heavy class imbalance: only **2.93%** of sessions result in an order
- 90,518 sessions have no promotion shown. Their promotion fields are missing for a structural reason, not because of random gaps.

## Approach

1. **EDA:** The strongest signals were cart items and cart value (f10, f11), the response to a promotion (f17), session duration, mealtime hours, and whether the customer was returning (f9).
2. **Preprocessing:** Timestamps were parsed into durations. Missing promotion fields were filled with `NONE` or 0. Infinite values produced by ratio features were replaced with NaN.
3. **Feature engineering:** About 60 features in six groups:
   - time
   - cart
   - promotion
   - interactions
   - frequency encoding
   - fold-safe target encoding
4. **Encoding:** Scikit-learn baselines used one-hot encoding plus `StandardScaler`. LightGBM used its native categorical handling.
5. **Modeling:** Five models were compared, from simplest to most complex. The final LightGBM was trained with 5-fold stratified cross-validation × 10 seeds, giving 50 models whose predictions were averaged.

### Final LightGBM hyperparameters

| Parameter | Value |
|---|---|
| n_estimators | 4000 (with early stopping after 100 rounds) |
| learning_rate | 0.02 |
| num_leaves | 63 |
| min_child_samples | 30 |
| subsample / colsample_bytree | 0.8 / 0.8 |
| reg_alpha / reg_lambda | 0.1 / 1.0 |
| Target-encoding smoothing | 100 |

## Key challenges

- **Target-encoding leakage:** Encodings were computed only on the training part of each fold, and smoothing was applied to rare categories.
- **Class imbalance:** Scikit-learn models used `class_weight="balanced"` or `"balanced_subsample"`. Model selection used AUC rather than accuracy.
- **KNN scalability:** KNN was trained on a stratified subsample of 15,000 sessions.

## Repository structure

```
├── data/          # dataset goes here 
├── figures/       # plots used 
├── notebooks/     # project notebook
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
unzip train.csv in data/, then open the notebook
jupyter notebook notebooks/
```


