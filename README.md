# Mid-Price Predictability in Event-Driven Order Book Data

## Executive Summary

This project studies short-horizon predictability of future mid-price returns in event-driven order book data.

The prediction target is defined as the fixed-horizon future log-return of mid-price at **1s / 5s / 10s in calendar time**, which is the correct formulation for event-driven market data.  
Given compute constraints and as explicitly allowed by the assignment, the main modeling stage is restricted to the **main trading session of the first trading date (`2025-06-03`)**. The experimental design uses a strict time-ordered **train / validation / test** split with a **10-second purge / embargo** around split boundaries to reduce leakage for fixed-horizon targets.

Main findings:

- the strongest horizon is **1 second**;
- the strongest simple and most interpretable baseline is the **calibrated order-book imbalance model**;
- zero prediction remains competitive in MAE because the target is highly concentrated around zero, so **information-coefficient metrics** are more informative than point-error metrics alone;
- adding **Ridge** as an intermediate supervised benchmark shows that richer linear combinations can look strong on validation but are **not robust out of sample** in the current setup;
- the strongest default nonlinear model is a **LightGBM using only top-of-book (`L1_only`) features**;
- default LightGBM is more stable than Ridge on held-out test, but still does **not beat the imbalance baseline**;
- a focused **Optuna-tuned LightGBM** achieves the strongest held-out **Pearson IC** in the current purged single-session experiment and exceeds the imbalance baseline on that metric.

Overall, the data does contain meaningful short-horizon predictive structure. Most of the robust signal is already captured by simple top-of-book and imbalance information, while the additional nonlinear uplift from a tuned LightGBM looks promising but still requires validation across additional trading days and market regimes.

---

## Objective

The goal of the project is to evaluate whether future mid-price movements are predictable at short horizons using event-driven order book features.

The assignment explicitly asks for:

- analysis of future mid-price predictability;
- target construction at horizons such as **1s / 5s / 10s**;
- regression as the main task;
- optional classification analysis;
- and, if compute is limited, the option to restrict analysis to a subset such as **one day or one main trading session**.

This project focuses on the regression setup and implements a full experimental pipeline:

1. data audit;
2. target construction;
3. baseline comparison;
4. nonlinear model ablation;
5. optional hyperparameter tuning.

---

## Data and Experimental Scope

The raw data consists of incremental order-book updates with nanosecond timestamps.  
This means that “the next row” is often just microstructure noise, so the target must be defined in **calendar time**, not in event-count space.

### Full raw dataset

The original dataset contains three daily parquet files with a total of **3,668,240 rows**.

### Audit findings

The audit stage established that:

- timestamps are monotonic within files;
- top-of-book variables are internally consistent;
- `mid_price` matches the arithmetic mid of best bid and best ask;
- inverted quotes are absent;
- spread is almost always positive;
- `file_date` behaves like a **trading-date anchor** rather than the literal calendar date of every row.

### Experimental subset

To keep the modeling stage computationally stable, and as explicitly allowed by the assignment, the main experiment is restricted to:

- **trading date:** `20250603`
- **session:** `main`

This subset contains **987,938 rows** and corresponds to a single coherent and liquid market regime.

---

## Methodology

### 1. Data audit
The raw data is checked for:

- timestamp monotonicity;
- session structure;
- spread consistency;
- mid-price reconstruction consistency;
- missing values and basic schema sanity.

### 2. Target construction
The target is defined as:

\[
y_H(t) = \log\left(\frac{\text{mid}(t+H)}{\text{mid}(t)}\right)
\]

for:

- `H = 1s`
- `H = 5s`
- `H = 10s`

Targets are built in **calendar time** inside the selected session.

### 3. Split design
A strict time-ordered split is used inside the session:

- **train:** first 60%
- **validation:** next 20%
- **test:** final 20%

In addition, a **10-second purge / embargo** is applied around the train/validation and validation/test boundaries, with the purge window set equal to the maximum forecast horizon.

This preserves temporal causality and reduces leakage risk for fixed-horizon targets.

### 4. Baselines
The following benchmark models are evaluated:

- **zero baseline**
- **calibrated microprice baseline**
- **calibrated imbalance baseline**
- **Ridge regression** as an intermediate supervised linear benchmark

Evaluation metrics include:

- `RMSE`
- `MAE`
- `R²`
- `Pearson IC`
- `Spearman IC`

### 5. Nonlinear model ablation
A LightGBM model is evaluated on cumulative feature blocks:

- `L1_only`
- `L1_plus_imb_micro`
- `plus_flow`
- `all_blocks`

The strongest horizon from the baseline stage is fixed for nonlinear modeling.

### 6. Optional tuning
A focused Optuna search is run on the best validation feature block to test whether the default LightGBM configuration is responsible for weak out-of-sample behavior.

---

## Results

### Baseline stage

Key baseline findings:

- the strongest horizon is **1 second**;
- the most robust simple predictor is the **calibrated imbalance baseline**;
- the zero predictor remains competitive in **MAE**, which is expected because the target is strongly concentrated near zero;
- therefore, **information coefficient** is the most informative metric for this task.

### Nonlinear ablation

After introducing the purged split and adding Ridge as an intermediate supervised benchmark, the default nonlinear comparison becomes more informative.

The strongest default LightGBM validation feature block is:

- **`L1_only`**

This suggests that the most robust default nonlinear signal is concentrated in **top-of-book information**, while broader feature sets do not improve out-of-sample stability in the current setup.

Ridge provides a useful diagnostic benchmark:

- it can achieve strong validation Pearson IC with richer feature blocks;
- however, that validation uplift does **not** generalize robustly to held-out test.

In contrast, default LightGBM is more stable than Ridge out of sample, but still remains below the calibrated imbalance baseline on held-out test.

### Optional Optuna tuning

A focused Optuna search is run on the best default LightGBM feature block.

The tuned nonlinear model reaches the strongest held-out **Pearson IC** in the current purged single-session experiment and improves materially over the untuned default LightGBM. On this metric, it also exceeds the calibrated imbalance baseline.

At the same time, this result should be interpreted carefully:

- the experiment is still restricted to a **single trading session**;
- tuning is performed against a single validation slice;
- and the analysis does not yet establish **cross-day robustness**.

Therefore, the tuned LightGBM result is best interpreted as **promising evidence of additional nonlinear signal**, rather than a fully established robust edge.

### Segment analysis

Model quality is strongest in the **tightest-spread regime** and deteriorates in wider-spread states.

This suggests that any residual nonlinear signal is conditional on specific market conditions rather than uniformly stable across the whole session.

---

## Main Conclusions

The main conclusions of the project are:

1. **The strongest horizon is 1 second.**
2. **The calibrated order-book imbalance model remains the strongest simple and most interpretable benchmark.**
3. **Zero prediction is competitive in MAE but not informative in IC, so information-coefficient metrics are more appropriate in this task.**
4. **Adding Ridge as an intermediate supervised linear benchmark shows that stronger validation ranking does not necessarily translate into robust out-of-sample performance.**
5. **The strongest default nonlinear model is LightGBM on `L1_only` features, which suggests that the most robust nonlinear signal is concentrated in top-of-book information.**
6. **In its default form, LightGBM does not beat the imbalance baseline on held-out test.**
7. **After focused hyperparameter tuning, LightGBM achieves the strongest held-out Pearson IC in the current purged single-session setup and exceeds the imbalance baseline on that metric.**
8. **However, this nonlinear uplift should still be treated as promising rather than definitive until it is validated across additional trading days and market regimes.**

### Final takeaway

The data contains meaningful short-horizon predictive structure, strongest at the 1-second horizon.  
A simple imbalance-based model remains the strongest baseline and the most interpretable reference point.  
Ridge does not generalize out of sample in the current setup.  
A carefully regularized LightGBM can achieve stronger held-out ranking quality on the purged single-session test, but confirming that result requires broader robustness checks beyond a single day.

---

## Limitations

The project has several important limitations:

- the main modeling stage is restricted to a **single main session** rather than multiple trading days;
- therefore, the analysis does not establish full **day-to-day robustness**;
- evaluation focuses on predictive-quality metrics rather than a full execution-aware trading backtest;
- the strongest nonlinear result comes from a **tuned model evaluated on a single held-out test slice**, so broader robustness validation is still needed;
- Ridge shows strong validation ranking in some feature blocks, but that uplift does not generalize robustly out of sample;
- the classification branch was not included in the core scope to keep the solution focused and computationally stable.


## Repository Structure

```text
project_root/
├─ data/
├─ notebooks/
│  ├─ 01_data_audit.ipynb
│  ├─ 02_target_building.ipynb
│  ├─ 03_baselines.ipynb
│  └─ 04_model_ablation.ipynb
├─ src/
│  ├─ io.py
│  ├─ session_utils.py
│  ├─ audit.py
│  ├─ target_building.py
│  ├─ splits.py
│  ├─ baselines.py
│  ├─ metrics.py
│  └─ models.py
├─ outputs/
│  ├─ datasets/
│  ├─ tables/
│  └─ figures/
├─ README.md
├─ requirements.txt
└─ config.yaml
```
## Reproducibility

Recommended execution order:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run notebooks in order:
- 01_data_audit.ipynb
- 02_target_building.ipynb
- 03_baselines.ipynb
- 04_model_ablation.ipynb

Before final submission, all notebooks should be rerun cleanly from top to bottom so that outputs correspond exactly to the current code and the final experimental scope.
