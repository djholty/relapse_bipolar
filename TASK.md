# Task Tracking

## Completed Tasks

### 2026-02-04
- [x] Fix NameError in multiple model cells - removed undefined baseline variables (`auroc_lr_simple`, `auprc_lr_simple`, `auroc_xgb_simple`, `auprc_xgb_simple`, `fpr_lr2`, `tpr_lr2`, `rec_lr2`, `prec_lr2`) that were never defined. Fixed 5 cells total:
  - Cell 23: "MODEL: 2 Sensor Features + Demographics" - completely rewrote to remove comparison section
  - Cell 27: "MODEL: 2 Sensor Features + Sleep-Verified HRV" - removed undefined variable references
  - Cell 31: "MODEL: Add Circadian Features to Best Model" - removed undefined variable references  
  - Cell 32: Unnamed cell - removed undefined variable references
  - Cell 39: Unnamed cell - removed undefined variable references

- [x] Created comprehensive XGBoost model in cell 23 - uses all sensor features (HRV, sleep, steps) plus demographics to predict relapse with AUROC and AUPRC metrics. Includes feature importance analysis and visualizations.

- [x] Updated cell 27 to comprehensive model combining ALL features:
  - Merges sleep-verified HRV features with combined_with_demo dataset
  - Combines nighttime HRV (from cell 16) + sleep-verified HRV (from cell 24) + all sleep features + all step features + demographics
  - Trains both XGBoost and Logistic Regression models
  - Feature importance analysis categorized by type (Nighttime HRV, Sleep-Verified HRV, Sleep, Steps, Demographics)
  - Comprehensive visualizations: ROC curves, PR curves, feature importance bar chart with color-coding by type, pie chart showing importance by feature type
  - Shows top 25 most important features with their type labels
  - Displays feature importance summary by type with percentages

## Discovered During Work
- None

- [x] Converted circadian features to percentage-based calculation (cell 29):
  - Updated baseline calculation to compute percentage distributions (mean and std per hour) instead of raw counts
  - Added 4 new percentage-based features:
    - `night_activity_proportion`: % of daily activity during 0-6 AM
    - `day_activity_proportion`: % of daily activity during 9 AM-9 PM
    - `circadian_deviation_pct`: Overall deviation using percentage z-scores
    - `circadian_shift_pct`: Shift in peak activity hour (by percentage)
  - Kept 4 original count-based features for comparison
  - Changed cache filename to `circadian_features_pct_v2.parquet` to force recomputation
  - Benefits: More robust to overall activity level changes, more interpretable clinically, better sensitivity to rhythm shape changes

- [x] Updated circadian analysis cell (cell 30) to analyze all 8 features:
  - Analyzes both percentage-based features (night_activity_proportion, day_activity_proportion, circadian_deviation_pct, circadian_shift_pct) and count-based features
  - Compares predictive power (AUC) between percentage-based vs count-based approaches
  - Shows which feature type performs better on average
  - Creates 2x4 grid of visualizations (top row = percentage-based, bottom row = count-based)
  - Provides recommendations for which features to use in final models based on AUC results
  - Includes clinical interpretation of findings

- [x] Updated circadian visualizations to use density plots (cell 30):
  - Added `density=True` parameter to all 8 histogram calls
  - Changed y-axis labels from 'Count' to 'Density'
  - Both relapse and non-relapse distributions now normalized to sum to 1.0
  - Makes visual comparison meaningful despite class imbalance (many more non-relapse days)
  - Easier to see if distribution shapes differ between relapse and non-relapse days

- [x] Fused gyroscope and linear accelerometer for circadian features (Option 1B, cell 29):
  - Added `use_linacc=True`, `w_gyr=0.5`, `w_lin=0.5` to `process_patient_circadian`
  - Loads both gyr.parquet and linacc.parquet per split; merges on `day_index` and `time`
  - Fused magnitude = w_gyr * mag_gyr + w_lin * mag_lin; falls back to gyr-only if linacc missing or merge empty
  - Cache path updated to `circadian_features_fused_v1.parquet`; docstring describes Option 1B

- [x] Fixed circadian gyr/linacc time alignment (cell 29):
  - Inner merge on exact (day_index, time) was dropping almost all rows (e.g. P1: 297k gyr + 312k lin -> 250 rows), so P1/P3 showed only 1 day.
  - Align by rounding time to 1s: add time_key with pd.to_datetime(...).dt.round('1s') for gyr and linacc.
  - Aggregate linacc by (day_index, time_key); left-merge gyr to linacc so all gyr rows kept.
  - Magnitude: fillna(0) for missing linacc, then magnitude = w_gyr*mag_gyr + w_lin*mag_lin; hourly aggregation unchanged.
  - Cache bumped to `circadian_features_fused_v2.parquet` so next run recomputes with new alignment.

- [x] Added 7 clinically-validated actigraphy features to cell 29 (2026-02-27):
  - `_compute_l5_m10`: helper computing L5 (least-active 5h) and M10 (most-active 10h) via circular sliding window
  - `_compute_cosinor`: helper fitting 24h cosine model to hourly % activity profile
  - `process_patient_circadian` now returns a third value `baseline_stats` with scalar RA, cosinor amp/acrophase, and L5/M10 onset means and stds computed over all non-relapse baseline days
  - `calculate_circadian_features` now accepts `baseline_stats` and computes 7 new features:
    - `evening_activity_proportion`: % activity 18–23h (mania prodrome signal)
    - `relative_amplitude_zscore`: (M10−L5)/(M10+L5) z-scored vs personal baseline (depression depth)
    - `l5_onset_deviation`: circular deviation of least-active 5h onset from baseline (sleep-phase shift)
    - `m10_onset_deviation`: circular deviation of most-active 10h onset from baseline (active-phase shift)
    - `intradaily_variability`: within-day fragmentation metric (bipolar marker)
    - `cosinor_amplitude_zscore`: rhythm strength vs personal baseline
    - `cosinor_acrophase_deviation`: circular phase shift vs personal baseline
  - Cache bumped from `circadian_features_fused_v2.parquet` → `circadian_features_fused_v3.parquet`
  - Total features: 15 (was 8)

- [x] Added Fused Self-Supervised + Supervised Transformer cell (2026-02-28):
  - New cell `0f7g2kpev2e` appended after cell `r6oz8ujp5w` (last cell)
  - Implements Bumblebee-inspired pre-train → fine-tune paradigm
  - SharedEncoder (input_proj → pos_embed → TransformerEncoder 2L → bottleneck): Linear(N_FEAT→32) → Linear(32→16)
  - TransformerAE (Stage 1): encoder + TransformerDecoder reconstruct 7-day windows; MSE loss on non-relapse days
  - FusedClassifier (Stage 2): encoder (pretrained or random) + Linear(16→1) head; BCEWithLogitsLoss
  - LOPO loop: for each held-out patient: AE on non-relapse val days → fused fine-tune → supervised-from-scratch baseline
  - Prints 3-row comparison table: AE-only / Supervised (fresh) / Fused (pre-train→finetune)
  - Architecture smoke-tested (all tensor shapes verified)

- [x] Rewrote cell `0f7g2kpev2e` with Dual-Stream Transformer Fusion + Separate HP Tuning (2026-02-28):
  - Replaced single-encoder pre-train→fine-tune with two independent encoders
  - **AE stream**: `TransformerAE` trained on all ~64 numeric features (non-relapse reconstruction)
  - **Sup stream**: `SupClassifier` trained on 14 Boruta-confirmed features (supervised BCE)
  - **DualStreamFusion**: frozen AE + Sup encoders concatenated; small FC fusion head trained
  - **Phase 1**: HP tune AE — 4 configs × 9 folds, score by reconstruction AUROC
  - **Phase 2**: HP tune Sup — 4 configs × 9 folds, score by classification AUROC
  - **Phase 3**: Fusion LOPO — best configs combined; per-fold prints AE/Sup/Fusion AUROC
  - HP grid: small(d=32,lat=16), medium(d=64,lat=32), large(d=128,lat=64), deep(d=64,lat=32,L=3)
  - Separate scalers per stream; `make_windows` called with different feature_cols per stream
  - Expected runtime ~3 min on CPU

- [x] Added AE + iNNE Anomaly Detection cell (2026-03-01):
  - New cell 50 appended after cell 49 (Dual-Stream Transformer Fusion)
  - Inspired by "Bumblebee Your Way" paper: MLP autoencoder trained on non-relapse days, iNNE (Isolation Nearest Neighbor Ensembles) from PyOD scores test days as anomalies
  - Purely unsupervised — AE never sees relapse labels during training
  - `MLP_AE`: n_features → n_features (hidden) → 16 (latent) → n_features (hidden) → n_features; MSE reconstruction loss, Adam lr=1e-3, 100 epochs, batch=32
  - LOPO loop: fit scaler on normal train, train AE on non-relapse val days, extract Z_normal/Z_test, fit INNE(n_est=200) on Z_normal, score Z_test
  - Ablation: also runs INNE directly on raw scaled features (no AE bottleneck) to measure AE contribution
  - Caching: AE weights per fold → `cache/ae_inne_models/ae_fold_{patient}.pth`; LOPO results → `cache/ae_inne_lopo_results.pkl`
  - Auto-installs pyod if missing; uses `feature_cols` + `combined_all_circadian` from cell 48

## Pending Tasks
- None
