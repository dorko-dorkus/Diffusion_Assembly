# Surrogate AI Model

Target: Predict assembly index A*(x|G,P).
Training: QM9-CHON split, labels from exact AI (timeout 1s/molecule; label drop if timeout).
Validation: 5-fold CV. Metrics: MAE, RMSE. Target MAE ≤ 0.5 AI units.

Uncertainty: Monte Carlo dropout (T=20) to estimate σ_pred. During guidance, clamp λ if σ_pred > σ_max.

Artifacts saved per training:
- model.pt
- cv_metrics.json
- calibration_plot.png

During evaluation runs the pipeline additionally writes an `ai_calibration.csv`
file containing quantiles of the absolute error between surrogate predictions
and Monte Carlo estimates on a subset of QM9 molecules.  The file is stored
alongside `metrics.json` for each run.
