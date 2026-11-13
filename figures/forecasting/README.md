# Forecasting research figures

Upload only selected PNG figures to this folder. The root README contains
the matching image placeholders.

Expected names:
- `simulation_forecast.png`: screenshot of the live GUI with orange forecast.
- `population_trajectory.png`: human and zombie counts over time.
- `mae_h10.png`: 10-tick forecast versus naive baseline.
- `mae_h25.png`: 25-tick forecast versus naive baseline.
- `mae_h50.png`: 50-tick forecast versus naive baseline.
- `strategy_shift_comparison.png`: future plot comparing pre/post shift results.

Generate the standard graphs locally with:

```powershell
py plot_experiments.py --input results/forecasting_v2
```

Then copy the selected PNGs from the results folder into this directory.
The strategy-shift comparison needs a dedicated plot; do not relabel an
existing MAE chart as a shift comparison.

No image assets are checked in yet. Raw experiment results are excluded
by `.gitignore`; the figures in this folder can be committed normally.
