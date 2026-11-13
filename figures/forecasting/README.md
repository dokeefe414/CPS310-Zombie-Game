# Forecasting research figures

All six selected PNG figures are uploaded and embedded in the root README.

Expected names:
- `Simulation_forecast.png`: screenshot of the live GUI with orange forecast.
- `population_trajectory.png`: human and zombie counts over time.
- `mae_h10.png`: 10-tick forecast versus naive baseline.
- `mae_h25.png`: 25-tick forecast versus naive baseline.
- `mae_h50.png`: 50-tick forecast versus naive baseline.
- `strategy_shift_comparison.png`: paired control-versus-intervention persistence-error comparison.

Generate the standard graphs locally with:

```powershell
py plot_experiments.py --input results/forecasting_v2
```

Then copy the selected PNGs from the results folder into this directory.
The strategy-shift comparison is a separate persistence-error plot; do not
interpret it as a direct comparison of Adaptive Kalman model errors.

The six uploaded PNGs are tracked on the forecasting feature branch.
Raw experiment results are excluded by `.gitignore`; future selected
figures in this folder can be committed normally.
