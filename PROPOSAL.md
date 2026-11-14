# Proposal

## Time Series Forecasting and Distribution Shift Detection in Dynamic Agent-Based Systems

**CPS310 Zombie Game · Stage 3**  
**Daniel O’Keefe, Owen Bowers, and Rahsaun Jeffrey**

## Background and problem statement

The CPS310 Zombie Game is an agent-based simulation in which humans and zombies interact over a series of simulation ticks. These interactions produce a time series of population counts. During normal gameplay, zombies use Thompson Sampling to choose among three strategies. The simulation also includes an Adaptive Kalman Filter that uses recent zombie counts to estimate the current population trend and forecast future counts.

When the agents’ strategy changes, the patterns that the forecasting model has been using may change as well. This is a possible **distribution shift**. Forecast errors can also increase temporarily during ordinary gameplay. We want to determine whether a sustained increase in forecast error can help identify a change in the simulation’s behavior.

## Research question

**Can unusually large zombie-population forecast errors signal a change in agent strategy, and how does the forecasting window affect prediction accuracy, detection delay, and false alarms?**

## Preliminary evidence

In the existing 10-tick forecasting experiment, the Adaptive Kalman Filter achieved its lowest mean absolute error with a 50-tick history window. A separate experiment compared five matched pairs of runs in which zombies either continued using Thompson Sampling or switched to a rush-only strategy at tick 150. The persistence baseline, which predicts that the zombie count will stay at its current value, had higher average errors in the switched runs across all three forecast horizons. These findings motivate testing whether an alert based on forecast errors can identify a shift while the simulation is running. [Forecast accuracy figure](https://github.com/dokeefe414/CPS310-Zombie-Game/blob/main/figures/forecasting/mae_h10.png) · [Strategy-switch figure](https://github.com/dokeefe414/CPS310-Zombie-Game/blob/main/figures/forecasting/strategy_shift_comparison.png)

## Proposed methodology

1. **Generate paired simulation runs.** We will use the existing headless experiment runner to start each pair with the same random seed, 1,000 agents, and the same settings. One run will retain Thompson Sampling. The other will switch to `rush_only` at tick 150. We will record the zombie count at each tick for up to 300 ticks and report any runs that end early.
2. **Compare forecasts.** We will evaluate the Adaptive Kalman Filter with 20-, 50-, and 100-tick history windows at forecast horizons of 10, 25, and 50 ticks. At each tick, the model will use only observations available at that time. Once the target tick arrives, we will compare the prediction with the observed count and score the persistence baseline on the same run.
3. **Test a shift detector.** For each forecasting window, we will start with 10-tick-ahead predictions and calculate the rolling average of the latest 10 absolute forecast errors. We will calibrate an alert threshold using runs with no strategy switch, then freeze it before testing on separate seeds. The detector will raise an alert when the rolling average crosses the threshold. It will use forecast errors as they become available. The known switch tick will be used afterward to score the alerts.
4. **Evaluate forecasting and detection.** We will report mean absolute error and root mean squared error for forecasting. For detection, we will report the share of switched runs with a post-switch alert, the delay from the switch to the first post-switch alert, and the share of control runs with any alert. We will also count pre-switch alerts in intervention runs and report missed detections.

## Expected deliverables and limitations

We will produce a reproducible experiment configuration, a table comparing forecast and detection results, and plots showing observed counts, forecasts, and alert timing for selected matched runs. A strategy change may have little effect on the total zombie count. In that case, the experiment will help establish when population counts alone provide too little information for reliable detection.
