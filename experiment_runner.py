"""Reproducible, headless simulation and forecast experiments.

Examples:
  py experiment_runner.py --agents 1000 --ticks 200 --seeds 1 2 3
  py experiment_runner.py --agents 5000 --ticks 100 --seeds 1 --windows 20 50 100
"""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
from forecasting import SlidingWindowAdaptiveKalman, ForecastTracker
from Zombie_Game import SimulationPlayground


def evaluate_series(counts, windows=(20, 50, 100), horizons=(10, 25, 50), capacity=None):
    """Paired online evaluation: same observed series for every model setting."""
    if not counts:
        raise ValueError("counts must not be empty")
    output = []
    for window in windows:
        model = SlidingWindowAdaptiveKalman(window=window, capacity=capacity)
        trackers = {h: ForecastTracker(horizon=h) for h in horizons}
        start = time.perf_counter()
        for tick, count in enumerate(counts):
            model.update(count)
            predictions = model.predict(max(horizons))
            for horizon, tracker in trackers.items():
                tracker.record(tick, count, predictions[horizon - 1], count)
        elapsed = time.perf_counter() - start
        for horizon, tracker in trackers.items():
            output.append({"window": window, "horizon": horizon,
                           "forecast_seconds": elapsed, **tracker.metrics()})
    return output


def run_simulation(agents, ticks, seed, workers=1):
    """One seed-controlled run, with complete tick-by-tick population history."""
    if agents < 3 or ticks < 1:
        raise ValueError("agents must be >=3 and ticks >=1")
    random.seed(seed)
    np.random.seed(seed)
    regular = round(agents * .75)
    armed = round(agents * .05)
    zombies = agents - regular - armed
    sim = SimulationPlayground(use_multiprocessing=workers > 1, workers=workers)
    sim.populate(regular, armed, zombies)
    records = [{"tick": 0, "humans": len(sim.humans), "zombies": len(sim.zombies)}]
    start = time.perf_counter()
    try:
        for _ in range(ticks):
            if not sim.humans or not sim.zombies:
                break
            sim.step()
            records.append({"tick": sim.tick_count, "humans": len(sim.humans),
                            "zombies": len(sim.zombies)})
    finally:
        sim.close()
    return records, time.perf_counter() - start


def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=1000)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--windows", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="results/forecasting")
    args = parser.parse_args(argv)
    if min(args.windows) < 3 or min(args.horizons) < 1 or args.workers < 1:
        parser.error("windows >=3, horizons >=1, workers >=1 required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in args.seeds:
        history, runtime = run_simulation(args.agents, args.ticks, seed, args.workers)
        write_csv(out / f"population_seed_{seed}.csv", history)
        for metric in evaluate_series([r["zombies"] for r in history],
                                      args.windows, args.horizons, capacity=args.agents):
            summary.append({"seed": seed, "agents": args.agents,
                            "requested_ticks": args.ticks,
                            "actual_ticks": len(history) - 1,
                            "simulation_seconds": runtime, **metric})
        print(f"seed={seed} ticks={len(history)-1} runtime={runtime:.2f}s")
    write_csv(out / "metrics.csv", summary)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved full tick history, metrics and config in {out}")
    valid = [r for r in summary if r["n"]]
    if valid:
        print("Window | Horizon | MAE | Naive MAE | RMSE | N")
        for r in valid:
            print(f'{r["window"]:>6} | {r["horizon"]:>7} | {r["mae"]:>6.2f} | '
                  f'{r["naive_mae"]:>9.2f} | {r["rmse"]:>6.2f} | {r["n"]}')


if __name__ == "__main__":
    main()
