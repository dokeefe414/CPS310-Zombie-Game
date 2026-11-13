"""Create research figures from experiment_runner.py CSV output.

Usage: py plot_experiments.py --input results/forecasting
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/forecasting")
    args = parser.parse_args(argv)
    folder = Path(args.input)
    with (folder / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if int(r["n"]) > 0]
    if not valid:
        raise SystemExit("No completed forecasts. Increase --ticks or reduce --horizons.")
    horizons = sorted({int(r["horizon"]) for r in valid})
    for horizon in horizons:
        subset = [r for r in valid if int(r["horizon"]) == horizon]
        windows = sorted({int(r["window"]) for r in subset})
        model_mae = [sum(float(r["mae"]) for r in subset if int(r["window"]) == w) /
                     sum(1 for r in subset if int(r["window"]) == w) for w in windows]
        naive_mae = [sum(float(r["naive_mae"]) for r in subset if int(r["window"]) == w) /
                     sum(1 for r in subset if int(r["window"]) == w) for w in windows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(windows, model_mae, "o-", label="Adaptive Kalman")
        ax.plot(windows, naive_mae, "s--", label="Persistence baseline")
        ax.set(xlabel="Innovation window (ticks)", ylabel="Mean MAE (agents)",
               title=f"Zombie forecasting accuracy: {horizon}-tick horizon")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / f"mae_h{horizon}.png", dpi=160)
        plt.close(fig)
    histories = sorted(folder.glob("population_seed_*.csv"))
    if histories:
        with histories[0].open(newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot([int(r["tick"]) for r in data],
                [int(r["zombies"]) for r in data], label="Zombies")
        ax.plot([int(r["tick"]) for r in data],
                [int(r["humans"]) for r in data], label="Humans")
        ax.set(xlabel="Simulation tick", ylabel="Agents", title=histories[0].stem)
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(folder / "population_trajectory.png", dpi=160)
        plt.close(fig)
    print(f"Saved figures to {folder}")


if __name__ == "__main__":
    main()
