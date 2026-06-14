"""Render the MAE comparison bar chart from the 28-dish Nutrition5k holdout eval.

Produces assets/mae_comparison.png. Update RESULTS below if you re-run
`python -m src.evaluate` and get new numbers.

Usage:
    python -m src.plot_results
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mean Absolute Error vs lab-measured truth, n=28 held-out dishes.
# nutrient -> {model: mae}
RESULTS = {
    "Calories (kcal)": {"base": 123.8, "food101": 142.4, "nutrition5k": 81.4},
    "Protein (g)":     {"base": 10.9,  "food101": 12.0,  "nutrition5k": 8.5},
    "Carbs (g)":       {"base": 14.5,  "food101": 17.9,  "nutrition5k": 8.0},
    "Fat (g)":         {"base": 6.0,   "food101": 9.3,   "nutrition5k": 5.3},
}

MODELS = ["base", "food101", "nutrition5k"]
LABELS = ["Base (no FT)", "Food101 LoRA", "Nutrition5k LoRA"]
COLORS = ["#9e9e9e", "#ef8a62", "#67a9cf"]

OUT = Path(__file__).resolve().parent.parent / "assets" / "mae_comparison.png"


def main():
    OUT.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Calorie/Macro Error vs Lab-Measured Truth\n"
                 "(Nutrition5k 28-dish holdout — lower is better)",
                 fontsize=13, fontweight="bold")

    for ax, (nutrient, vals) in zip(axes.flat, RESULTS.items()):
        heights = [vals[m] for m in MODELS]
        bars = ax.bar(LABELS, heights, color=COLORS, edgecolor="black", linewidth=0.6)
        ax.set_title(nutrient, fontsize=11)
        ax.set_ylabel("MAE")
        ax.margins(y=0.18)
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=9)
        # highlight the best (lowest) model
        best = min(range(len(heights)), key=lambda i: heights[i])
        bars[best].set_edgecolor("#1a7a1a")
        bars[best].set_linewidth(2.2)
        ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
