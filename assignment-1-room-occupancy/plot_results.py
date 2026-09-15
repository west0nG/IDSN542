"""Figures for the two sensor experiments."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

COLORS = ["#4279a3", "#d57249"]


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_decision_regions(X, y, classifier, mean, scale, path):
    # Plot raw units; apply the training scaling before asking for predictions.
    low = X.min(axis=0)
    high = X.max(axis=0)
    margin = np.maximum((high - low) * 0.05, 0.1)
    xx, yy = np.meshgrid(
        np.linspace(low[0] - margin[0], high[0] + margin[0], 250),
        np.linspace(low[1] - margin[1], high[1] + margin[1], 250),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    pred = classifier.predict((grid - mean) / scale).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5],
                cmap=ListedColormap(COLORS), alpha=0.18)
    for label, name, marker in [(0, "Empty (actual)", "o"), (1, "Occupied (actual)", "^")]:
        ax.scatter(X[y == label, 0], X[y == label, 1], s=13,
                   color=COLORS[label], marker=marker, alpha=0.55, label=name)
    ax.set(xlabel="S1 temperature (°C)", ylabel="CO₂ (ppm)",
           title="Two sensors: test observations and predicted regions")
    ax.legend()
    save_figure(fig, path)


def plot_training(models, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, model in models.items():
        ax.plot(range(1, len(model.errors_) + 1), model.errors_, label=name)
    ax.set(xlabel="Epoch", ylabel="Updates during the epoch",
           title="Perceptron training updates")
    ax.legend()
    save_figure(fig, path)


def plot_confusion(matrices, path):
    fig, axes = plt.subplots(1, len(matrices), figsize=(10, 4))
    for ax, (name, matrix) in zip(axes, matrices.items()):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black", fontsize=15)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Empty", "Occupied"],
               yticklabels=["Empty", "Occupied"], xlabel="Predicted", ylabel="Actual",
               title=name)
    save_figure(fig, path)


def plot_timeline(test, predictions, path):
    # Separate dates so that lines do not bridge the overnight gap.
    dates = test["timestamp"].dt.date
    days = dates.unique()
    fig, axes = plt.subplots(len(days), 1, figsize=(11, 3 * len(days)), squeeze=False)
    for ax, day in zip(axes[:, 0], days):
        selected = (dates == day).to_numpy()
        time = test.loc[selected, "timestamp"]
        actual = test.loc[selected, "occupied"].to_numpy()
        ax.step(time, actual, where="post", color="black", linewidth=2, label="Actual")
        for i, (name, pred) in enumerate(predictions.items()):
            ax.step(time, pred[selected] + 0.035 * (i + 1), where="post",
                    alpha=0.7, label=name, color=COLORS[i])
        ax.set(yticks=[0, 1], yticklabels=["Empty", "Occupied"], ylim=(-0.15, 1.2),
               title=str(day), xlabel="Recorded time")
        ax.legend(loc="upper right", ncol=3)
    fig.suptitle("Test days: actual occupancy and predictions (slightly offset)")
    save_figure(fig, path)
