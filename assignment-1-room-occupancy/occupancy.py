"""Compare two sensor configurations using the same perceptron settings."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from perceptron import Perceptron
from plot_results import plot_decision_regions, plot_training, plot_confusion, plot_timeline

HERE = Path(__file__).resolve().parent
FEATURES = {
    "Two sensors": ["S1_Temp", "S5_CO2"],
    "More sensors": ["S1_Temp", "S5_CO2", "S1_Light", "S1_Sound", "S6_PIR", "S7_PIR"],
}
SPLIT_DATE = "2018-01-01"


def load_data():
    df = pd.read_csv(HERE / "data" / "Occupancy_Estimation.csv")
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y/%m/%d %H:%M:%S")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["occupied"] = (df["Room_Occupancy_Count"] > 0).astype(int)
    columns = FEATURES["More sensors"] + ["Room_Occupancy_Count"]
    if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
        raise ValueError("Expected complete, finite sensor data.")
    train = df[df["timestamp"] < SPLIT_DATE].copy()
    test = df[df["timestamp"] >= SPLIT_DATE].copy()
    if train["occupied"].nunique() != 2 or test["occupied"].nunique() != 2:
        raise ValueError("Both time periods must contain empty and occupied observations.")
    return train, test


def scale_features(train, test, columns):
    X_train = train[columns].to_numpy(dtype=float)
    X_test = test[columns].to_numpy(dtype=float)
    # The test period has no role in fitting the mean or standard deviation.
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1.0
    return (X_train - mean) / scale, (X_test - mean) / scale, mean, scale


def evaluate(y, pred):
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "occupied_precision": precision,
        "occupied_recall": recall,
        "false_empty": fn,
        "false_occupied": fp,
        "true_empty": tn,
        "true_occupied": tp,
    }


def main():
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    train, test = load_data()
    y_train = train["occupied"].to_numpy()
    y_test = test["occupied"].to_numpy()
    print(f"Train: {len(train)} rows; test: {len(test)} rows")
    print(f"Occupied rows: train={y_train.sum()}, test={y_test.sum()}")

    models, matrices, predictions, parameters = {}, {}, {}, {}
    scores = [{"model": "Always empty", **evaluate(y_test, np.zeros_like(y_test))}]
    for name, columns in FEATURES.items():
        X_train, X_test, mean, scale = scale_features(train, test, columns)
        ppn = Perceptron(eta=0.01, n_iter=30, random_state=1)
        ppn.fit(X_train, y_train)
        pred = ppn.predict(X_test)
        score = evaluate(y_test, pred)
        scores.append({"model": name, **score})
        models[name] = ppn
        predictions[name] = pred
        matrices[name] = np.array([[score["true_empty"], score["false_occupied"]],
                                   [score["false_empty"], score["true_occupied"]]])
        parameters[name] = {
            "features": columns, "weights": ppn.w_.tolist(), "bias": float(ppn.b_),
            "train_mean": mean.tolist(), "train_std": scale.tolist(),
            "eta": ppn.eta, "n_iter": ppn.n_iter, "random_state": ppn.random_state,
            "final_training_mistakes": int(np.sum(ppn.predict(X_train) != y_train)),
        }
        print(f"\n{name}\nWeights: {ppn.w_}\nBias: {ppn.b_}")
        if name == "Two sensors":
            plot_decision_regions(test[columns].to_numpy(), y_test, ppn, mean, scale,
                                  results / "decision_regions.png")

    metrics = pd.DataFrame(scores)
    metrics.to_csv(results / "metrics.csv", index=False)
    print("\n" + metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    test_predictions = test[["timestamp", "Room_Occupancy_Count", "occupied"]].copy()
    for name, pred in predictions.items():
        test_predictions[name] = pred
    test_predictions.to_csv(results / "test_predictions.csv", index=False)
    pd.DataFrame({name: model.errors_ for name, model in models.items()},
                 index=pd.Index(range(1, 31), name="epoch")).to_csv(results / "training_updates.csv")
    (results / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    split = []
    for name, frame in [("train", train), ("test", test)]:
        split.append({"split": name, "rows": len(frame), "occupied": int(frame["occupied"].sum()),
                      "start": str(frame["timestamp"].min()), "end": str(frame["timestamp"].max())})
    (results / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    plot_training(models, results / "training_updates.png")
    plot_confusion(matrices, results / "confusion_matrices.png")
    plot_timeline(test, predictions, results / "test_timeline.png")


if __name__ == "__main__":
    main()
