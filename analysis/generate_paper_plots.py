import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "analysis/results/local_benchmark_summary.csv"
OUT_DIR = "LTC_CFC_ContinualLearning/figures"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_buffer_methods(df: pd.DataFrame, dataset: str, baseline: str, ours: str, out_name: str, title: str) -> None:
    methods = ["er200", "er500", "er1000", "derpp200", "derpp500", "derpp1000", "erace200", "erace500", "erace1000"]
    order = [m for m in methods if m in set(df[(df["dataset"] == dataset) & (df["backbone"] == baseline)]["model"]) or m in set(df[(df["dataset"] == dataset) & (df["backbone"] == ours)]["model"])]

    base = df[(df["dataset"] == dataset) & (df["backbone"] == baseline)][["model", "mean", "std"]].rename(columns={"mean": "base_mean", "std": "base_std"})
    cfc = df[(df["dataset"] == dataset) & (df["backbone"] == ours)][["model", "mean", "std"]].rename(columns={"mean": "cfc_mean", "std": "cfc_std"})

    merged = pd.merge(base, cfc, on="model", how="inner")
    merged = merged[merged["model"].isin(order)].copy()
    merged["model"] = pd.Categorical(merged["model"], categories=order, ordered=True)
    merged = merged.sort_values("model")

    if merged.empty:
        print(f"No overlapping models for {dataset}: {baseline} vs {ours}")
        return

    x = range(len(merged))
    width = 0.38

    plt.figure(figsize=(10.5, 4.8))
    plt.bar([i - width / 2 for i in x], merged["base_mean"], width=width, yerr=merged["base_std"], capsize=3, label=baseline.upper(), color="#4C78A8")
    plt.bar([i + width / 2 for i in x], merged["cfc_mean"], width=width, yerr=merged["cfc_std"], capsize=3, label=ours.upper(), color="#F58518")

    plt.xticks(list(x), [m.upper() for m in merged["model"].astype(str)], rotation=25, ha="right")
    plt.ylabel("Class-IL Accuracy (%)")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_delta(df: pd.DataFrame, dataset: str, baseline: str, ours: str, out_name: str, title: str) -> None:
    base = df[(df["dataset"] == dataset) & (df["backbone"] == baseline)][["model", "mean"]].rename(columns={"mean": "base_mean"})
    cfc = df[(df["dataset"] == dataset) & (df["backbone"] == ours)][["model", "mean"]].rename(columns={"mean": "cfc_mean"})
    merged = pd.merge(base, cfc, on="model", how="inner")
    if merged.empty:
        print(f"No overlap for delta plot {dataset}")
        return

    merged["delta"] = merged["cfc_mean"] - merged["base_mean"]
    merged = merged.sort_values("delta", ascending=False)

    colors = ["#2E8B57" if d >= 0 else "#C44E52" for d in merged["delta"]]

    plt.figure(figsize=(7.5, 4.8))
    plt.barh(merged["model"].str.upper(), merged["delta"], color=colors)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Delta Accuracy (CfC - Baseline, pp)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    ensure_dir(OUT_DIR)
    df = pd.read_csv(INPUT_CSV)

    plot_buffer_methods(
        df,
        dataset="mnist",
        baseline="mlp",
        ours="cfc",
        out_name="mnist_baseline_vs_cfc.png",
        title="Split-MNIST: MLP vs CfC by Method/Buffer",
    )

    plot_buffer_methods(
        df,
        dataset="cifar",
        baseline="resnet",
        ours="cfc",
        out_name="cifar_baseline_vs_cfc.png",
        title="Split-CIFAR-10: ResNet vs CfC by Method/Buffer",
    )

    plot_delta(
        df,
        dataset="mnist",
        baseline="mlp",
        ours="cfc",
        out_name="mnist_delta_cfc_minus_baseline.png",
        title="Split-MNIST Deltas: CfC - MLP",
    )

    plot_delta(
        df,
        dataset="cifar",
        baseline="resnet",
        ours="cfc",
        out_name="cifar_delta_cfc_minus_baseline.png",
        title="Split-CIFAR-10 Deltas: CfC - ResNet",
    )


if __name__ == "__main__":
    main()
