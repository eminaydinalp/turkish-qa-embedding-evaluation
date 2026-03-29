from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: Generate t-SNE plots for question and answer embeddings by model."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--dataset-keys", type=str, default="")
    parser.add_argument("--model-keys", type=str, default="")
    parser.add_argument(
        "--max-points-per-type",
        type=int,
        default=0,
        help="Optional cap for visualization speed. 0 means all points.",
    )
    return parser.parse_args()


def parse_key_filter(value: str) -> set[str] | None:
    if not value.strip():
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return set(keys) if keys else None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_items(items: list[dict[str, Any]], key_name: str, keys: set[str] | None) -> list[dict[str, Any]]:
    if keys is None:
        return items
    selected = [item for item in items if item[key_name] in keys]
    missing = sorted(keys.difference({item[key_name] for item in selected}))
    if missing:
        raise ValueError(f"Unknown {key_name}: {', '.join(missing)}")
    return selected


def filter_enabled_datasets(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dataset for dataset in datasets if dataset.get("enabled", True)]


def maybe_cap_embeddings(arr: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(arr) <= max_points:
        return arr
    rng = np.random.default_rng(42)
    idx = rng.choice(len(arr), size=max_points, replace=False)
    return arr[idx]


def build_plot_frame(q_2d: np.ndarray, a_2d: np.ndarray) -> pd.DataFrame:
    q_df = pd.DataFrame({"x": q_2d[:, 0], "y": q_2d[:, 1], "type": "question"})
    a_df = pd.DataFrame({"x": a_2d[:, 0], "y": a_2d[:, 1], "type": "answer"})
    return pd.concat([q_df, a_df], ignore_index=True)


def run_tsne(emb: np.ndarray, tsne_cfg: dict[str, Any]) -> np.ndarray:
    tsne = TSNE(
        n_components=int(tsne_cfg.get("n_components", 2)),
        perplexity=float(tsne_cfg.get("perplexity", 30)),
        learning_rate=tsne_cfg.get("learning_rate", "auto"),
        init=tsne_cfg.get("init", "pca"),
        random_state=int(tsne_cfg.get("random_state", 42)),
    )
    return tsne.fit_transform(emb)


def save_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 8))
    ax = sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="type",
        palette={"question": "#2563eb", "answer": "#dc2626"},
        alpha=0.65,
        s=16,
        linewidth=0,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(title="Type")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_filter = parse_key_filter(args.dataset_keys)
    model_filter = parse_key_filter(args.model_keys)

    datasets = filter_enabled_datasets(cfg["datasets"])
    datasets = filter_items(datasets, "key", dataset_filter)
    models = filter_items(cfg["models"], "key", model_filter)

    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    plot_dir = Path(cfg["paths"]["plots_dir"])
    tsne_cfg = cfg.get("tsne", {})

    for dataset in datasets:
        dataset_key = dataset["key"]
        for model in models:
            model_key = model["key"]

            q_path = emb_dir / dataset_key / f"{model_key}_questions.npy"
            a_path = emb_dir / dataset_key / f"{model_key}_answers.npy"
            if not q_path.exists() or not a_path.exists():
                raise FileNotFoundError(f"Missing embeddings for {dataset_key}/{model_key}")

            q = np.load(q_path, allow_pickle=False).astype(np.float32)
            a = np.load(a_path, allow_pickle=False).astype(np.float32)

            q = maybe_cap_embeddings(q, args.max_points_per_type)
            a = maybe_cap_embeddings(a, args.max_points_per_type)

            all_emb = np.vstack([q, a])
            all_2d = run_tsne(all_emb, tsne_cfg)

            q_2d = all_2d[: len(q)]
            a_2d = all_2d[len(q) :]
            frame = build_plot_frame(q_2d, a_2d)

            out_path = plot_dir / f"{dataset_key}_{model_key}_tsne.png"
            title = f"t-SNE - {dataset_key} - {model_key}"
            save_plot(frame, out_path, title)
            print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()