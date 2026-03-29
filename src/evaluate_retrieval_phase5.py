from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: Evaluate angle-based QA retrieval (Q->A and A->Q)."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--dataset-keys", type=str, default="")
    parser.add_argument("--model-keys", type=str, default="")
    parser.add_argument("--chunk-size", type=int, default=256)
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


def load_embeddings(
    embeddings_dir: Path,
    dataset_key: str,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = embeddings_dir / dataset_key
    q = np.load(base / f"{model_key}_questions.npy", allow_pickle=False).astype(np.float32)
    a = np.load(base / f"{model_key}_answers.npy", allow_pickle=False).astype(np.float32)
    ids = np.load(base / f"{model_key}_pair_ids.npy", allow_pickle=True).astype(str)
    return q, a, ids


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def topk_indices_by_angle(query: np.ndarray, cand: np.ndarray, top_k: int) -> np.ndarray:
    cosine = query @ cand.T
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = np.arccos(cosine)

    k = min(top_k, angles.shape[1])
    part = np.argpartition(angles, kth=k - 1, axis=1)[:, :k]
    part_vals = np.take_along_axis(angles, part, axis=1)
    order = np.argsort(part_vals, axis=1)
    return np.take_along_axis(part, order, axis=1)


def compute_direction_metrics(
    queries: np.ndarray,
    candidates: np.ndarray,
    top_k: int,
    chunk_size: int,
) -> tuple[float, float]:
    n = queries.shape[0]
    top1_hits = 0
    topk_hits = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx = topk_indices_by_angle(queries[start:end], candidates, top_k)

        gt = np.arange(start, end, dtype=np.int64)[:, None]
        top1_hits += int((idx[:, :1] == gt).sum())
        topk_hits += int(np.any(idx == gt, axis=1).sum())

    return top1_hits / n, topk_hits / n


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_filter = parse_key_filter(args.dataset_keys)
    model_filter = parse_key_filter(args.model_keys)

    datasets = filter_enabled_datasets(cfg["datasets"])
    datasets = filter_items(datasets, "key", dataset_filter)
    models = filter_items(cfg["models"], "key", model_filter)

    top_k = int(cfg.get("retrieval", {}).get("top_k", 5))
    embeddings_dir = Path(cfg["paths"]["embeddings_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        dataset_key = dataset["key"]
        rows: list[dict[str, Any]] = []

        for model in models:
            model_key = model["key"]
            q_emb, a_emb, pair_ids = load_embeddings(embeddings_dir, dataset_key, model_key)

            if q_emb.shape[0] != a_emb.shape[0] or q_emb.shape[0] != len(pair_ids):
                raise ValueError(f"Embedding size mismatch for {dataset_key}/{model_key}")

            q_norm = l2_normalize(q_emb)
            a_norm = l2_normalize(a_emb)

            qa_top1, qa_topk = compute_direction_metrics(
                queries=q_norm,
                candidates=a_norm,
                top_k=top_k,
                chunk_size=args.chunk_size,
            )
            aq_top1, aq_topk = compute_direction_metrics(
                queries=a_norm,
                candidates=q_norm,
                top_k=top_k,
                chunk_size=args.chunk_size,
            )

            metrics = {
                "dataset_key": dataset_key,
                "model_key": model_key,
                "num_pairs": int(len(pair_ids)),
                "similarity": "angle",
                "top_k": top_k,
                "q_to_a_top1": qa_top1,
                f"q_to_a_top{top_k}": qa_topk,
                "a_to_q_top1": aq_top1,
                f"a_to_q_top{top_k}": aq_topk,
            }
            rows.append(metrics)

            model_metrics_path = metrics_dir / f"{dataset_key}_{model_key}_retrieval.json"
            model_metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved metrics: {model_metrics_path}")

        summary_df = pd.DataFrame(rows)
        summary_csv = metrics_dir / f"{dataset_key}_retrieval_summary.csv"
        summary_json = metrics_dir / f"{dataset_key}_retrieval_summary.json"
        summary_df.to_csv(summary_csv, index=False)
        summary_json.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved summary: {summary_csv}")
        print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()