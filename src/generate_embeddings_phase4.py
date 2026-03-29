from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: Generate question and answer embeddings for all configured datasets and models."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--dataset-keys",
        type=str,
        default="",
        help="Comma-separated dataset keys. Empty means all datasets.",
    )
    parser.add_argument(
        "--model-keys",
        type=str,
        default="",
        help="Comma-separated model keys. Empty means all models.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use first N rows per dataset for quick tests. 0 means full dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def pick_device(device_value: str) -> str:
    if device_value != "auto":
        return device_value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_key_filter(value: str) -> set[str] | None:
    if not value.strip():
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return set(keys) if keys else None


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_items(items: list[dict[str, Any]], key_name: str, keys: set[str] | None) -> list[dict[str, Any]]:
    if keys is None:
        return items
    filtered = [item for item in items if item[key_name] in keys]
    missing = sorted(keys.difference({item[key_name] for item in filtered}))
    if missing:
        raise ValueError(f"Unknown {key_name} values: {', '.join(missing)}")
    return filtered


def filter_enabled_datasets(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dataset for dataset in datasets if dataset.get("enabled", True)]


def load_dataset_frame(dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    processed_file = Path(dataset_cfg["processed_file"])
    if processed_file.suffix == ".parquet":
        df = pd.read_parquet(processed_file)
    elif processed_file.suffix == ".jsonl":
        df = pd.read_json(processed_file, lines=True)
    else:
        parquet_candidate = processed_file.with_suffix(".parquet")
        jsonl_candidate = processed_file.with_suffix(".jsonl")
        if parquet_candidate.exists():
            df = pd.read_parquet(parquet_candidate)
        elif jsonl_candidate.exists():
            df = pd.read_json(jsonl_candidate, lines=True)
        else:
            raise FileNotFoundError(
                f"Could not find processed dataset file for {dataset_cfg['key']}: {processed_file}"
            )

    required_cols = {"id", "question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset {dataset_cfg['key']} is missing required columns. "
            f"Expected {required_cols}, got {set(df.columns)}"
        )
    return df[["id", "question", "answer"]].copy()


def format_texts(texts: pd.Series, prefix: str) -> list[str]:
    if not prefix:
        return texts.astype(str).tolist()
    return (prefix + texts.astype(str)).tolist()


def ensure_output_dirs(base_dir: Path, dataset_key: str) -> Path:
    out_dir = base_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def model_output_paths(out_dir: Path, model_key: str) -> dict[str, Path]:
    return {
        "questions": out_dir / f"{model_key}_questions.npy",
        "answers": out_dir / f"{model_key}_answers.npy",
        "ids": out_dir / f"{model_key}_pair_ids.npy",
        "meta": out_dir / f"{model_key}_meta.json",
    }


def outputs_are_usable(paths: dict[str, Path], expected_pairs: int) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False

    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if int(meta.get("num_pairs", -1)) != expected_pairs:
            return False

        ids = np.load(paths["ids"], allow_pickle=True)
        q_emb = np.load(paths["questions"], allow_pickle=False)
        a_emb = np.load(paths["answers"], allow_pickle=False)
        if len(ids) != expected_pairs:
            return False
        if q_emb.shape[0] != expected_pairs or a_emb.shape[0] != expected_pairs:
            return False
    except Exception:
        return False

    return True


def save_embeddings(
    model: SentenceTransformer,
    questions: list[str],
    answers: list[str],
    pair_ids: np.ndarray,
    normalize_embeddings: bool,
    batch_size: int,
    paths: dict[str, Path],
    meta_payload: dict[str, Any],
) -> None:
    q_emb = model.encode(
        questions,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    ).astype(np.float32)

    a_emb = model.encode(
        answers,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(paths["questions"], q_emb)
    np.save(paths["answers"], a_emb)
    np.save(paths["ids"], pair_ids)

    meta_payload["embedding_dim"] = int(q_emb.shape[1])
    paths["meta"].write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_filter = parse_key_filter(args.dataset_keys)
    model_filter = parse_key_filter(args.model_keys)

    datasets_cfg = filter_enabled_datasets(cfg["datasets"])
    datasets_cfg = filter_items(datasets_cfg, "key", dataset_filter)
    models_cfg = filter_items(cfg["models"], "key", model_filter)

    device = pick_device(cfg.get("device", "auto"))
    normalize_embeddings = bool(cfg.get("normalize_embeddings", True))
    batch_size = int(args.batch_size or cfg.get("batch_size", 32))
    embeddings_base = Path(cfg["paths"]["embeddings_dir"])
    embeddings_base.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Normalize embeddings: {normalize_embeddings}")

    for dataset_cfg in datasets_cfg:
        df = load_dataset_frame(dataset_cfg)
        if args.max_samples and args.max_samples > 0:
            df = df.head(args.max_samples).copy()
        if len(df) < int(dataset_cfg.get("min_pairs_required", 1000)) and not args.max_samples:
            raise ValueError(
                f"Dataset {dataset_cfg['key']} has {len(df)} pairs, below min_pairs_required="
                f"{dataset_cfg.get('min_pairs_required')}"
            )

        out_dir = ensure_output_dirs(embeddings_base, dataset_cfg["key"])
        print(f"\nDataset: {dataset_cfg['key']} | pairs: {len(df)}")

        for model_cfg in models_cfg:
            model_key = model_cfg["key"]
            hf_id = model_cfg["hf_id"]
            trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
            model_device = str(model_cfg.get("force_device", device))
            paths = model_output_paths(out_dir, model_key)

            if outputs_are_usable(paths, expected_pairs=len(df)) and not args.force:
                print(f"  - Skipping {model_key} (already exists)")
                continue

            q_prefix = model_cfg.get("query_prefix", "") if model_cfg.get("use_task_prefix") else ""
            a_prefix = model_cfg.get("passage_prefix", "") if model_cfg.get("use_task_prefix") else ""

            questions = format_texts(df["question"], q_prefix)
            answers = format_texts(df["answer"], a_prefix)
            pair_ids = np.asarray(df["id"].astype(str).tolist(), dtype=np.str_)

            print(f"  - Encoding {model_key} ({hf_id})")
            print(f"    question_prefix={q_prefix!r} | answer_prefix={a_prefix!r}")
            print(f"    model_device={model_device}")

            if args.dry_run:
                print("    dry-run active, files are not written")
                continue

            model = SentenceTransformer(
                hf_id,
                device=model_device,
                trust_remote_code=trust_remote_code,
            )
            meta_payload = {
                "dataset_key": dataset_cfg["key"],
                "model_key": model_key,
                "hf_id": hf_id,
                "trust_remote_code": trust_remote_code,
                "model_device": model_device,
                "num_pairs": int(len(df)),
                "normalize_embeddings": normalize_embeddings,
                "batch_size": batch_size,
                "question_prefix": q_prefix,
                "answer_prefix": a_prefix,
            }

            save_embeddings(
                model=model,
                questions=questions,
                answers=answers,
                pair_ids=pair_ids,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size,
                paths=paths,
                meta_payload=meta_payload,
            )
            print("    saved")


if __name__ == "__main__":
    main()