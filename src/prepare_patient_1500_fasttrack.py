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
        description="Create a subset dataset and reuse already computed model embeddings when available."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--source-key", type=str, default="patient_doctor_qa_tr")
    parser.add_argument("--target-key", type=str, default="patient_doctor_qa_tr_1500")
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_cfg_by_key(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    for ds in cfg["datasets"]:
        if ds["key"] == key:
            return ds
    raise KeyError(f"Dataset key not found in config: {key}")


def output_paths_for_model(base_dir: Path, dataset_key: str, model_key: str) -> dict[str, Path]:
    ds_dir = base_dir / dataset_key
    ds_dir.mkdir(parents=True, exist_ok=True)
    return {
        "questions": ds_dir / f"{model_key}_questions.npy",
        "answers": ds_dir / f"{model_key}_answers.npy",
        "ids": ds_dir / f"{model_key}_pair_ids.npy",
        "meta": ds_dir / f"{model_key}_meta.json",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    source_cfg = dataset_cfg_by_key(cfg, args.source_key)
    target_cfg = dataset_cfg_by_key(cfg, args.target_key)

    full_data_path = Path(source_cfg["processed_file"])
    target_data_path = Path(target_cfg["processed_file"])
    target_jsonl_path = target_data_path.with_suffix(".jsonl")

    full_df = pd.read_parquet(full_data_path)
    full_df = full_df[["id", "question", "answer"]].copy()
    sampled_original = full_df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    subset_df = sampled_original.copy()
    subset_df["id"] = [f"{args.target_key}_{idx:05d}" for idx in range(len(subset_df))]

    target_data_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(target_data_path, index=False)
    subset_df.to_json(target_jsonl_path, orient="records", lines=True, force_ascii=False)

    embeddings_base = Path(cfg["paths"]["embeddings_dir"])
    source_ids = sampled_original["id"].astype(str).tolist()
    target_ids = np.asarray(subset_df["id"].astype(str).tolist(), dtype=np.str_)

    transferred_models: list[str] = []
    skipped_models: list[str] = []

    for model in cfg["models"]:
        model_key = model["key"]
        src_paths = output_paths_for_model(embeddings_base, args.source_key, model_key)
        dst_paths = output_paths_for_model(embeddings_base, args.target_key, model_key)

        src_ready = all(path.exists() for path in src_paths.values())
        if not src_ready:
            skipped_models.append(model_key)
            continue

        if all(path.exists() for path in dst_paths.values()) and not args.force:
            skipped_models.append(model_key)
            continue

        src_pair_ids = np.load(src_paths["ids"], allow_pickle=True).astype(str)
        src_q = np.load(src_paths["questions"], allow_pickle=False)
        src_a = np.load(src_paths["answers"], allow_pickle=False)

        id_to_idx = {pair_id: idx for idx, pair_id in enumerate(src_pair_ids.tolist())}
        if any(pair_id not in id_to_idx for pair_id in source_ids):
            skipped_models.append(model_key)
            continue

        select_idx = np.asarray([id_to_idx[pair_id] for pair_id in source_ids], dtype=np.int64)
        sub_q = src_q[select_idx]
        sub_a = src_a[select_idx]

        np.save(dst_paths["questions"], sub_q)
        np.save(dst_paths["answers"], sub_a)
        np.save(dst_paths["ids"], target_ids)

        src_meta = json.loads(src_paths["meta"].read_text(encoding="utf-8"))
        src_meta["dataset_key"] = args.target_key
        src_meta["num_pairs"] = int(args.sample_size)
        src_meta["reused_from_dataset_key"] = args.source_key
        src_meta["reused_from_full_embeddings"] = True
        dst_paths["meta"].write_text(json.dumps(src_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        transferred_models.append(model_key)

    summary = {
        "source_dataset": args.source_key,
        "target_dataset": args.target_key,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "transferred_models": transferred_models,
        "skipped_models": skipped_models,
    }
    summary_path = target_data_path.with_name(f"{args.target_key}_reuse_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved subset: {target_data_path} ({len(subset_df)} rows)")
    print(f"Saved subset jsonl: {target_jsonl_path}")
    print(f"Transferred models: {', '.join(transferred_models) if transferred_models else '-'}")
    print(f"Skipped models: {', '.join(skipped_models) if skipped_models else '-'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()