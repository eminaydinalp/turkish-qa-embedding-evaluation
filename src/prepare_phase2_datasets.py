from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    question_field: str
    answer_field: str


DATASETS = [
    DatasetSpec(
        key="patient_doctor_qa_tr",
        repo_id="kayrab/patient-doctor-qa-tr-5695",
        question_field="question",
        answer_field="answer",
    ),
    DatasetSpec(
        key="tquad",
        repo_id="dilanbakr/tquad",
        question_field="questions",
        answer_field="answers",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess selected Turkish QA datasets for Phase 2."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-question-chars", type=int, default=10)
    parser.add_argument("--min-answer-chars", type=int, default=3)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u0000", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split()).strip()


def extract_answer_text(answer_value: Any) -> str:
    if isinstance(answer_value, dict):
        text_value = answer_value.get("text", "")
        if isinstance(text_value, list):
            return " ".join(clean_text(item) for item in text_value if clean_text(item))
        return clean_text(text_value)
    if isinstance(answer_value, list):
        return " ".join(clean_text(item) for item in answer_value if clean_text(item))
    return clean_text(answer_value)


def get_dataset_license(repo_id: str) -> str | None:
    try:
        info = HfApi().dataset_info(repo_id)
        card_data = info.card_data or {}
        license_name = card_data.get("license")
        if isinstance(license_name, list):
            return ", ".join(str(item) for item in license_name)
        return str(license_name) if license_name else None
    except Exception:
        return None


def load_and_standardize(spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, int]]:
    dataset_dict = load_dataset(spec.repo_id)
    split_sizes = {split_name: len(split_data) for split_name, split_data in dataset_dict.items()}

    frames: list[pd.DataFrame] = []
    for split_name, split_data in dataset_dict.items():
        split_df = split_data.to_pandas()
        split_df = split_df[[spec.question_field, spec.answer_field]].copy()
        split_df["source_split"] = split_name
        frames.append(split_df)

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={spec.question_field: "question", spec.answer_field: "answer"})
    if spec.key == "tquad":
        df["answer"] = df["answer"].apply(extract_answer_text)
    else:
        df["answer"] = df["answer"].apply(clean_text)
    df["question"] = df["question"].apply(clean_text)
    df["source_dataset"] = spec.key
    df.insert(0, "id", [f"{spec.key}_{idx:06d}" for idx in range(len(df))])

    return df[["id", "question", "answer", "source_dataset", "source_split"]], split_sizes


def preprocess(
    df: pd.DataFrame,
    min_question_chars: int,
    min_answer_chars: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "rows_before": int(len(df)),
        "removed_empty": 0,
        "removed_short": 0,
        "removed_duplicates": 0,
    }

    not_empty_mask = (df["question"] != "") & (df["answer"] != "")
    stats["removed_empty"] = int((~not_empty_mask).sum())
    df = df[not_empty_mask].copy()

    long_enough_mask = (
        df["question"].str.len() >= min_question_chars
    ) & (df["answer"].str.len() >= min_answer_chars)
    stats["removed_short"] = int((~long_enough_mask).sum())
    df = df[long_enough_mask].copy()

    before_dupes = len(df)
    df = df.drop_duplicates(subset=["question", "answer"]).copy()
    stats["removed_duplicates"] = int(before_dupes - len(df))

    df = df.reset_index(drop=True)
    df["id"] = [f"{df.loc[idx, 'source_dataset']}_{idx:06d}" for idx in range(len(df))]

    stats["rows_after"] = int(len(df))
    return df, stats


def summarize(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "num_pairs": int(len(df)),
        "question_avg_chars": float(df["question"].str.len().mean()),
        "answer_avg_chars": float(df["answer"].str.len().mean()),
        "question_median_chars": float(df["question"].str.len().median()),
        "answer_median_chars": float(df["answer"].str.len().median()),
    }


def save_outputs(
    spec: DatasetSpec,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    split_sizes: dict[str, int],
    preprocess_stats: dict[str, int],
    raw_dir: Path,
    processed_dir: Path,
) -> None:
    raw_path = raw_dir / f"{spec.key}_raw.jsonl"
    processed_path = processed_dir / f"{spec.key}_qa.parquet"
    processed_jsonl_path = processed_dir / f"{spec.key}_qa.jsonl"
    stats_path = processed_dir / f"{spec.key}_stats.json"

    raw_df.to_json(raw_path, orient="records", lines=True, force_ascii=False)
    clean_df[["id", "question", "answer"]].to_parquet(processed_path, index=False)
    clean_df[["id", "question", "answer"]].to_json(
        processed_jsonl_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    stats_payload = {
        "dataset_key": spec.key,
        "repo_id": spec.repo_id,
        "license": get_dataset_license(spec.repo_id),
        "source_splits": split_sizes,
        "preprocess": preprocess_stats,
        "summary": summarize(clean_df),
    }
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    for spec in DATASETS:
        raw_df, split_sizes = load_and_standardize(spec)
        clean_df, preprocess_stats = preprocess(
            raw_df,
            min_question_chars=args.min_question_chars,
            min_answer_chars=args.min_answer_chars,
        )
        save_outputs(
            spec=spec,
            raw_df=raw_df,
            clean_df=clean_df,
            split_sizes=split_sizes,
            preprocess_stats=preprocess_stats,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
        )
        print(
            f"Prepared {spec.key}: {preprocess_stats['rows_before']} -> "
            f"{preprocess_stats['rows_after']} pairs"
        )


if __name__ == "__main__":
    main()