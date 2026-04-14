import argparse
import json
import re
from pathlib import Path

import pandas as pd

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


CODE_PATTERN = re.compile(
    r"(```|`[^`]+`|\bdef\b|\bclass\b|\bimport\b|#include|SELECT\s+.+\s+FROM)",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare model prediction files on one shared evaluation set."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="CSV files containing at least: prompt, reference, prediction.",
    )
    parser.add_argument(
        "--output-csv",
        default="model_comparison_summary.csv",
        help="Where to save the aggregate comparison table.",
    )
    parser.add_argument(
        "--output-json",
        default="model_comparison_details.json",
        help="Where to save bucketed metrics and config details.",
    )
    return parser.parse_args()


def normalize_columns(df):
    rename_map = {}
    for column in df.columns:
        lower = column.strip().lower()
        if lower in {"answer", "target", "gold", "label"}:
            rename_map[column] = "reference"
        elif lower in {"output", "predicted", "predicted_prompt", "generated_text", "response"}:
            rename_map[column] = "prediction"
        elif lower in {"input"}:
            rename_map[column] = "prompt"

    df = df.rename(columns=rename_map)
    required = {"prompt", "reference", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df[list(required)].copy()


def token_set(text):
    return re.findall(r"\w+", str(text).lower())


def token_f1(reference, prediction):
    ref_tokens = token_set(reference)
    pred_tokens = token_set(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = {}
    pred_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    overlap = 0
    for token, count in ref_counts.items():
        overlap += min(count, pred_counts.get(token, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(reference, prediction):
    return float(str(reference).strip() == str(prediction).strip())


def has_code(text):
    return bool(CODE_PATTERN.search(str(text)))


def prompt_bucket(prompt):
    tokens = token_set(prompt)
    length = len(tokens)

    if has_code(prompt):
        return "code_prompt"
    if length <= 8:
        return "short_prompt"
    if length >= 25:
        return "long_prompt"
    return "medium_prompt"


def build_scorer():
    if rouge_scorer is None:
        return None
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_row(reference, prediction, scorer):
    metrics = {
        "exact_match": exact_match(reference, prediction),
        "token_f1": token_f1(reference, prediction),
        "prediction_tokens": len(token_set(prediction)),
        "reference_tokens": len(token_set(reference)),
        "length_ratio": (
            len(token_set(prediction)) / len(token_set(reference))
            if len(token_set(reference)) > 0
            else 0.0
        ),
        "prediction_has_code": float(has_code(prediction)),
        "reference_has_code": float(has_code(reference)),
    }

    if scorer is not None:
        rouge_scores = scorer.score(str(reference), str(prediction))
        metrics["rouge1"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL"] = rouge_scores["rougeL"].fmeasure

    return metrics


def aggregate_metrics(df, metric_columns):
    summary = df[metric_columns].mean().to_dict()
    return {key: round(float(value), 4) for key, value in summary.items()}


def compare_file(path, scorer):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}. "
            "Export predictions from each notebook first, then pass those CSV files here."
        )

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    df = df.dropna(subset=["prompt", "reference", "prediction"]).reset_index(drop=True)
    df["bucket"] = df["prompt"].apply(prompt_bucket)

    metric_rows = []
    for row in df.itertuples(index=False):
        metrics = score_row(row.reference, row.prediction, scorer)
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    combined = pd.concat([df, metrics_df], axis=1)

    metric_columns = list(metrics_df.columns)
    overall = aggregate_metrics(combined, metric_columns)

    buckets = {}
    for bucket_name, bucket_df in combined.groupby("bucket"):
        buckets[bucket_name] = {
            "rows": int(len(bucket_df)),
            "metrics": aggregate_metrics(bucket_df, metric_columns),
        }

    return {
        "model_name": Path(path).stem,
        "rows": int(len(combined)),
        "metrics": overall,
        "buckets": buckets,
    }


def main():
    args = parse_args()
    scorer = build_scorer()

    reports = []
    for path in args.predictions:
        report = compare_file(path, scorer)
        reports.append(report)

    if not reports:
        raise ValueError("No prediction reports were generated.")

    summary_rows = []
    for report in reports:
        row = {"model_name": report["model_name"], "rows": report["rows"]}
        row.update(report["metrics"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="rougeL" if "rougeL" in summary_rows[0] else "token_f1",
        ascending=False,
    )
    summary_df.to_csv(args.output_csv, index=False)

    details = {
        "used_rouge_score": scorer is not None,
        "reports": reports,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)

    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {args.output_csv}")
    print(f"Saved detailed report to {args.output_json}")


if __name__ == "__main__":
    main()
