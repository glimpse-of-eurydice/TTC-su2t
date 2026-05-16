# 12_verify_case.py
import argparse
import csv
import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from s2ut_config import add_num_clusters_arg, build_experiment_config

TRAIN_CSV = "./data/train.csv"
DEV_CSV = "./data/dev.csv"
TEST_CSV = "./data/test.csv"
DEFAULT_TEST_AUDIO = "./TCST/wav/Amdo/maqufa/maqufa_002.wav"
DEFAULT_UNITS_KEY = "test_audio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether predicted units retrieve text consistent with the query sample."
    )
    add_num_clusters_arg(parser)
    parser.add_argument("--predicted-units", default=None)
    parser.add_argument("--units-key", default=DEFAULT_UNITS_KEY)
    parser.add_argument("--units-json", default=None)
    parser.add_argument("--test-audio", default=DEFAULT_TEST_AUDIO)
    parser.add_argument("--sample-id", default="")
    parser.add_argument(
        "--knn-pool",
        default="train",
        choices=["train", "dev", "test", "all"],
        help="Candidate pool for KNN retrieval. Use train to mimic the diagnostic decoder.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for Mandarin and Tibetan text consistency.",
    )
    parser.add_argument("--save-report", default=None)
    return parser.parse_args()


def norm_path(path_text: str) -> str:
    return str(Path(path_text).expanduser().resolve(strict=False))


def load_csv_rows(csv_path: str, split: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["split"] = split
            rows.append(row)
    return rows


def load_split_rows(pool: str) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "test": TEST_CSV,
    }
    split_order = ["train", "dev", "test"] if pool == "all" else [pool]

    all_rows: List[Dict[str, str]] = []
    id_to_row: Dict[str, Dict[str, str]] = {}
    audio_to_row: Dict[str, Dict[str, str]] = {}

    for split in split_order:
        rows = load_csv_rows(split_files[split], split)
        all_rows.extend(rows)
        for row in rows:
            sid = row.get("sample_id", "")
            if sid and sid not in id_to_row:
                id_to_row[sid] = row
            audio_path = row.get("tibetan_audio", "")
            if audio_path:
                audio_to_row[norm_path(audio_path)] = row
    return all_rows, id_to_row, audio_to_row


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、；：,.!?;:\"“”'‘’（）()【】《》<>—…\-།༑༔]", "", text)
    return text


def text_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def extract_units(value: object) -> Optional[List[int]]:
    if isinstance(value, list):
        return [int(x) for x in value]
    return None


def load_predicted_units(path: str, units_key: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return extract_units(data) or []

    if isinstance(data, dict):
        if units_key in data:
            units = extract_units(data[units_key])
            if units is not None:
                return units
        for value in data.values():
            units = extract_units(value)
            if units is not None:
                return units
    raise ValueError(f"Could not find a usable unit sequence in {path}.")


def levenshtein(seq1: Sequence[int], seq2: Sequence[int]) -> int:
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1
    if not seq2:
        return len(seq1)
    previous = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1, 1):
        current = [i]
        for j, b in enumerate(seq2, 1):
            ins = previous[j] + 1
            dele = current[j - 1] + 1
            sub = previous[j - 1] + (a != b)
            current.append(min(ins, dele, sub))
        previous = current
    return previous[-1]


def parse_sample_id_from_audio(audio_path: str) -> str:
    stem = Path(audio_path).stem
    m = re.match(r"^([a-z]+)_(\d+)$", stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return stem


def resolve_query_row(
    sample_id: str,
    test_audio: str,
    id_to_row: Dict[str, Dict[str, str]],
    audio_to_row: Dict[str, Dict[str, str]],
) -> Tuple[str, Optional[Dict[str, str]]]:
    if sample_id:
        return sample_id, id_to_row.get(sample_id)

    audio_key = norm_path(test_audio)
    if audio_key in audio_to_row:
        row = audio_to_row[audio_key]
        return row["sample_id"], row

    inferred = parse_sample_id_from_audio(test_audio)
    return inferred, id_to_row.get(inferred)


def find_best_knn_match(
    predicted_units: List[int],
    candidate_rows: List[Dict[str, str]],
    units_dict: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, str], int]:
    best_row: Optional[Dict[str, str]] = None
    best_distance = 10**9
    for row in candidate_rows:
        sid = row.get("sample_id", "")
        if sid not in units_dict:
            continue
        ref_units = units_dict[sid].get("units", [])
        if not isinstance(ref_units, list):
            continue
        dist = levenshtein(predicted_units, ref_units)
        if dist < best_distance:
            best_distance = dist
            best_row = row
    if best_row is None:
        raise RuntimeError("The candidate pool has no usable KNN samples; target units may be missing.")
    return best_row, best_distance


def print_report(report: Dict[str, object]) -> None:
    print("\n" + "=" * 60)
    print("K-means unit alignment diagnostic")
    print("=" * 60)
    print(f"Query sample ID: {report['query_sample_id']}")
    print(f"Query split: {report['query_split']}")
    print(f"Query Tibetan text: {report['query_tibetan']}")
    print(f"Reference Mandarin text: {report['query_chinese']}")
    print("-" * 60)
    print(f"Retrieved sample ID: {report['retrieved_sample_id']}")
    print(f"Retrieved split: {report['retrieved_split']}")
    print(f"Unit edit distance: {report['unit_edit_distance']}")
    print(f"Retrieved Tibetan text: {report['retrieved_tibetan']}")
    print(f"Retrieved Mandarin text: {report['retrieved_chinese']}")
    print("-" * 60)
    print(
        "Mandarin consistency: "
        f"{report['chinese_consistent']} "
        f"(similarity={report['chinese_similarity']:.4f})"
    )
    print(
        "Tibetan consistency: "
        f"{report['tibetan_consistent']} "
        f"(similarity={report['tibetan_similarity']:.4f})"
    )
    print(f"Overall consistent: {report['overall_consistent']}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    config = build_experiment_config(args.num_clusters)
    predicted_units_path = args.predicted_units or config.predicted_units_path
    units_json = args.units_json or config.units_json
    save_report_path = args.save_report or config.verify_report_path

    predicted_units = load_predicted_units(predicted_units_path, args.units_key)
    if not predicted_units:
        raise RuntimeError("Predicted units are empty; cannot run the diagnostic.")

    with open(units_json, "r", encoding="utf-8") as f:
        units_dict = json.load(f)

    candidate_rows, _, _ = load_split_rows(args.knn_pool)
    all_rows, id_to_row_all, audio_to_row_all = load_split_rows("all")
    _ = all_rows

    query_sample_id, query_row = resolve_query_row(
        sample_id=args.sample_id.strip(),
        test_audio=args.test_audio,
        id_to_row=id_to_row_all,
        audio_to_row=audio_to_row_all,
    )

    retrieved_row, unit_edit_distance = find_best_knn_match(
        predicted_units=predicted_units,
        candidate_rows=candidate_rows,
        units_dict=units_dict,
    )

    query_tibetan = query_row.get("tibetan_text", "") if query_row else ""
    query_chinese = query_row.get("chinese_text", "") if query_row else ""
    query_split = query_row.get("split", "unknown") if query_row else "unknown"

    retrieved_tibetan = retrieved_row.get("tibetan_text", "")
    retrieved_chinese = retrieved_row.get("chinese_text", "")

    chinese_similarity = text_similarity(query_chinese, retrieved_chinese)
    tibetan_similarity = text_similarity(query_tibetan, retrieved_tibetan)
    chinese_consistent = chinese_similarity >= args.sim_threshold
    tibetan_consistent = tibetan_similarity >= args.sim_threshold
    overall_consistent = chinese_consistent and tibetan_consistent

    report: Dict[str, object] = {
        "num_clusters": config.num_clusters,
        "predicted_units_path": predicted_units_path,
        "predicted_units_key": args.units_key,
        "predicted_units_length": len(predicted_units),
        "units_json_path": units_json,
        "knn_pool": args.knn_pool,
        "query_sample_id": query_sample_id,
        "query_split": query_split,
        "query_audio": norm_path(args.test_audio),
        "query_tibetan": query_tibetan,
        "query_chinese": query_chinese,
        "retrieved_sample_id": retrieved_row.get("sample_id", ""),
        "retrieved_split": retrieved_row.get("split", ""),
        "retrieved_tibetan": retrieved_tibetan,
        "retrieved_chinese": retrieved_chinese,
        "unit_edit_distance": unit_edit_distance,
        "sim_threshold": args.sim_threshold,
        "chinese_similarity": chinese_similarity,
        "tibetan_similarity": tibetan_similarity,
        "chinese_consistent": chinese_consistent,
        "tibetan_consistent": tibetan_consistent,
        "overall_consistent": overall_consistent,
    }

    if query_row is None:
        report["warning"] = (
            "The query audio was not found in train/dev/test, so reference "
            "Tibetan and Mandarin text could not be loaded."
        )

    print_report(report)

    save_path = Path(save_report_path)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report: {save_path.resolve(strict=False)}")


if __name__ == "__main__":
    main()
