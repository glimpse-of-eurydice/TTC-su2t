from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

SUPPORTED_NUM_CLUSTERS = (100, 200, 500, 1000)
DEFAULT_NUM_CLUSTERS = 100
DEFAULT_MAX_DECODE_LEN = 600
NUM_CLUSTERS_ENV = "S2UT_NUM_CLUSTERS"


@dataclass(frozen=True)
class ExperimentConfig:
    num_clusters: int
    bos_token: int
    eos_token: int
    pad_token: int
    vocab_size: int
    units_json: str
    kmeans_model_path: str
    checkpoint_path: str
    predicted_units_path: str
    output_audio_path: str
    ablation_results_path: str
    ablation_plot_path: str
    verify_report_path: str


def validate_num_clusters(num_clusters: int) -> int:
    if num_clusters not in SUPPORTED_NUM_CLUSTERS:
        supported = ", ".join(str(value) for value in SUPPORTED_NUM_CLUSTERS)
        raise ValueError(
            f"Unsupported num_clusters={num_clusters}. Choose one of: {supported}."
        )
    return num_clusters


def get_default_num_clusters() -> int:
    raw_value = os.getenv(NUM_CLUSTERS_ENV, str(DEFAULT_NUM_CLUSTERS))
    try:
        num_clusters = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{NUM_CLUSTERS_ENV} must be an integer, got: {raw_value!r}."
        ) from exc
    return validate_num_clusters(num_clusters)


def _legacy_or_suffix_path(num_clusters: int, legacy_path: str, template: str) -> str:
    if num_clusters == DEFAULT_NUM_CLUSTERS:
        return legacy_path
    return template.format(k=num_clusters)


def build_experiment_config(num_clusters: int | None = None) -> ExperimentConfig:
    resolved_num_clusters = (
        get_default_num_clusters() if num_clusters is None else validate_num_clusters(num_clusters)
    )

    return ExperimentConfig(
        num_clusters=resolved_num_clusters,
        bos_token=resolved_num_clusters,
        eos_token=resolved_num_clusters + 1,
        pad_token=resolved_num_clusters + 2,
        vocab_size=resolved_num_clusters + 3,
        units_json=_legacy_or_suffix_path(
            resolved_num_clusters,
            "./data/TCST/target_units.json",
            "./data/TCST/target_units_k{k}.json",
        ),
        kmeans_model_path=f"./data/TCST/kmeans_{resolved_num_clusters}.pkl",
        checkpoint_path=_legacy_or_suffix_path(
            resolved_num_clusters,
            "./checkpoints/best_s2ut_model.pth",
            "./checkpoints/best_s2ut_model_k{k}.pth",
        ),
        predicted_units_path=_legacy_or_suffix_path(
            resolved_num_clusters,
            "./results/predicted_units.json",
            "./results/predicted_units_k{k}.json",
        ),
        output_audio_path=_legacy_or_suffix_path(
            resolved_num_clusters,
            "./final_translation.wav",
            "./final_translation_k{k}.wav",
        ),
        ablation_results_path=_legacy_or_suffix_path(
            resolved_num_clusters,
            "./results/ablation_results.csv",
            "./results/ablation_results_k{k}.csv",
        ),
        ablation_plot_path=_legacy_or_suffix_path(
            resolved_num_clusters,
            "./results/lm_weight_ablation.png",
            "./results/lm_weight_ablation_k{k}.png",
        ),
        verify_report_path=_legacy_or_suffix_path(
            resolved_num_clusters,
            "./results/verify_report.json",
            "./results/verify_report_k{k}.json",
        ),
    )


def add_num_clusters_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=get_default_num_clusters(),
        choices=SUPPORTED_NUM_CLUSTERS,
        help="Number of discrete units / K-means clusters.",
    )


def add_max_len_arg(
    parser: argparse.ArgumentParser,
    default: int = DEFAULT_MAX_DECODE_LEN,
) -> None:
    parser.add_argument(
        "--max-len",
        type=int,
        default=default,
        help="Maximum autoregressive decoding length.",
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
