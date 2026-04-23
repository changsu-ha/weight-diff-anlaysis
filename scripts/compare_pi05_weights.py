#!/usr/bin/env python3
"""CLI scaffold for comparing OpenPI pi0.5 checkpoint weights."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TOP_K = 50
DEFAULT_HISTOGRAM_SAMPLE_SIZE = 1_000_000
DEFAULT_SCATTER_SAMPLE_SIZE = 100_000
DEFAULT_CHUNK_SIZE = 10_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare OpenPI pi0.5 checkpoints and write analysis artifacts.",
    )
    parser.add_argument("--a", required=True, help="First checkpoint path (file or directory).")
    parser.add_argument("--b", required=True, help="Second checkpoint path (file or directory).")
    parser.add_argument("--out", required=True, help="Output directory for analysis artifacts.")

    parser.add_argument(
        "--include-other",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include unclassified parameters in aggregate plots (default: True).",
    )
    parser.add_argument(
        "--exclude-buffers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Exclude obvious non-trainable buffers and non-floating tensors (default: True).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of most changed parameters to report (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--histogram-sample-size",
        type=int,
        default=DEFAULT_HISTOGRAM_SAMPLE_SIZE,
        help=(
            "Maximum number of elementwise diffs to sample for histograms "
            f"(default: {DEFAULT_HISTOGRAM_SAMPLE_SIZE})."
        ),
    )
    parser.add_argument(
        "--scatter-sample-size",
        type=int,
        default=DEFAULT_SCATTER_SAMPLE_SIZE,
        help=(
            "Maximum number of weight pairs to sample for scatter plots "
            f"(default: {DEFAULT_SCATTER_SAMPLE_SIZE})."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum number of elements to process at once (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--fail-on-shape-mismatch",
        action="store_true",
        default=False,
        help="Fail when common keys have shape mismatches (default: False).",
    )
    parser.add_argument(
        "--component-map-json",
        type=str,
        default=None,
        help="Optional JSON file to override key-to-component mapping.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print additional key classification and mismatch information (default: False).",
    )
    return parser


def validate_input_path(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    return path


def compare_checkpoints(path_a: Path, path_b: Path, out_dir: Path, args: argparse.Namespace) -> None:
    """Run checkpoint comparison and write analysis artifacts.

    This is intentionally scaffold-only in this initial task step.
    """
    raise NotImplementedError(
        "compare_checkpoints() is not implemented yet. "
        "This scaffold only sets up CLI arguments and runtime structure."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    path_a = validate_input_path(args.a, "--a")
    path_b = validate_input_path(args.b, "--b")

    if args.component_map_json:
        validate_input_path(args.component_map_json, "--component-map-json")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_checkpoints(path_a=path_a, path_b=path_b, out_dir=out_dir, args=args)


if __name__ == "__main__":
    main()
