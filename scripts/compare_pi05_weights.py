from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import re
import subprocess
from typing import TYPE_CHECKING, Dict, Iterator, Mapping

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised in environments without torch
        torch = None

SCRIPT_VERSION = "1.0.0"
PREFERRED_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
    "checkpoint.pt",
)
GLOB_PATTERNS = ("*.safetensors", "*.pt", "*.pth", "*.bin")
WRAPPER_KEYS = ("state_dict", "model_state_dict", "model", "module")
LOGGER = logging.getLogger("compare_pi05_weights")


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class KeyIndex:
    common_keys: set[str]
    only_in_a: set[str]
    only_in_b: set[str]
    shape_mismatch: set[str]
    comparable: set[str]


@dataclass
class RawStats:
    n: int = 0
    sum_a2: float = 0.0
    sum_b2: float = 0.0
    sum_diff2: float = 0.0
    dot: float = 0.0
    sum_abs_diff: float = 0.0
    max_abs_diff: float = 0.0
    sign_flip_count: int = 0
    nan_count_a: int = 0
    nan_count_b: int = 0
    nan_count_diff: int = 0
    inf_count_a: int = 0
    inf_count_b: int = 0
    inf_count_diff: int = 0

    def merge(self, other: "RawStats") -> None:
        self.n += other.n
        self.sum_a2 += other.sum_a2
        self.sum_b2 += other.sum_b2
        self.sum_diff2 += other.sum_diff2
        self.dot += other.dot
        self.sum_abs_diff += other.sum_abs_diff
        self.max_abs_diff = max(self.max_abs_diff, other.max_abs_diff)
        self.sign_flip_count += other.sign_flip_count
        self.nan_count_a += other.nan_count_a
        self.nan_count_b += other.nan_count_b
        self.nan_count_diff += other.nan_count_diff
        self.inf_count_a += other.inf_count_a
        self.inf_count_b += other.inf_count_b
        self.inf_count_diff += other.inf_count_diff


@dataclass(frozen=True)
class FinalizedStats:
    n: int
    norm_a: float
    norm_b: float
    diff_norm: float
    relative_diff: float
    cosine_similarity: float
    mean_abs_diff: float
    max_abs_diff: float
    sign_flip_ratio: float
    sign_flip_count: int
    nan_count_a: int
    nan_count_b: int
    nan_count_diff: int
    inf_count_a: int
    inf_count_b: int
    inf_count_diff: int


@dataclass(frozen=True)
class PerParameterRow:
    name: str
    component: str
    layer_id: str
    param_type: str
    shape: str
    num_params: int
    norm_a: float
    norm_b: float
    diff_norm: float
    relative_diff: float
    cosine_similarity: float
    mean_abs_diff: float
    max_abs_diff: float
    sign_flip_ratio: float


def resolve_weight_file(path: Path | str) -> Path:
    path = Path(path)
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Expected file or directory path: {path}")

    for name in PREFERRED_WEIGHT_FILENAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate

    for pattern in GLOB_PATTERNS:
        matches = sorted(p for p in path.glob(pattern) if p.is_file())
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not locate a checkpoint file. Tried preferred names and supported glob patterns."
    )


def _unwrap_checkpoint_dict(payload: object) -> Mapping[str, torch.Tensor]:
    current = payload
    visited = set()

    while isinstance(current, Mapping):
        marker = id(current)
        if marker in visited:
            break
        visited.add(marker)

        advanced = False
        for key in WRAPPER_KEYS:
            if key in current and isinstance(current[key], Mapping):
                current = current[key]
                advanced = True
                break
        if not advanced:
            break

    if not isinstance(current, Mapping):
        raise TypeError("Checkpoint payload is not a mapping after wrapper unwrapping.")

    tensor_map: Dict[str, torch.Tensor] = {}
    for key, value in current.items():
        if torch is not None and isinstance(value, torch.Tensor):
            tensor_map[str(key)] = value
    return tensor_map


def normalize_param_name(name: str) -> str:
    prefixes = ("module.", "_orig_mod.", "model.")
    normalized = name
    while True:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
        if not changed:
            break
    return normalized


class TensorStore:
    def __init__(self, path: Path | str):
        if torch is None:
            raise ImportError("TensorStore requires PyTorch. Install with: pip install torch")
        self.path = resolve_weight_file(path)
        self.suffix = self.path.suffix.lower()
        self._is_safetensors = self.suffix == ".safetensors"
        self._safe_file = None
        self._safe_key_map: Dict[str, str] | None = None
        self._state_dict: Dict[str, torch.Tensor] | None = None

    def _open_safetensors(self):
        if self._safe_file is not None:
            return self._safe_file

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors requires the 'safetensors' package. "
                "Install with: pip install safetensors"
            ) from exc

        self._safe_file = safe_open(str(self.path), framework="pt", device="cpu")
        return self._safe_file

    def _get_safetensors_key_map(self) -> Dict[str, str]:
        if self._safe_key_map is not None:
            return self._safe_key_map

        sf = self._open_safetensors()
        normalized_map: Dict[str, str] = {}
        for src_key in sf.keys():
            normalized_key = normalize_param_name(src_key)
            if normalized_key in normalized_map and normalized_map[normalized_key] != src_key:
                raise ValueError(
                    f"Normalized key collision in safetensors file {self.path}: "
                    f"{normalized_map[normalized_key]!r} and {src_key!r} both map to {normalized_key!r}"
                )
            normalized_map[normalized_key] = src_key
        self._safe_key_map = normalized_map
        return self._safe_key_map

    def _open_torch(self) -> Dict[str, torch.Tensor]:
        if self._state_dict is not None:
            return self._state_dict

        payload = torch.load(self.path, map_location="cpu")
        raw = _unwrap_checkpoint_dict(payload)

        normalized: Dict[str, torch.Tensor] = {}
        for key, tensor in raw.items():
            normalized_key = normalize_param_name(key)
            if normalized_key in normalized and normalized[normalized_key] is not tensor:
                raise ValueError(
                    f"Normalized key collision in torch checkpoint {self.path}: "
                    f"multiple keys map to {normalized_key!r}"
                )
            normalized[normalized_key] = tensor

        self._state_dict = normalized
        return self._state_dict

    def keys(self) -> set[str]:
        if self._is_safetensors:
            return set(self._get_safetensors_key_map().keys())
        return set(self._open_torch().keys())

    def get(self, key: str) -> torch.Tensor:
        normalized_key = normalize_param_name(key)
        if self._is_safetensors:
            sf = self._open_safetensors()
            key_map = self._get_safetensors_key_map()
            if normalized_key not in key_map:
                raise KeyError(normalized_key)
            return sf.get_tensor(key_map[normalized_key])

        state = self._open_torch()
        if normalized_key not in state:
            raise KeyError(normalized_key)
        return state[normalized_key]


def index_checkpoint_keys(store_a: TensorStore, store_b: TensorStore) -> KeyIndex:
    keys_a = store_a.keys()
    keys_b = store_b.keys()

    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    shape_mismatch: set[str] = set()
    comparable: set[str] = set()

    for key in sorted(common):
        tensor_a = store_a.get(key)
        tensor_b = store_b.get(key)

        if tensor_a.shape != tensor_b.shape:
            shape_mismatch.add(key)
            continue

        if tensor_a.is_floating_point() and tensor_b.is_floating_point():
            comparable.add(key)

    return KeyIndex(
        common_keys=common,
        only_in_a=only_a,
        only_in_b=only_b,
        shape_mismatch=shape_mismatch,
        comparable=comparable,
    )


def compute_raw_stats(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 1_000_000) -> RawStats:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    a_flat = a.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    b_flat = b.detach().reshape(-1).to(dtype=torch.float64, device="cpu")

    out = RawStats()
    total = int(a_flat.numel())

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        ac = a_flat[start:end]
        bc = b_flat[start:end]
        diff = bc - ac

        out.n += int(ac.numel())
        out.nan_count_a += int(torch.isnan(ac).sum().item())
        out.nan_count_b += int(torch.isnan(bc).sum().item())
        out.nan_count_diff += int(torch.isnan(diff).sum().item())
        out.inf_count_a += int(torch.isinf(ac).sum().item())
        out.inf_count_b += int(torch.isinf(bc).sum().item())
        out.inf_count_diff += int(torch.isinf(diff).sum().item())

        finite_mask = torch.isfinite(ac) & torch.isfinite(bc) & torch.isfinite(diff)
        if not bool(finite_mask.any()):
            continue

        ac_f = ac[finite_mask]
        bc_f = bc[finite_mask]
        diff_f = diff[finite_mask]

        out.sum_a2 += float((ac_f * ac_f).sum().item())
        out.sum_b2 += float((bc_f * bc_f).sum().item())
        out.sum_diff2 += float((diff_f * diff_f).sum().item())
        out.dot += float((ac_f * bc_f).sum().item())
        out.sum_abs_diff += float(diff_f.abs().sum().item())

        chunk_max = float(diff_f.abs().max().item())
        out.max_abs_diff = max(out.max_abs_diff, chunk_max)

        flips = ((ac_f > 0) & (bc_f < 0)) | ((ac_f < 0) & (bc_f > 0))
        out.sign_flip_count += int(flips.sum().item())

    return out


def finalize_stats(stats: RawStats, eps: float = 1e-12) -> FinalizedStats:
    norm_a = float(stats.sum_a2 ** 0.5)
    norm_b = float(stats.sum_b2 ** 0.5)
    diff_norm = float(stats.sum_diff2 ** 0.5)

    relative_diff = diff_norm / max(norm_a, eps)
    cosine_similarity = stats.dot / max(norm_a * norm_b, eps)
    mean_abs_diff = stats.sum_abs_diff / max(stats.n, 1)
    sign_flip_ratio = stats.sign_flip_count / max(stats.n, 1)

    return FinalizedStats(
        n=stats.n,
        norm_a=norm_a,
        norm_b=norm_b,
        diff_norm=diff_norm,
        relative_diff=relative_diff,
        cosine_similarity=cosine_similarity,
        mean_abs_diff=mean_abs_diff,
        max_abs_diff=stats.max_abs_diff,
        sign_flip_ratio=sign_flip_ratio,
        sign_flip_count=stats.sign_flip_count,
        nan_count_a=stats.nan_count_a,
        nan_count_b=stats.nan_count_b,
        nan_count_diff=stats.nan_count_diff,
        inf_count_a=stats.inf_count_a,
        inf_count_b=stats.inf_count_b,
        inf_count_diff=stats.inf_count_diff,
    )


def parameter_group_keys(param_name: str, component: str, layer_id: str, param_type: str) -> dict[str, str]:
    return {
        "global": "__all__",
        "component": component,
        "layer": f"{component}|{layer_id}",
        "param_type": f"{component}|{param_type}",
        "parameter": param_name,
    }


def load_component_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("component map json must be an object: {\"prefix\": \"component\"}")
    mapping: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("component map json entries must be string->string")
        mapping[key] = value
    return mapping


def component_of(key: str, component_map: Mapping[str, str] | None = None) -> str:
    if component_map:
        for prefix, component in sorted(component_map.items(), key=lambda kv: len(kv[0]), reverse=True):
            if key == prefix or key.startswith(f"{prefix}."):
                return component

    lowered = key.lower()
    action_patterns = (
        "paligemma_with_expert.gemma_expert",
        "gemma_expert",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "attn_vec_einsum_1",
        "kv_einsum_1",
        "q_einsum_1",
        "mlp_1",
        "final_norm_1",
        "pre_attention_norm_1",
        "pre_ffw_norm_1",
        "action_expert",
    )
    if any(p in lowered for p in action_patterns):
        return "action_expert"

    vit_patterns = ("vision_tower", "vision_model", "image_encoder", "img/", "/img/", "vit.")
    if any(p in lowered for p in vit_patterns) or ("patch_embedding" in lowered and "vision" in lowered):
        return "vit"

    vlm_patterns = ("language_model", "paligemma", "embed_tokens", "lm_head", "vlm.", "llm.")
    if any(p in lowered for p in vlm_patterns):
        return "vlm_backbone"

    return "other"


def layer_id_of(key: str, component: str) -> str:
    prefix = {"vit": "vit", "vlm_backbone": "vlm", "action_expert": "action_expert"}.get(
        component, "other"
    )
    lowered = key.lower()

    embedding_tokens = ("token_embedding", "embed_tokens", "word_embeddings", "tok_embeddings", "wte")
    if any(token in lowered for token in embedding_tokens):
        return f"{prefix}.token_embedding"
    if "lm_head" in lowered:
        return f"{prefix}.lm_head"
    if any(token in lowered for token in ("projection", "projections", "projector", "_proj", ".proj")):
        return f"{prefix}.projections"

    block_match = re.search(r"(?:^|\.)(?:blocks?|layers?|block|layer)\.(\d+)(?:\.|$)", lowered)
    if block_match:
        return f"{prefix}.block_{int(block_match.group(1)):02d}"
    return f"{prefix}.global"


def param_type_of(key: str) -> str:
    lowered = key.lower()
    if "action_projection" in lowered or ("action" in lowered and "proj" in lowered):
        return "action_projection"
    if "patch_embedding" in lowered:
        return "patch_embedding"
    if "position_embedding" in lowered or "position_embeddings" in lowered:
        return "position_embedding"
    if any(token in lowered for token in ("token_embedding", "embed_tokens", "word_embeddings", "tok_embeddings", "wte")):
        return "token_embedding"
    if "projector" in lowered or "multimodal" in lowered:
        return "multimodal_projector"
    if "attn" in lowered or "attention" in lowered:
        if "q_proj" in lowered:
            return "attention.q_proj"
        if "k_proj" in lowered:
            return "attention.k_proj"
        if "v_proj" in lowered:
            return "attention.v_proj"
        if "o_proj" in lowered or "out_proj" in lowered:
            return "attention.o_proj"
        return "attention.other"
    if any(token in lowered for token in ("mlp", "ffn", "feed_forward")):
        if "gate_proj" in lowered:
            return "mlp.gate_proj"
        if "up_proj" in lowered:
            return "mlp.up_proj"
        if "down_proj" in lowered:
            return "mlp.down_proj"
        return "mlp.other"
    if any(token in lowered for token in ("norm", "layernorm", "rmsnorm", ".ln", "ln_")):
        return "normalization"
    if "bias" in lowered:
        return "bias"
    return "other"


def write_key_classification_csv(
    store: TensorStore,
    keys: set[str],
    output_path: Path,
    component_map: Mapping[str, str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["name", "component", "layer_id", "param_type", "shape", "num_params"],
        )
        writer.writeheader()
        for key in sorted(keys):
            tensor = store.get(key)
            component = component_of(key, component_map=component_map)
            writer.writerow(
                {
                    "name": key,
                    "component": component,
                    "layer_id": layer_id_of(key, component),
                    "param_type": param_type_of(key),
                    "shape": str(tuple(tensor.shape)),
                    "num_params": int(tensor.numel()),
                }
            )


def compute_hierarchical_stats(
    pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]], chunk_size: int = 1_000_000
) -> dict[str, dict[str, FinalizedStats]]:
    accumulators: dict[str, dict[str, RawStats]] = {
        "global": {},
        "component": {},
        "layer": {},
        "param_type": {},
        "parameter": {},
    }

    for param_name, (a_tensor, b_tensor) in pairs.items():
        raw = compute_raw_stats(a_tensor, b_tensor, chunk_size=chunk_size)
        component = component_of(param_name)
        layer = layer_id_of(param_name, component)
        ptype = param_type_of(param_name)
        for level, group_key in parameter_group_keys(param_name, component, layer, ptype).items():
            level_map = accumulators[level]
            if group_key not in level_map:
                level_map[group_key] = RawStats()
            level_map[group_key].merge(raw)

    return {
        level: {group: finalize_stats(raw) for group, raw in grouped.items()}
        for level, grouped in accumulators.items()
    }


def _reservoir_extend(reservoir: list[float], values: list[float], limit: int, seen: int, rng: random.Random) -> int:
    for value in values:
        seen += 1
        if len(reservoir) < limit:
            reservoir.append(value)
            continue
        idx = rng.randrange(seen)
        if idx < limit:
            reservoir[idx] = value
    return seen


def _to_metric_row(name: str, component: str, layer_id: str, param_type: str, shape: str, stats: FinalizedStats) -> PerParameterRow:
    return PerParameterRow(
        name=name,
        component=component,
        layer_id=layer_id,
        param_type=param_type,
        shape=shape,
        num_params=stats.n,
        norm_a=stats.norm_a,
        norm_b=stats.norm_b,
        diff_norm=stats.diff_norm,
        relative_diff=stats.relative_diff,
        cosine_similarity=stats.cosine_similarity,
        mean_abs_diff=stats.mean_abs_diff,
        max_abs_diff=stats.max_abs_diff,
        sign_flip_ratio=stats.sign_flip_ratio,
    )


def write_simple_keys_csv(path: Path, header: str, keys: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([header])
        for key in keys:
            writer.writerow([key])


def write_shape_mismatch_csv(path: Path, keys: list[str], store_a: TensorStore, store_b: TensorStore) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["name", "shape_a", "shape_b"])
        for key in keys:
            writer.writerow([key, str(tuple(store_a.get(key).shape)), str(tuple(store_b.get(key).shape))])


def _write_summary_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 4.5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_hist(path: Path, data: list[float], title: str) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data, bins=80)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Weight diff (b-a)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_scatter(path: Path, x: list[float], y: list[float], title: str) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=2, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Weight A")
    ax.set_ylabel("Weight B")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip() or "unknown"
        )
    except Exception:
        return "unknown"


def _write_report(out_dir: Path, summary: dict[str, object], per_param: list[PerParameterRow], component_rows: list[dict[str, object]]) -> None:
    top20 = sorted(per_param, key=lambda r: r.relative_diff, reverse=True)[:20]
    by_rel = sorted(component_rows, key=lambda r: float(r["relative_diff"]), reverse=True)
    by_cos = sorted(component_rows, key=lambda r: float(r["cosine_similarity"]))

    lines = [
        "# Weight Diff Report",
        "",
        f"- Checkpoint A: `{summary['checkpoint_a_path']}`",
        f"- Checkpoint B: `{summary['checkpoint_b_path']}`",
        "",
        "## Global metrics",
        "",
    ]
    gm = summary["global_metrics"]
    lines.extend(
        [
            f"- relative_diff: {gm['relative_diff']:.6f}",
            f"- cosine_similarity: {gm['cosine_similarity']:.6f}",
            f"- mean_abs_diff: {gm['mean_abs_diff']:.6f}",
            "",
            "## Component ranking by relative_diff",
            "",
        ]
    )
    lines.extend([f"- {r['component']}: {float(r['relative_diff']):.6f}" for r in by_rel])
    lines.extend(["", "## Component ranking by lowest cosine_similarity", ""])
    lines.extend([f"- {r['component']}: {float(r['cosine_similarity']):.6f}" for r in by_cos])
    lines.extend(["", "## Top-20 changed parameters", ""])
    lines.extend([f"- {r.name}: relative_diff={r.relative_diff:.6f}, cosine={r.cosine_similarity:.6f}" for r in top20])
    lines.extend(
        [
            "",
            "## Missing keys and shape mismatches",
            "",
            f"- only_in_a: {summary['num_only_in_a']}",
            f"- only_in_b: {summary['num_only_in_b']}",
            f"- shape_mismatches: {summary['num_shape_mismatches']}",
            "",
            "## Interpretation notes",
            "",
            "- Weight-space distance is most interpretable when both checkpoints share architecture, base checkpoint, and action dimensionality.",
            "- These are weight-level observations; compare activations/logits/actions on shared validation observations for behavior-level conclusions.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def compare_and_write_outputs(args: argparse.Namespace) -> None:
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    store_a = TensorStore(args.a)
    store_b = TensorStore(args.b)
    idx = index_checkpoint_keys(store_a, store_b)
    component_map = load_component_map(args.component_map_json)

    if args.fail_on_shape_mismatch and idx.shape_mismatch:
        raise ValueError(f"Found {len(idx.shape_mismatch)} shape mismatches")

    write_key_classification_csv(store=store_a, keys=idx.comparable, output_path=out_dir / "key_classification.csv", component_map=component_map)
    write_simple_keys_csv(out_dir / "only_in_a.csv", "name", sorted(idx.only_in_a))
    write_simple_keys_csv(out_dir / "only_in_b.csv", "name", sorted(idx.only_in_b))
    write_shape_mismatch_csv(out_dir / "shape_mismatches.csv", sorted(idx.shape_mismatch), store_a, store_b)

    per_param_rows: list[PerParameterRow] = []
    raw_global = RawStats()
    raw_by_component: dict[str, RawStats] = {}
    raw_by_layer: dict[tuple[str, str], RawStats] = {}
    raw_by_type: dict[tuple[str, str], RawStats] = {}

    hist_global: list[float] = []
    hist_component: dict[str, list[float]] = {"vit": [], "vlm_backbone": [], "action_expert": []}
    scatter_x: list[float] = []
    scatter_y: list[float] = []
    seen_hist_global = 0
    seen_hist_component = {"vit": 0, "vlm_backbone": 0, "action_expert": 0}
    seen_scatter = 0

    for key in sorted(idx.comparable):
        a = store_a.get(key)
        b = store_b.get(key)
        component = component_of(key, component_map=component_map)
        layer_id = layer_id_of(key, component)
        ptype = param_type_of(key)

        raw = compute_raw_stats(a, b, chunk_size=args.chunk_size)
        final = finalize_stats(raw)

        raw_global.merge(raw)
        raw_by_component.setdefault(component, RawStats()).merge(raw)
        raw_by_layer.setdefault((component, layer_id), RawStats()).merge(raw)
        raw_by_type.setdefault((component, ptype), RawStats()).merge(raw)

        per_param_rows.append(
            _to_metric_row(key, component, layer_id, ptype, str(tuple(a.shape)), final)
        )

        a_flat = a.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        b_flat = b.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        diff = (b_flat - a_flat)
        mask = torch.isfinite(a_flat) & torch.isfinite(b_flat) & torch.isfinite(diff)
        if bool(mask.any()):
            dv = diff[mask].tolist()
            seen_hist_global = _reservoir_extend(hist_global, dv, args.histogram_sample_size, seen_hist_global, rng)
            if component in hist_component:
                seen_hist_component[component] = _reservoir_extend(
                    hist_component[component], dv, args.histogram_sample_size, seen_hist_component[component], rng
                )
            av = a_flat[mask].tolist()
            bv = b_flat[mask].tolist()
            points = list(zip(av, bv))
            seen_scatter += len(points)
            for x, y in points:
                if len(scatter_x) < args.scatter_sample_size:
                    scatter_x.append(x)
                    scatter_y.append(y)
                else:
                    idx_s = rng.randrange(seen_scatter)
                    if idx_s < args.scatter_sample_size:
                        scatter_x[idx_s] = x
                        scatter_y[idx_s] = y

    global_stats = finalize_stats(raw_global)
    component_metrics = {k: finalize_stats(v) for k, v in raw_by_component.items()}

    component_rows = [
        {
            "component": c,
            "num_params": s.n,
            "norm_a": s.norm_a,
            "norm_b": s.norm_b,
            "diff_norm": s.diff_norm,
            "relative_diff": s.relative_diff,
            "cosine_similarity": s.cosine_similarity,
            "mean_abs_diff": s.mean_abs_diff,
            "max_abs_diff": s.max_abs_diff,
            "sign_flip_ratio": s.sign_flip_ratio,
        }
        for c, s in sorted(component_metrics.items())
    ]
    _write_summary_csv(
        out_dir / "component_summary.csv",
        ["component", "num_params", "norm_a", "norm_b", "diff_norm", "relative_diff", "cosine_similarity", "mean_abs_diff", "max_abs_diff", "sign_flip_ratio"],
        component_rows,
    )

    layer_rows = []
    for (component, layer_id), raw in sorted(raw_by_layer.items()):
        s = finalize_stats(raw)
        layer_rows.append(
            {
                "component": component,
                "layer_id": layer_id,
                "num_params": s.n,
                "norm_a": s.norm_a,
                "norm_b": s.norm_b,
                "diff_norm": s.diff_norm,
                "relative_diff": s.relative_diff,
                "cosine_similarity": s.cosine_similarity,
                "mean_abs_diff": s.mean_abs_diff,
                "max_abs_diff": s.max_abs_diff,
                "sign_flip_ratio": s.sign_flip_ratio,
            }
        )
    _write_summary_csv(
        out_dir / "layer_summary.csv",
        ["component", "layer_id", "num_params", "norm_a", "norm_b", "diff_norm", "relative_diff", "cosine_similarity", "mean_abs_diff", "max_abs_diff", "sign_flip_ratio"],
        layer_rows,
    )

    type_rows = []
    for (component, param_type), raw in sorted(raw_by_type.items()):
        s = finalize_stats(raw)
        type_rows.append(
            {
                "component": component,
                "param_type": param_type,
                "num_params": s.n,
                "norm_a": s.norm_a,
                "norm_b": s.norm_b,
                "diff_norm": s.diff_norm,
                "relative_diff": s.relative_diff,
                "cosine_similarity": s.cosine_similarity,
                "mean_abs_diff": s.mean_abs_diff,
                "max_abs_diff": s.max_abs_diff,
                "sign_flip_ratio": s.sign_flip_ratio,
            }
        )
    _write_summary_csv(
        out_dir / "parameter_type_summary.csv",
        ["component", "param_type", "num_params", "norm_a", "norm_b", "diff_norm", "relative_diff", "cosine_similarity", "mean_abs_diff", "max_abs_diff", "sign_flip_ratio"],
        type_rows,
    )

    _write_summary_csv(
        out_dir / "per_parameter.csv",
        ["name", "component", "layer_id", "param_type", "shape", "num_params", "norm_a", "norm_b", "diff_norm", "relative_diff", "cosine_similarity", "mean_abs_diff", "max_abs_diff", "sign_flip_ratio"],
        [asdict(r) for r in per_param_rows],
    )

    summary = {
        "checkpoint_a_path": str(store_a.path),
        "checkpoint_b_path": str(store_b.path),
        "num_common_keys": len(idx.common_keys),
        "num_compared_tensors": len(idx.comparable),
        "num_shape_mismatches": len(idx.shape_mismatch),
        "num_only_in_a": len(idx.only_in_a),
        "num_only_in_b": len(idx.only_in_b),
        "global_metrics": asdict(global_stats),
        "component_metrics": {k: asdict(v) for k, v in component_metrics.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_version_or_git_commit_if_available": _git_commit(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if component_rows:
        _plot_bar(
            out_dir / "component_relative_diff.png",
            [r["component"] for r in component_rows],
            [float(r["relative_diff"]) for r in component_rows],
            "Component relative diff",
            "relative_diff",
        )
        _plot_bar(
            out_dir / "component_cosine_similarity.png",
            [r["component"] for r in component_rows],
            [float(r["cosine_similarity"]) for r in component_rows],
            "Component cosine similarity",
            "cosine_similarity",
        )
        _plot_bar(
            out_dir / "component_mean_abs_diff.png",
            [r["component"] for r in component_rows],
            [float(r["mean_abs_diff"]) for r in component_rows],
            "Component mean abs diff",
            "mean_abs_diff",
        )

    for comp in ("vit", "vlm_backbone", "action_expert"):
        rows = [r for r in layer_rows if r["component"] == comp]
        if not rows:
            LOGGER.warning("No comparable tensors for component %s; skipping layer plots", comp)
            continue
        _plot_bar(
            out_dir / f"layer_relative_diff_{comp}.png",
            [str(r["layer_id"]) for r in rows],
            [float(r["relative_diff"]) for r in rows],
            f"Layer relative diff ({comp})",
            "relative_diff",
        )
        _plot_bar(
            out_dir / f"layer_cosine_similarity_{comp}.png",
            [str(r["layer_id"]) for r in rows],
            [float(r["cosine_similarity"]) for r in rows],
            f"Layer cosine similarity ({comp})",
            "cosine_similarity",
        )

    if type_rows:
        _plot_bar(
            out_dir / "parameter_type_relative_diff.png",
            [f"{r['component']}:{r['param_type']}" for r in type_rows],
            [float(r["relative_diff"]) for r in type_rows],
            "Parameter type relative diff",
            "relative_diff",
        )

    topk = sorted(per_param_rows, key=lambda r: r.relative_diff, reverse=True)[: args.top_k]
    if topk:
        _plot_bar(
            out_dir / "top_changed_parameters.png",
            [r.name for r in topk],
            [r.relative_diff for r in topk],
            f"Top {len(topk)} changed parameters",
            "relative_diff",
        )

    if hist_global:
        _plot_hist(out_dir / "weight_diff_histogram_global.png", hist_global, "Weight diff histogram (global)")

    for comp in ("vit", "vlm_backbone", "action_expert"):
        data = hist_component[comp]
        if not data:
            LOGGER.warning("No comparable tensors for component %s; skipping histogram", comp)
            continue
        _plot_hist(out_dir / f"weight_diff_histogram_{comp}.png", data, f"Weight diff histogram ({comp})")

    if scatter_x and scatter_y:
        _plot_scatter(out_dir / "weight_scatter_global.png", scatter_x, scatter_y, "Weight scatter (global)")

    _write_report(out_dir, summary, per_param_rows, component_rows)

    print(
        f"common={len(idx.common_keys)} only_in_a={len(idx.only_in_a)} only_in_b={len(idx.only_in_b)} "
        f"shape_mismatch={len(idx.shape_mismatch)} comparable={len(idx.comparable)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OpenPI checkpoints.")
    parser.add_argument("--a", type=Path, required=True, help="Checkpoint A path or directory")
    parser.add_argument("--b", type=Path, required=True, help="Checkpoint B path or directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--component-map-json", type=Path, default=None)
    parser.add_argument("--include-other", action="store_true", default=True)
    parser.add_argument("--exclude-buffers", action="store_true", default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--histogram-sample-size", type=int, default=1_000_000)
    parser.add_argument("--scatter-sample-size", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=10_000_000)
    parser.add_argument("--fail-on-shape-mismatch", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--key-classification-csv",
        type=Path,
        default=Path("key_classification.csv"),
        help="Backward-compatible option (ignored when --out is used).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    compare_and_write_outputs(args)
