from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, Mapping

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised in environments without torch
        torch = None

PREFERRED_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
    "checkpoint.pt",
)
GLOB_PATTERNS = ("*.safetensors", "*.pt", "*.pth", "*.bin")
WRAPPER_KEYS = ("state_dict", "model_state_dict", "model", "module")


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



def resolve_weight_file(path: Path | str) -> Path:
    """Resolve a weight file from a direct file path or a directory.

    Directory priority:
    model.safetensors -> pytorch_model.bin -> model.pt -> checkpoint.pt -> glob fallback.
    """
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
    """Unwrap common checkpoint wrappers in order until a tensor mapping is reached."""
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
    """Remove common wrapper prefixes from parameter names repeatedly."""
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
    """Unified checkpoint reader for safetensors and torch serialized checkpoints."""

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

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        for key in sorted(self.keys()):
            yield key, self.get(key)



def index_checkpoint_keys(store_a: TensorStore, store_b: TensorStore) -> KeyIndex:
    """Split keys by shared/missing/shape-mismatch/comparable sets.

    Comparable tensors are common keys where both tensors are floating-point and shapes match.
    """
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
    """Compute chunked accumulators from two tensors of identical shape."""
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
        diff = ac - bc

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

        # Sign flip: strict opposite sign only (0 is neutral and excluded).
        flips = ((ac_f > 0) & (bc_f < 0)) | ((ac_f < 0) & (bc_f > 0))
        out.sign_flip_count += int(flips.sum().item())

    return out


def finalize_stats(stats: RawStats, eps: float = 1e-12) -> FinalizedStats:
    """Convert raw accumulators into normalized metrics."""
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


def parameter_group_keys(param_name: str) -> dict[str, str]:
    """Build reusable grouping keys for multi-level aggregation."""
    tokens = param_name.split(".")
    component = tokens[0] if tokens else "unknown"
    layer = ".".join(tokens[:2]) if len(tokens) >= 2 else component
    param_type = tokens[-1] if tokens else "unknown"

    return {
        "global": "__all__",
        "component": component,
        "layer": layer,
        "param_type": param_type,
        "parameter": param_name,
    }


def compute_hierarchical_stats(
    pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]], chunk_size: int = 1_000_000
) -> dict[str, dict[str, FinalizedStats]]:
    """Reuse the same raw/finalize pipeline for global/component/layer/type/parameter levels."""
    accumulators: dict[str, dict[str, RawStats]] = {
        "global": {},
        "component": {},
        "layer": {},
        "param_type": {},
        "parameter": {},
    }

    for param_name, (a_tensor, b_tensor) in pairs.items():
        raw = compute_raw_stats(a_tensor, b_tensor, chunk_size=chunk_size)
        for level, group_key in parameter_group_keys(param_name).items():
            level_map = accumulators[level]
            if group_key not in level_map:
                level_map[group_key] = RawStats()
            level_map[group_key].merge(raw)

    return {
        level: {group: finalize_stats(raw) for group, raw in grouped.items()}
        for level, grouped in accumulators.items()
    }



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OpenPI checkpoints.")
    parser.add_argument("--a", type=Path, required=True, help="Checkpoint A path or directory")
    parser.add_argument("--b", type=Path, required=True, help="Checkpoint B path or directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    store_a = TensorStore(args.a)
    store_b = TensorStore(args.b)
    idx = index_checkpoint_keys(store_a, store_b)
    print(
        f"common={len(idx.common_keys)} only_in_a={len(idx.only_in_a)} only_in_b={len(idx.only_in_b)} "
        f"shape_mismatch={len(idx.shape_mismatch)} comparable={len(idx.comparable)}"
    )
