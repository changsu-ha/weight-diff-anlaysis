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
