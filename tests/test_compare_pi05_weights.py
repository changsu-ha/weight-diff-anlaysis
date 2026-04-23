from pathlib import Path

import torch

from scripts.compare_pi05_weights import (
    TensorStore,
    index_checkpoint_keys,
    normalize_param_name,
    resolve_weight_file,
)


def test_resolve_weight_file_priority(tmp_path: Path):
    (tmp_path / "model.pt").write_bytes(b"x")
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")
    resolved = resolve_weight_file(tmp_path)
    assert resolved.name == "pytorch_model.bin"


def test_normalize_param_name_prefixes():
    assert normalize_param_name("module._orig_mod.model.layer.weight") == "layer.weight"


def test_torch_wrapper_and_indexing(tmp_path: Path):
    ckpt_a = {
        "state_dict": {
            "module.layer.weight": torch.ones(2, 2),
            "module.layer.bias": torch.ones(2),
            "module.counter": torch.tensor([1, 2], dtype=torch.int64),
            "module.mismatch": torch.ones(2, 1),
        }
    }
    ckpt_b = {
        "model": {
            "_orig_mod.layer.weight": torch.ones(2, 2) * 2,
            "_orig_mod.layer.bias": torch.ones(2) * 3,
            "_orig_mod.counter": torch.tensor([1, 2], dtype=torch.int64),
            "_orig_mod.mismatch": torch.ones(1, 2),
            "_orig_mod.extra": torch.ones(1),
        }
    }

    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(ckpt_a, path_a)
    torch.save(ckpt_b, path_b)

    store_a = TensorStore(path_a)
    store_b = TensorStore(path_b)
    index = index_checkpoint_keys(store_a, store_b)

    assert "layer.weight" in index.common_keys
    assert "extra" in index.only_in_b
    assert "mismatch" in index.shape_mismatch
    assert "layer.weight" in index.comparable
    assert "counter" not in index.comparable
