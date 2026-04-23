from pathlib import Path

import pytest

from scripts.compare_pi05_weights import (
    RawStats,
    compute_hierarchical_stats,
    compute_raw_stats,
    finalize_stats,
    TensorStore,
    index_checkpoint_keys,
    normalize_param_name,
    resolve_weight_file,
)

torch = pytest.importorskip("torch")


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


def test_compute_and_finalize_stats_chunked():
    a = torch.tensor([1.0, -2.0, 0.0, float("nan"), float("inf")])
    b = torch.tensor([-1.0, -1.0, 0.0, 0.0, float("inf")])

    raw = compute_raw_stats(a, b, chunk_size=2)

    assert raw.n == 5
    assert raw.nan_count_a == 1
    assert raw.nan_count_b == 0
    assert raw.inf_count_a == 1
    assert raw.inf_count_b == 1
    assert raw.sign_flip_count == 1

    final = finalize_stats(raw)
    assert final.n == 5
    assert final.max_abs_diff == 2.0
    assert final.mean_abs_diff == pytest.approx(3.0 / 5.0)
    assert final.sign_flip_ratio == pytest.approx(1.0 / 5.0)


def test_hierarchical_stats_reuse_pipeline():
    pairs = {
        "encoder.layer1.weight": (torch.tensor([1.0, -1.0]), torch.tensor([1.0, 1.0])),
        "encoder.layer1.bias": (torch.tensor([0.5]), torch.tensor([0.0])),
        "head.weight": (torch.tensor([2.0]), torch.tensor([1.0])),
    }

    summary = compute_hierarchical_stats(pairs, chunk_size=1)

    assert set(summary.keys()) == {"global", "component", "layer", "param_type", "parameter"}
    assert "__all__" in summary["global"]
    assert "encoder" in summary["component"]
    assert "encoder.layer1" in summary["layer"]
    assert "weight" in summary["param_type"]
    assert "encoder.layer1.weight" in summary["parameter"]

    global_stats = summary["global"]["__all__"]
    assert global_stats.n == 4
    assert isinstance(global_stats, type(finalize_stats(RawStats())))
