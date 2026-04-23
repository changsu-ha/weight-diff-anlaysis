from pathlib import Path

import pytest

from scripts.compare_pi05_weights import (
    RawStats,
    compute_hierarchical_stats,
    compute_raw_stats,
    component_of,
    finalize_stats,
    layer_id_of,
    load_component_map,
    param_type_of,
    TensorStore,
    index_checkpoint_keys,
    normalize_param_name,
    resolve_weight_file,
    write_key_classification_csv,
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


def test_key_classification_helpers():
    assert component_of("action_expert.blocks.3.attn.q_proj.weight") == "action_expert"
    assert component_of("vit.blocks.1.mlp.fc1.weight") == "vit"
    assert component_of("vlm.layers.2.self_attn.q_proj.weight") == "vlm_backbone"
    assert layer_id_of("vit.blocks.12.attn.q_proj.weight", "vit") == "vit.block_12"
    assert layer_id_of("vlm.embed_tokens.weight", "vlm_backbone") == "vlm.token_embedding"
    assert layer_id_of("vlm.lm_head.weight", "vlm_backbone") == "vlm.lm_head"
    assert layer_id_of("action_expert.action_projection.weight", "action_expert") == "action_expert.projections"
    assert param_type_of("vlm.layers.0.self_attn.q_proj.weight") == "attention"
    assert param_type_of("vit.blocks.0.mlp.fc1.weight") == "mlp"
    assert param_type_of("vlm.layers.0.input_layernorm.weight") == "norm"
    assert param_type_of("vlm.embed_tokens.weight") == "embedding"
    assert param_type_of("action_expert.action_projection.weight") == "action_projection"


def test_component_map_and_csv_output(tmp_path: Path):
    map_path = tmp_path / "component_map.json"
    map_path.write_text('{"my_custom":"vit"}', encoding="utf-8")
    assert load_component_map(map_path) == {"my_custom": "vit"}

    ckpt_path = tmp_path / "a.pt"
    torch.save({"my_custom.blocks.0.attn.q_proj.weight": torch.ones(2, 3)}, ckpt_path)
    store = TensorStore(ckpt_path)

    out_csv = tmp_path / "key_classification.csv"
    write_key_classification_csv(
        store=store,
        keys={"my_custom.blocks.0.attn.q_proj.weight"},
        output_path=out_csv,
        component_map={"my_custom": "vit"},
    )
    csv_text = out_csv.read_text(encoding="utf-8")
    assert "name,component,layer_id,param_type,shape,num_params" in csv_text
    assert "my_custom.blocks.0.attn.q_proj.weight,vit,vit.block_00,attention,\"(2, 3)\",6" in csv_text
