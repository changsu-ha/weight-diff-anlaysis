from pathlib import Path

import pytest

from scripts.compare_pi05_weights import (
    RawStats,
    compare_and_write_outputs,
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


def test_same_structure_without_shape_mismatch(tmp_path: Path):
    ckpt_a = {
        "state_dict": {
            "module.vit.blocks.0.attn.q_proj.weight": torch.ones(2, 2),
            "module.vit.blocks.0.attn.q_proj.bias": torch.zeros(2),
        }
    }
    ckpt_b = {
        "model": {
            "_orig_mod.vit.blocks.0.attn.q_proj.weight": torch.ones(2, 2) * 1.5,
            "_orig_mod.vit.blocks.0.attn.q_proj.bias": torch.ones(2),
        }
    }

    path_a = tmp_path / "same_a.pt"
    path_b = tmp_path / "same_b.pt"
    torch.save(ckpt_a, path_a)
    torch.save(ckpt_b, path_b)

    index = index_checkpoint_keys(TensorStore(path_a), TensorStore(path_b))
    assert index.shape_mismatch == set()
    assert index.only_in_a == set()
    assert index.only_in_b == set()
    assert index.comparable == {
        "vit.blocks.0.attn.q_proj.weight",
        "vit.blocks.0.attn.q_proj.bias",
    }


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
        "encoder.layer.1.weight": (torch.tensor([1.0, -1.0]), torch.tensor([1.0, 1.0])),
        "encoder.layer.1.bias": (torch.tensor([0.5]), torch.tensor([0.0])),
        "head.weight": (torch.tensor([2.0]), torch.tensor([1.0])),
    }

    summary = compute_hierarchical_stats(pairs, chunk_size=1)

    assert set(summary.keys()) == {"global", "component", "layer", "param_type", "parameter"}
    assert "__all__" in summary["global"]
    assert "other" in summary["component"]
    assert "other|other.block_01" in summary["layer"]
    assert "other|other" in summary["param_type"]
    assert "encoder.layer.1.weight" in summary["parameter"]

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
    assert param_type_of("vlm.layers.0.self_attn.q_proj.weight") == "attention.q_proj"
    assert param_type_of("vit.blocks.0.mlp.fc1.weight") == "mlp.other"
    assert param_type_of("vlm.layers.0.input_layernorm.weight") == "normalization"
    assert param_type_of("vlm.embed_tokens.weight") == "token_embedding"
    assert param_type_of("action_expert.action_projection.weight") == "action_projection"


def test_openpi_representative_key_classification(tmp_path: Path):
    ckpt_path = tmp_path / "openpi.pt"
    torch.save(
        {
            "vit.blocks.0.attn.q_proj.weight": torch.ones(2, 2),
            "vlm.layers.3.self_attn.k_proj.weight": torch.ones(2, 2),
            "action_expert.blocks.7.mlp.fc1.weight": torch.ones(2, 2),
        },
        ckpt_path,
    )
    store = TensorStore(ckpt_path)

    out_csv = tmp_path / "openpi_key_classification.csv"
    write_key_classification_csv(
        store=store,
        keys={
            "vit.blocks.0.attn.q_proj.weight",
            "vlm.layers.3.self_attn.k_proj.weight",
            "action_expert.blocks.7.mlp.fc1.weight",
        },
        output_path=out_csv,
        component_map=None,
    )
    rows = out_csv.read_text(encoding="utf-8")
    assert "vit.blocks.0.attn.q_proj.weight,vit,vit.block_00,attention.q_proj" in rows
    assert "vlm.layers.3.self_attn.k_proj.weight,vlm_backbone,vlm.block_03,attention.k_proj" in rows
    assert "action_expert.blocks.7.mlp.fc1.weight,action_expert,action_expert.block_07,mlp.fc1" in rows


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
    assert "my_custom.blocks.0.attn.q_proj.weight,vit,vit.block_00,attention.q_proj,\"(2, 3)\",6" in csv_text


def test_full_output_artifacts(tmp_path: Path):
    pytest.importorskip("matplotlib")
    a_path = tmp_path / "a.pt"
    b_path = tmp_path / "b.pt"
    out = tmp_path / "out"

    torch.save(
        {
            "vit.blocks.0.attn.q_proj.weight": torch.ones(2, 2),
            "vlm.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
            "action_expert.blocks.0.weight": torch.ones(2),
            "only_a": torch.ones(1),
            "shape_x": torch.ones(2, 3),
        },
        a_path,
    )
    torch.save(
        {
            "vit.blocks.0.attn.q_proj.weight": torch.ones(2, 2) * 2,
            "vlm.layers.0.mlp.up_proj.weight": torch.ones(2, 2) * 0.5,
            "action_expert.blocks.0.weight": torch.ones(2) * -1,
            "only_b": torch.ones(1),
            "shape_x": torch.ones(3, 2),
        },
        b_path,
    )

    class A:
        pass

    ns = A()
    ns.a = a_path
    ns.b = b_path
    ns.out = out
    ns.component_map_json = None
    ns.include_other = True
    ns.exclude_buffers = True
    ns.top_k = 10
    ns.histogram_sample_size = 10
    ns.scatter_sample_size = 10
    ns.chunk_size = 2
    ns.fail_on_shape_mismatch = False
    ns.verbose = False
    ns.key_classification_csv = Path("unused.csv")

    compare_and_write_outputs(ns)

    required = [
        "component_summary.csv",
        "layer_summary.csv",
        "parameter_type_summary.csv",
        "per_parameter.csv",
        "shape_mismatches.csv",
        "only_in_a.csv",
        "only_in_b.csv",
        "summary.json",
        "report.md",
        "component_relative_diff.png",
        "component_cosine_similarity.png",
        "component_mean_abs_diff.png",
        "layer_relative_diff_vit.png",
        "layer_relative_diff_vlm_backbone.png",
        "layer_relative_diff_action_expert.png",
        "layer_cosine_similarity_vit.png",
        "layer_cosine_similarity_vlm_backbone.png",
        "layer_cosine_similarity_action_expert.png",
        "parameter_type_relative_diff.png",
        "top_changed_parameters.png",
        "weight_diff_histogram_global.png",
        "weight_diff_histogram_vit.png",
        "weight_diff_histogram_vlm_backbone.png",
        "weight_diff_histogram_action_expert.png",
        "weight_scatter_global.png",
    ]
    for name in required:
        assert (out / name).exists(), name

    payload = (out / "summary.json").read_text(encoding="utf-8")
    assert '"num_shape_mismatches": 1' in payload
    assert '"num_only_in_a": 1' in payload
    assert '"num_only_in_b": 1' in payload
