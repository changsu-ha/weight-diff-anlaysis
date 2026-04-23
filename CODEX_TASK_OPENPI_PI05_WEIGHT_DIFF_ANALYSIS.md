# Codex Task: Implement OpenPI π0.5 Weight Difference Analysis

## One-line objective

Implement a reusable CLI tool for OpenPI `pi0.5` checkpoints that compares two model weight files numerically and visually, with explicit breakdowns for:

1. ViT / vision encoder
2. VLM backbone
3. Action expert
4. Other or unclassified parameters

The tool should help users answer: **When the same OpenPI `pi0.5` model is trained or fine-tuned on two different datasets, where and by how much did the weights change?**

---

## Context and intent

We have two checkpoints of the same OpenPI `pi0.5` model architecture. They were trained or fine-tuned on different datasets. We need to compare how different the resulting weights are.

A single global number is not sufficient. The comparison must be interpretable at multiple levels:

- global model-level difference
- component-level difference: ViT, VLM backbone, action expert
- layer/block-level difference inside each component
- parameter-type-level difference: attention, MLP, normalization, embeddings, projections, action projections
- individual parameter-level difference for debugging

The primary use case is comparing two PyTorch-converted OpenPI checkpoints, typically `.safetensors`, but the implementation should also support common PyTorch checkpoint formats where feasible.

---

## Expected repository integration

Implement this as a non-invasive analysis utility. Do **not** change training, inference, model definition, dataset, or checkpoint conversion behavior.

Preferred locations, in order:

1. `scripts/compare_pi05_weights.py`
2. `tools/compare_pi05_weights.py`
3. If the repository already has an analysis or examples directory, use the most consistent existing location.

Also add a short usage document if the repo has a docs area. If no obvious docs location exists, add:

```text
docs/compare_pi05_weights.md
```

The script must be runnable from the repository root.

---

## CLI interface

Provide a command-line interface similar to:

```bash
python scripts/compare_pi05_weights.py \
  --a /path/to/checkpoint_a_or_dir \
  --b /path/to/checkpoint_b_or_dir \
  --out /path/to/output_dir
```

Required arguments:

- `--a`: first checkpoint file or directory
- `--b`: second checkpoint file or directory
- `--out`: output directory

Recommended optional arguments:

- `--include-other`: include unclassified parameters in aggregate plots; default true
- `--exclude-buffers`: exclude non-floating tensors and obvious non-trainable buffers; default true
- `--top-k`: number of most changed parameters to show in top-change plot; default 50
- `--histogram-sample-size`: max number of elementwise diffs to sample for histograms; default 1_000_000
- `--scatter-sample-size`: max number of weight pairs to sample for scatter plots; default 100_000
- `--chunk-size`: max number of tensor elements processed at once; default around 10_000_000
- `--fail-on-shape-mismatch`: fail if common keys have shape mismatch; default false
- `--component-map-json`: optional user-supplied JSON file for overriding key-to-component mapping
- `--verbose`: print additional key classification and mismatch information

The tool should print a concise summary to stdout and write full artifacts to `--out`.

---

## Checkpoint loading requirements

The utility must accept either a weight file or a directory.

When a directory is provided, search for common names in this order:

```text
model.safetensors
pytorch_model.bin
model.pt
checkpoint.pt
*.safetensors
*.pt
*.pth
*.bin
```

Support these formats:

- `.safetensors` using `safetensors.safe_open`
- `.pt`, `.pth`, `.bin` using `torch.load(..., map_location="cpu")`

For PyTorch checkpoint dictionaries, robustly handle common wrappers:

```python
state_dict
model_state_dict
model
module
```

Normalize parameter names by removing common prefixes:

```text
module.
_orig_mod.
model.
```

Only compare tensors that are:

- present in both checkpoints
- same shape
- floating point

For everything else, write diagnostic CSV files rather than silently ignoring them.

---

## Numerical metrics

For each comparable parameter tensor, flatten both tensors and compute:

```text
num_params
norm_a
norm_b
diff_norm
relative_diff
cosine_similarity
mean_abs_diff
max_abs_diff
sign_flip_ratio
```

Definitions:

```text
diff = weight_b - weight_a
norm_a = ||weight_a||_2
norm_b = ||weight_b||_2
diff_norm = ||diff||_2
relative_diff = ||weight_b - weight_a||_2 / (||weight_a||_2 + eps)
cosine_similarity = dot(weight_a, weight_b) / ((||weight_a||_2 * ||weight_b||_2) + eps)
mean_abs_diff = mean(abs(diff))
max_abs_diff = max(abs(diff))
sign_flip_ratio = mean(weight_a * weight_b < 0)
```

Use `eps = 1e-12` or equivalent to avoid division-by-zero.

The implementation must aggregate raw sufficient statistics instead of averaging already-finalized parameter metrics. For example, component-level `relative_diff` must be computed from aggregated sums of squares, not from the mean of per-parameter relative differences.

Maintain raw aggregate stats such as:

```text
n
sum_a2
sum_b2
sum_diff2
dot
sum_abs_diff
max_abs_diff
sign_flip_count
```

Then finalize metrics from the aggregate.

Process large tensors in chunks to avoid excessive memory use.

---

## Component classification

Implement a classification function:

```python
def component_of(key: str) -> str:
    ...
```

It should return one of:

```text
vit
vlm_backbone
action_expert
other
```

The exact OpenPI key names may vary depending on conversion path and version, so inspect actual keys and keep the matching logic robust. Do not hard-code a single brittle layout.

Initial mapping heuristics:

### Action expert

Classify as `action_expert` if the key contains or starts with patterns such as:

```text
paligemma_with_expert.gemma_expert
gemma_expert
action_in_proj
action_out_proj
time_mlp_in
time_mlp_out
state_proj
action_time_mlp_in
action_time_mlp_out
```

Also support JAX-style or converted expert suffix patterns if present:

```text
attn_vec_einsum_1
kv_einsum_1
q_einsum_1
mlp_1
final_norm_1
pre_attention_norm_1
pre_ffw_norm_1
```

### ViT / vision encoder

Classify as `vit` if the key contains or starts with patterns such as:

```text
vision_tower
vision_model
image_encoder
img/
/img/
patch_embedding
```

Be careful: `patch_embedding` alone should only imply `vit` when the surrounding key context suggests the vision model.

### VLM backbone

Classify as `vlm_backbone` if the key contains or starts with patterns such as:

```text
language_model
paligemma
embed_tokens
lm_head
multi_modal_projector
llm/
```

Do not classify `gemma_expert` as VLM backbone even if it contains `gemma`; action expert must take precedence over VLM backbone.

### Other

Everything not matched above should be `other`.

Write a `key_classification.csv` with columns:

```text
name
component
layer_id
param_type
shape
num_params
```

This file is important because users must be able to audit whether the heuristic classified OpenPI keys correctly.

---

## Layer/block classification

Implement:

```python
def layer_id_of(key: str, component: str) -> str:
    ...
```

Suggested behavior:

### ViT

Detect vision blocks such as:

```text
vision_model.encoder.layers.<N>
vision_tower.vision_model.encoder.layers.<N>
```

Return:

```text
vit.block_00
vit.block_01
...
```

Special cases:

```text
vit.patch_embedding
vit.position_embedding
vit.post_layernorm
vit.other
```

### VLM backbone

Detect language model blocks such as:

```text
language_model.layers.<N>
model.layers.<N>
```

Return:

```text
vlm.block_00
vlm.block_01
...
```

Special cases:

```text
vlm.token_embedding
vlm.multimodal_projector
vlm.final_norm
vlm.lm_head
vlm.other
```

### Action expert

Detect expert blocks such as:

```text
gemma_expert.model.layers.<N>
gemma_expert.layers.<N>
```

Return:

```text
action_expert.block_00
action_expert.block_01
...
```

Special cases:

```text
action_expert.projections
action_expert.final_norm
action_expert.other
```

---

## Parameter type classification

Implement:

```python
def param_type_of(key: str) -> str:
    ...
```

Return useful types such as:

```text
patch_embedding
position_embedding
token_embedding
multimodal_projector
attention.q_proj
attention.k_proj
attention.v_proj
attention.o_proj
attention.other
mlp.gate_proj
mlp.up_proj
mlp.down_proj
mlp.other
normalization
action_projection
bias
other
```

The goal is not perfect taxonomy. The goal is a useful breakdown that helps users see whether differences concentrate in attention, MLP, normalization, embeddings, or action projection layers.

---

## Output files

Write the following CSV files:

```text
component_summary.csv
layer_summary.csv
parameter_type_summary.csv
per_parameter.csv
key_classification.csv
shape_mismatches.csv
only_in_a.csv
only_in_b.csv
summary.json
```

### `component_summary.csv`

One row per component:

```text
component
num_params
norm_a
norm_b
diff_norm
relative_diff
cosine_similarity
mean_abs_diff
max_abs_diff
sign_flip_ratio
```

### `layer_summary.csv`

One row per component/layer:

```text
component
layer_id
num_params
norm_a
norm_b
diff_norm
relative_diff
cosine_similarity
mean_abs_diff
max_abs_diff
sign_flip_ratio
```

### `parameter_type_summary.csv`

One row per component/parameter type:

```text
component
param_type
num_params
norm_a
norm_b
diff_norm
relative_diff
cosine_similarity
mean_abs_diff
max_abs_diff
sign_flip_ratio
```

### `per_parameter.csv`

One row per compared parameter:

```text
name
component
layer_id
param_type
shape
num_params
norm_a
norm_b
diff_norm
relative_diff
cosine_similarity
mean_abs_diff
max_abs_diff
sign_flip_ratio
```

### `summary.json`

Include:

```text
checkpoint_a_path
checkpoint_b_path
num_common_keys
num_compared_tensors
num_shape_mismatches
num_only_in_a
num_only_in_b
global_metrics
component_metrics
created_at
script_version_or_git_commit_if_available
```

---

## Required plots

Use `matplotlib`. Do not require a display server; use a non-interactive backend such as `Agg` if necessary.

Generate at least these PNG files:

```text
component_relative_diff.png
component_cosine_similarity.png
component_mean_abs_diff.png
layer_relative_diff_vit.png
layer_relative_diff_vlm_backbone.png
layer_relative_diff_action_expert.png
layer_cosine_similarity_vit.png
layer_cosine_similarity_vlm_backbone.png
layer_cosine_similarity_action_expert.png
parameter_type_relative_diff.png
top_changed_parameters.png
weight_diff_histogram_global.png
weight_diff_histogram_vit.png
weight_diff_histogram_vlm_backbone.png
weight_diff_histogram_action_expert.png
weight_scatter_global.png
```

Plot requirements:

- Use readable figure sizes.
- Rotate long x-axis labels.
- Save plots at `dpi=200` or higher.
- If a component has no comparable tensors, skip that component plot and log a warning.
- For scatter plots, sample weight pairs rather than plotting all parameters.
- For histograms, sample elementwise differences rather than loading all diffs into memory.
- Include clear titles and y-axis labels.

Optional but useful:

- `top_changed_parameters_by_component_<component>.png`
- separate scatter plots per component
- separate histograms for `relative_diff` per parameter

---

## Interpretation helper

Add a small text or Markdown report:

```text
report.md
```

The report should summarize the results in human-readable terms. It should be generated automatically from the CSV/JSON outputs.

Include:

1. Paths of the two checkpoints
2. Global metrics
3. Component ranking by `relative_diff`
4. Component ranking by lowest `cosine_similarity`
5. Top 20 changed individual parameters
6. Notes about missing keys and shape mismatches
7. A reminder that raw weight distance is most interpretable when both checkpoints share the same architecture, same base checkpoint, same action dimension, and same initial checkpoint before dataset-specific fine-tuning
8. A reminder that behavior-level comparison may require comparing activations, logits, or action outputs on the same validation observations

Example wording:

```text
The action_expert component has the largest relative L2 difference, suggesting that dataset-specific fine-tuning affected the action-generation part of the policy more strongly than the ViT or VLM backbone. This is a weight-space observation only; confirm with action-output comparisons on shared validation observations.
```

The report should avoid overclaiming causality.

---

## Optional extension: activation/output comparison scaffold

The main deliverable is weight comparison. If the codebase structure makes it straightforward, add a placeholder or documented extension point for behavior-level comparison.

Do not overbuild this unless the implementation is simple.

Suggested future metrics:

### ViT

- hidden-state cosine similarity on the same image batch
- patch-token L2 distance
- block-wise activation CKA

### VLM backbone

- text-token hidden-state cosine similarity on the same image/prompt batch
- logits KL divergence
- prediction agreement if logits are available

### Action expert

- denoising vector L2 difference for same observation, prompt, noisy action, and timestep
- action chunk MAE/MSE for same observation and prompt
- action chunk cosine similarity

If added, put it behind a separate script or a clearly disabled-by-default flag. Do not make the weight comparison tool depend on loading the full model for inference.

---

## Error handling and edge cases

Handle these cases explicitly:

1. No common comparable floating tensors: raise a clear error.
2. Some common keys have shape mismatch: write `shape_mismatches.csv`; continue by default.
3. Entire component has no comparable tensors: skip plots for that component and warn.
4. Missing `safetensors` dependency: provide a clear installation message if user tries to load `.safetensors`.
5. Very large tensors: process in chunks; do not concatenate all model weights unless sampling is bounded.
6. NaN or Inf values: count and report them; avoid crashing if possible.
7. Zero-norm tensors: keep metrics finite using epsilon.
8. Different action spaces: action projection shapes may mismatch; record this clearly.

---

## Tests

Add lightweight tests if the repository has a test suite. If not, add a simple smoke-test script or document a command that creates toy checkpoints and runs the comparison.

Test cases should cover:

1. Two tiny synthetic checkpoints with identical keys and shapes.
2. Known numerical differences where relative difference and cosine similarity can be checked approximately.
3. Missing keys in A and B.
4. Shape mismatch handling.
5. Component classification for representative OpenPI-like key names:

```text
paligemma_with_expert.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight
paligemma_with_expert.language_model.layers.0.self_attn.q_proj.weight
paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight
action_in_proj.weight
action_out_proj.weight
time_mlp_in.weight
time_mlp_out.weight
```

6. Output directory contains required CSV and PNG files after a smoke run.

Recommended validation command examples:

```bash
python -m pytest tests/test_compare_pi05_weights.py
python scripts/compare_pi05_weights.py --a /tmp/toy_a.safetensors --b /tmp/toy_b.safetensors --out /tmp/pi05_weight_compare_smoke
```

If `pytest` or `safetensors` is not available in the environment, make tests degrade gracefully or document the dependency.

---

## Implementation constraints

- Use Python.
- Use `torch` for tensor operations.
- Use `pandas` for CSV creation if available; otherwise use Python CSV as fallback only if necessary.
- Use `matplotlib` for plots.
- Keep memory usage bounded.
- Keep changes scoped to analysis tooling and documentation.
- Do not require GPU.
- Do not upload checkpoints or require network access.
- Do not alter model weights.
- Do not import heavy OpenPI model-loading code unless necessary. Weight comparison should work directly from state dicts.

---

## Suggested implementation structure

Inside the script, prefer functions/classes like:

```python
class TensorStore:
    """Lazy-ish checkpoint reader for safetensors and torch checkpoints."""

@dataclass
class RawStats:
    n: int
    sum_a2: float
    sum_b2: float
    sum_diff2: float
    dot: float
    sum_abs_diff: float
    max_abs_diff: float
    sign_flip_count: int


def resolve_weight_file(path: Path) -> Path:
    ...


def load_keys(path: Path) -> set[str]:
    ...


def compute_raw_stats(a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> RawStats:
    ...


def finalize_stats(stats: RawStats, eps: float = 1e-12) -> dict:
    ...


def component_of(key: str) -> str:
    ...


def layer_id_of(key: str, component: str) -> str:
    ...


def param_type_of(key: str) -> str:
    ...


def compare_checkpoints(path_a: Path, path_b: Path, out_dir: Path, args) -> None:
    ...


def write_report(out_dir: Path, summary: dict, dataframes: dict) -> None:
    ...


def main() -> None:
    ...
```

Keep plotting code isolated from metric computation so that tests can exercise metric logic without creating all plots.

---

## Acceptance criteria

The task is complete when all of the following are true:

1. A user can run the CLI on two OpenPI `pi0.5` PyTorch-converted checkpoints.
2. The script compares common floating-point tensors without loading the entire model into GPU memory.
3. The script produces numerical summaries at global, component, layer, parameter-type, and individual-parameter levels.
4. The script generates the required plots.
5. The script writes diagnostic files for missing keys and shape mismatches.
6. The script includes auditable key classification output.
7. The implementation handles large checkpoints through chunking and sampling.
8. The repository includes at least minimal documentation or usage instructions.
9. There is a smoke test or documented smoke-test command using toy checkpoints.
10. The implementation does not modify training, inference, model definitions, checkpoint conversion, or existing checkpoint files.

---

## Suggested first action for Codex

Before editing files:

1. Inspect the repository structure.
2. Find existing script/tool conventions.
3. Inspect representative OpenPI `pi0.5` state dict key names if any checkpoint fixtures or conversion code are available.
4. Choose the least intrusive file location.
5. Implement the utility, tests, and documentation.
6. Run the smoke test and report generated artifacts.

