# OpenPI pi0.5 Weight Difference Analysis

OpenPI `pi0.5` 체크포인트 2개를 **가중치(weight) 공간에서 정량/시각 비교**하는 분석 도구입니다.

동일한 구조(아키텍처)의 모델이 서로 다른 데이터셋으로 학습/파인튜닝되었을 때,
어느 컴포넌트(ViT / VLM Backbone / Action Expert)에서 얼마나 크게 변화했는지 빠르게 파악할 수 있습니다.

---

## 1. 기능 목적

이 도구의 핵심 목적은 다음 질문에 답하는 것입니다.

- "두 OpenPI `pi0.5` 체크포인트의 차이는 **전체적으로** 얼마나 큰가?"
- "차이가 **어느 컴포넌트**(ViT, VLM backbone, action expert)에 집중되는가?"
- "어떤 **레이어/파라미터 타입**(attention, MLP, norm, embedding, action projection)에서 차이가 큰가?"
- "디버깅을 위해 **개별 파라미터 단위**로 가장 많이 변한 항목은 무엇인가?"

> 중요한 점: 이 도구는 "가중치 공간 비교" 도구입니다.  
> 즉, 모델 동작(출력/행동)의 차이를 직접 보장하지 않으며, 필요 시 activation/logit/action 비교를 병행해야 합니다.

---

## 2. 주요 기능 요약

- 체크포인트 입력 형식 지원
  - `.safetensors`
  - `.pt`, `.pth`, `.bin`
  - 파일 경로 또는 디렉토리 경로 입력 모두 지원
- 디렉토리 입력 시 자동 탐색 우선순위
  1. `model.safetensors`
  2. `pytorch_model.bin`
  3. `model.pt`
  4. `checkpoint.pt`
  5. `*.safetensors`, `*.pt`, `*.pth`, `*.bin`
- 공통 키/누락 키/shape mismatch 자동 분리
- float 텐서 + 동일 shape 텐서만 비교
- 대형 텐서 청크(chunk) 처리로 메모리 사용 제어
- 통계 집계 후 최종 메트릭 계산(평균의 평균 방지)
- 컴포넌트/레이어/파라미터 타입/개별 파라미터 레벨 CSV 출력
- 히스토그램/바차트/스캐터 PNG 자동 생성
- `report.md` 자동 생성(해석 보조)

---

## 3. 설치 및 환경

## 필수

- Python 3.10+
- `torch`
- `matplotlib`

## 선택

- `safetensors` (`.safetensors` 로딩 시 필요)
- `pytest` (테스트 실행 시 필요)

예시 설치:

```bash
pip install torch matplotlib safetensors pytest
```

---

## 4. 기본 사용법

```bash
python scripts/compare_pi05_weights.py \
  --a /path/to/checkpoint_a_or_dir \
  --b /path/to/checkpoint_b_or_dir \
  --out /path/to/output_dir
```

실행이 끝나면 콘솔에 요약 카운트가 출력되고,
`--out` 디렉토리에 CSV/JSON/PNG/Markdown 리포트가 생성됩니다.

---

## 5. CLI 옵션 상세

## 필수 인자

- `--a`: 첫 번째 체크포인트(파일 또는 디렉토리)
- `--b`: 두 번째 체크포인트(파일 또는 디렉토리)
- `--out`: 결과 저장 디렉토리

## 주요 선택 인자

- `--component-map-json`: 키 prefix → 컴포넌트 매핑 JSON
- `--top-k`: top changed parameter 플롯 개수 (기본 50)
- `--histogram-sample-size`: 히스토그램 샘플 최대 개수 (기본 1,000,000)
- `--scatter-sample-size`: 스캐터 샘플 최대 개수 (기본 100,000)
- `--chunk-size`: 통계 계산 청크 크기 (기본 10,000,000)
- `--fail-on-shape-mismatch`: shape mismatch 발견 시 즉시 실패
- `--verbose`: 상세 로그 출력

> 참고: `--include-other`, `--exclude-buffers` 옵션은 현재 인터페이스 호환성 용도로 유지되어 있으며, 추후 세밀한 필터링 옵션으로 확장 가능합니다.

---

## 6. 분류 체계(해석의 핵심)

이 도구는 파라미터 키 문자열을 기반으로 아래 분류를 수행합니다.

- component
  - `vit`
  - `vlm_backbone`
  - `action_expert`
  - `other`
- layer_id
  - 예: `vit.block_03`, `vlm.block_12`, `action_expert.block_00`
  - 특수: `vlm.token_embedding`, `vlm.lm_head`, `*.projections` 등
- param_type
  - 예: `attention.q_proj`, `attention.k_proj`, `mlp.up_proj`, `normalization`, `token_embedding`, `action_projection` 등

추론이 완벽히 맞지 않을 수 있으므로,
`key_classification.csv`를 통해 분류 결과를 반드시 점검하는 것을 권장합니다.

---

## 7. 산출물 설명

`--out` 디렉토리에 다음 파일들이 생성됩니다.

## CSV

- `component_summary.csv`
- `layer_summary.csv`
- `parameter_type_summary.csv`
- `per_parameter.csv`
- `key_classification.csv`
- `shape_mismatches.csv`
- `only_in_a.csv`
- `only_in_b.csv`

## JSON

- `summary.json`
  - 입력 경로, 키 카운트, 글로벌/컴포넌트 메트릭, 생성 시각, git commit 정보 포함

## PNG

- `component_relative_diff.png`
- `component_cosine_similarity.png`
- `component_mean_abs_diff.png`
- `layer_relative_diff_vit.png`
- `layer_relative_diff_vlm_backbone.png`
- `layer_relative_diff_action_expert.png`
- `layer_cosine_similarity_vit.png`
- `layer_cosine_similarity_vlm_backbone.png`
- `layer_cosine_similarity_action_expert.png`
- `parameter_type_relative_diff.png`
- `top_changed_parameters.png`
- `weight_diff_histogram_global.png`
- `weight_diff_histogram_vit.png`
- `weight_diff_histogram_vlm_backbone.png`
- `weight_diff_histogram_action_expert.png`
- `weight_scatter_global.png`

## Markdown

- `report.md`
  - 글로벌 지표 요약
  - 컴포넌트 relative diff/cosine ranking
  - Top-20 changed parameters
  - 누락 키 및 shape mismatch 요약
  - 해석 주의사항

---

## 8. 메트릭 정의

각 비교 대상 텐서에 대해 주요 지표를 계산합니다.

- `norm_a`, `norm_b`: 각 텐서 L2 norm
- `diff_norm`: `||b-a||_2`
- `relative_diff`: `||b-a||_2 / max(||a||_2, eps)`
- `cosine_similarity`: `dot(a,b) / max(||a||_2*||b||_2, eps)`
- `mean_abs_diff`: `mean(abs(b-a))`
- `max_abs_diff`: `max(abs(b-a))`
- `sign_flip_ratio`: 부호 반전 비율

추가로 NaN/Inf 카운트도 집계합니다.

---

## 9. 커스텀 컴포넌트 매핑 예시

일부 변환 파이프라인에서 키 prefix가 다를 수 있으므로,
`--component-map-json`으로 분류를 덮어쓸 수 있습니다.

예시 `component_map.json`:

```json
{
  "my_openpi.vision": "vit",
  "my_openpi.llm": "vlm_backbone",
  "my_openpi.expert": "action_expert"
}
```

실행:

```bash
python scripts/compare_pi05_weights.py \
  --a /path/a \
  --b /path/b \
  --out /tmp/pi05_compare \
  --component-map-json /path/component_map.json
```

---

## 10. 테스트/검증

기본 테스트 실행:

```bash
pytest -q tests/test_compare_pi05_weights.py
```

스모크 절차 문서:

- `docs/compare_pi05_weights.md`

테스트는 대략 다음을 검증합니다.

- 파일 탐색 우선순위
- 키 정규화/래퍼 해제
- 누락 키/shape mismatch 처리
- 청크 통계/최종 메트릭 계산
- 분류 로직(component/layer/param_type)
- 전체 산출물(CSV/PNG/report) 생성

---

## 11. 에러/주의사항

- 공통 float 텐서가 없으면 비교 의미가 떨어질 수 있습니다.
- shape mismatch가 많은 경우 action dimension/헤드 구조 차이를 먼저 확인하세요.
- `.safetensors` 사용 시 `safetensors` 패키지가 없으면 로딩 실패합니다.
- 큰 모델은 `--chunk-size`, 샘플 사이즈 옵션을 조정해 메모리와 속도를 균형화하세요.
- 이 도구는 학습/추론 경로를 건드리지 않는 분석 유틸리티입니다(가중치 파일 변경 없음).

---

## 12. 빠른 시작 예시

```bash
python scripts/compare_pi05_weights.py \
  --a /data/ckpt/openpi_pi05_datasetA \
  --b /data/ckpt/openpi_pi05_datasetB \
  --out /data/analysis/pi05_weight_diff \
  --top-k 100 \
  --chunk-size 5000000 \
  --verbose
```

결과 확인 우선순위 추천:

1. `summary.json` / `report.md`
2. `component_summary.csv`
3. `layer_summary.csv`
4. `top_changed_parameters.png`
5. `key_classification.csv` (분류 정확도 감사)

