# compare_pi05_weights 스모크 절차

`tests/` 관례가 있으므로 자동 검증은 `tests/test_compare_pi05_weights.py`에서 수행합니다. 이 문서는 로컬/CI에서 수동 스모크를 수행할 때 참고용 체크리스트입니다.

## 1) 선택 의존성 안내

- `pytest`가 없으면 테스트 실행이 불가합니다.
  - 설치: `pip install pytest`
  - 우회: `python scripts/compare_pi05_weights.py --help`로 CLI 동작만 우선 확인
- `safetensors`가 없으면 `.safetensors` 입력을 로딩할 수 없습니다.
  - 설치: `pip install safetensors`
  - 우회: `.pt`/`.bin` 체크포인트를 사용해 동일한 비교 파이프라인을 검증

> 참고: 본 스크립트는 `torch`가 반드시 필요하며, 테스트 파일은 `torch` 미설치 환경에서 skip 됩니다.

## 2) 스모크 실행

```bash
pytest -q tests/test_compare_pi05_weights.py
```

## 3) 케이스 커버리지 확인

아래 케이스가 테스트 파일에 포함되어야 합니다.

- 동일 구조 비교
- 수치 검증 가능한 차이(평균/최대/부호 반전 등)
- `only_in_a`, `only_in_b`
- shape mismatch
- 대표 OpenPI 키 분류(`vit`, `vlm_backbone`, `action_expert`)

## 4) 필수 산출물 체크

스모크 테스트(`test_full_output_artifacts`)는 아래 산출물 존재를 검증합니다.

- CSV: `component_summary.csv`, `layer_summary.csv`, `parameter_type_summary.csv`, `per_parameter.csv`, `shape_mismatches.csv`, `only_in_a.csv`, `only_in_b.csv`
- PNG: 컴포넌트/레이어/파라미터 타입 요약 그래프, 히스토그램, 산점도
- 보고서: `report.md`, `summary.json`

실패 시 먼저 의존성(`torch`, `matplotlib`) 설치 여부와 출력 디렉토리 권한을 점검하세요.
