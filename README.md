# Anomaly Detection API

[![CI](https://github.com/YanissAmz/anomaly-detection-api/actions/workflows/ci.yml/badge.svg)](https://github.com/YanissAmz/anomaly-detection-api/actions)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Visual anomaly detection service using **PatchCore** on MVTec AD, served as a production-ready **FastAPI** endpoint with heatmap overlay visualization and a web demo.

> PatchCore (Roth et al., CVPR 2022) achieves state-of-the-art anomaly detection by building a memory bank of normal patch features and detecting anomalies via nearest-neighbor distance.

---

## Pipeline

```mermaid
flowchart LR
    A[MVTec AD\ndataset] --> B[Feature\nextraction]
    B --> C[Memory bank\ncoreset]
    C --> D[Nearest-neighbor\nscoring]
    D --> E[Anomaly map\nheatmap]
    E --> F[FastAPI\nserving]

    style A fill:#f0f0f0,stroke:#333
    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#fef3c7,stroke:#d97706
    style D fill:#d1fae5,stroke:#059669
    style E fill:#fee2e2,stroke:#dc2626
    style F fill:#ede9fe,stroke:#7c3aed
```

| Stage | What | Key metric |
|---|---|---|
| **Feature extraction** | WideResNet50 layer2+3 patch features | -- |
| **Memory bank** | Coreset subsampling of normal features | Bank size |
| **Scoring** | K-NN distance to memory bank | AUROC |
| **Visualization** | Per-pixel anomaly heatmap overlay | -- |
| **Serving** | FastAPI with image upload endpoint | Latency (ms) |

---

## Quick start

```bash
git clone https://github.com/YanissAmz/anomaly-detection-api.git
cd anomaly-detection-api

# Install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
make test

# Start API server
make serve
```

---

## Project structure

```
src/
  models/             PatchCore implementation
  api/                FastAPI application
  preprocessing/      Image transforms & utilities
  demo/               Web interface
configs/              YAML configs per MVTec category
scripts/              CLI entrypoints (train, evaluate, serve)
tests/                Unit & integration tests
results/              AUROC tables, heatmap samples
docs/                 Technical writeup
```

---

## API

```bash
# Health check
curl http://localhost:8000/health

# Predict anomaly
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.png"
```

**Endpoints:**
| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | API & model status |
| `POST` | `/predict` | Upload image, get anomaly score + heatmap |

---

## Results

> Benchmarks on MVTec AD. Full results in `results/`.

| Category | AUROC (image) | AUROC (pixel) | Latency (ms) |
|---|---|---|---|
| Bottle | -- | -- | -- |
| Cable | -- | -- | -- |
| Capsule | -- | -- | -- |

*Results will be filled after running evaluation on all categories.*

---

## Tech stack

| | |
|---|---|
| **Model** | PatchCore (WideResNet50 backbone) |
| **Dataset** | MVTec AD (15 categories) |
| **Scoring** | K-NN distance to coreset memory bank |
| **Serving** | FastAPI + Uvicorn |
| **CI** | GitHub Actions (ruff + pytest) |

---

## Limitations & future work

- Coreset subsampling is random; greedy coreset (original paper) not yet implemented
- Single-category models; no unified multi-category model
- No comparison with other methods (STFPM, FastFlow, EfficientAD)
- Planned: heatmap overlay endpoint, Gradio demo, multi-method benchmarks

---

## License

MIT
