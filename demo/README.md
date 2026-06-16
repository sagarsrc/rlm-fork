# RLM Demo

Self-contained demo for the Recursive Language Models paper.

## Layout

- `backend/` — FastAPI server (`fastapi_server.py`) and data loader (`dataloader.py`).
- `frontend/` — `demo.html`, plus `static/` with CSS and JS.
- `data/` — Sample OOLONG datasets under `demo-upload/`.
- `notebooks/` — Standalone Python scripts showing the RLM workflow.
- `docs/screenshots/` — Demo screenshots.

## Run

```bash
python -m demo.backend.fastapi_server
```

Then open http://localhost:3000.
