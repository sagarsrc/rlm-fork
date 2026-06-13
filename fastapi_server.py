#!/usr/bin/env python3
"""FastAPI server for RLM visualizer, demo, and live RLM jobs."""

from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
VISUALIZER_DIR = ROOT / "rlm" / "visualizer"
VISUALIZER_OUT = VISUALIZER_DIR / "out"
LOGS_DIR = ROOT / "logs"
PUBLIC_LOGS = VISUALIZER_DIR / "public" / "logs"

# In-memory job store (single uvicorn worker is fine for demo).
_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    sync_logs()
    yield


app = FastAPI(title="RLM Visualizer Server", lifespan=lifespan)


def sync_logs() -> None:
    """Ensure public logs exist and mirror source trajectory files."""
    PUBLIC_LOGS.mkdir(parents=True, exist_ok=True)
    if not LOGS_DIR.exists():
        return
    for src in LOGS_DIR.glob("*.jsonl"):
        dest = PUBLIC_LOGS / src.name
        if dest.exists() and dest.stat().st_mtime_ns >= src.stat().st_mtime_ns:
            continue
        shutil.copy2(src, dest)


@app.get("/api/logs")
async def list_logs() -> JSONResponse:
    """List available trajectory files."""
    sync_logs()
    files = sorted(file.name for file in PUBLIC_LOGS.glob("*.jsonl") if file.is_file())
    return JSONResponse({"files": files})


@app.get("/logs/{filename}")
async def get_log(filename: str) -> FileResponse:
    """Serve trajectory file contents."""
    sync_logs()
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=404, detail="File not found")
    path = PUBLIC_LOGS / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/api/oolong-data")
async def oolong_data() -> JSONResponse:
    """Return OOLONG trec_coarse data for demo."""
    import sys

    sys.path.insert(0, str(ROOT))
    from dataloader import get_oolong_trec_coarse

    data = get_oolong_trec_coarse(32768)
    num_questions = len([l for l in data["context"].split("\n") if l.startswith("Date:")])
    return JSONResponse(
        {
            "context": data["context"],
            "question": data["question"],
            "answer": data["answer"],
            "num_questions": num_questions,
        }
    )


@app.post("/api/baseline")
async def baseline_llm(request: dict) -> JSONResponse:
    """Run direct LLM call (Algorithm 2) for comparison."""
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    context = request.get("context", "")
    query = request.get("query", "")
    prompt = context + "\n\n" + query if query else context

    client = OpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.ai/v1",
    )
    try:
        response = client.chat.completions.create(
            model="kimi-k2.6",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return JSONResponse(
            {
                "response": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            }
        )
    except Exception as e:
        return JSONResponse({"response": f"ERROR: {e}", "finish_reason": "error"})


@app.get("/demo")
async def demo_page() -> FileResponse:
    """Serve the live demo page."""
    demo_html = ROOT / "demo.html"
    if demo_html.exists():
        return FileResponse(demo_html)
    raise HTTPException(status_code=404, detail="Demo page not found")


# ── Background RLM jobs ──────────────────────────────────────────────────────


def _run_rlm_job(job_id: str, prompt: str, max_iters: int, log_file: str) -> None:
    """Run RLM synchronously and update the in-memory job store."""
    import io
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    sys.path.insert(0, str(ROOT / "rlm"))
    from rlm import RLM
    from rlm.logger import RLMLogger

    logger = RLMLogger(log_dir=str(LOGS_DIR))
    # Predictable filename so frontend can poll it immediately.
    if logger.log_file_path:
        # Move the auto-created file out of the way and use our chosen name.
        target = LOGS_DIR / log_file
        if Path(logger.log_file_path).exists():
            Path(logger.log_file_path).unlink()
        logger.log_file_path = str(target)

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        rlm = RLM(
            backend="moonshot",
            backend_kwargs={"model_name": "kimi-k2.6"},
            environment="local",
            max_iterations=max_iters,
            logger=logger,
            verbose=True,
        )
        result = rlm.completion(prompt)
        _jobs[job_id].update(
            {
                "status": "done",
                "response": result.response,
                "execution_time": result.execution_time,
                "verbose_output": captured.getvalue(),
                "error": None,
            }
        )
    except Exception as e:
        _jobs[job_id].update(
            {
                "status": "error",
                "response": f"ERROR: {e}",
                "execution_time": 0,
                "verbose_output": captured.getvalue(),
                "error": str(e),
            }
        )
    finally:
        sys.stdout = old_stdout
        sync_logs()


@app.post("/api/run")
async def run_rlm(request: dict, background_tasks: BackgroundTasks) -> JSONResponse:
    """Start an RLM job in the background; return job_id and log_file to poll."""
    import uuid
    from datetime import datetime

    context = request.get("context", "")
    query = request.get("query", "")
    max_iters = request.get("max_iterations", 6)
    prompt = context + "\n\n" + query if query else context

    job_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"rlm_{timestamp}_{job_id}.jsonl"

    _jobs[job_id] = {
        "status": "running",
        "log_file": log_file,
        "response": None,
        "execution_time": None,
        "error": None,
    }

    background_tasks.add_task(_run_rlm_job, job_id, prompt, max_iters, log_file)

    return JSONResponse({"job_id": job_id, "log_file": log_file, "status": "running"})


@app.get("/api/job/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    """Poll job status and result."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


# ── Static visualizer ────────────────────────────────────────────────────────

if VISUALIZER_OUT.exists():
    app.mount("/", StaticFiles(directory=str(VISUALIZER_OUT), html=True), name="visualizer")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
