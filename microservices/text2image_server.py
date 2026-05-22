#!/usr/bin/env python3
"""
Микросервис генерации изображений по тексту.
Загружает модель при старте и держит её в памяти.
"""
import argparse
import logging
import time
from io import BytesIO
from collections import deque
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse, HTMLResponse
from pydantic import BaseModel

from services.text2image_service import Text2ImageService
from config import settings

# ---------- Инициализация ----------
app = FastAPI(title="Text2Image Microservice")
logger = logging.getLogger("text2image_service")
logging.basicConfig(level=logging.INFO)

service = None

# История запросов (последние 30)
history = deque(maxlen=30)

class GenerateRequest(BaseModel):
    prompt: str
    seed: int = 1234

# ---------- Эндпоинты ----------
@app.on_event("startup")
async def startup():
    global service
    logger.info(f"Запуск Text2Image микросервиса, профиль {settings.profile}")
    service = Text2ImageService(settings.profile)
    await service.load_model()
    logger.info("Модель загружена и готова к работе")

@app.get("/health")
async def health():
    if service and service.is_loaded():
        return {"status": "ready"}
    return JSONResponse({"status": "not loaded"}, status_code=503)

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not service or not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    task_id = str(time.time()).replace('.', '')[-8:]  # простой идентификатор
    start_time = time.time()
    try:
        image = await service.generate(req.prompt, seed=req.seed)
        duration = round(time.time() - start_time, 2)
        # Сохраняем в историю
        history.appendleft({
            "task_id": task_id,
            "prompt": req.prompt[:80] + ("..." if len(req.prompt) > 80 else ""),
            "status": "completed",
            "created_at": datetime.now().isoformat(timespec='seconds'),
            "duration": duration
        })
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        history.appendleft({
            "task_id": task_id,
            "prompt": req.prompt[:80] + ("..." if len(req.prompt) > 80 else ""),
            "status": "failed",
            "created_at": datetime.now().isoformat(timespec='seconds'),
            "duration": duration,
            "error": str(e)
        })
        logger.exception("Ошибка генерации")
        raise HTTPException(status_code=500, detail=str(e))

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/list")
async def list_history():
    return list(history)

# ---------- Веб-монитор ----------
HTML_MONITOR = """
<!DOCTYPE html>
<html>
<head>
    <title>Text2Image Monitor</title>
    <style>
        body { font-family: monospace; margin: 2rem; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .completed { background-color: #d4edda; }
        .failed { background-color: #f8d7da; }
    </style>
</head>
<body>
    <h1>🖼️ Text2Image Service Monitor</h1>
    <table id="tasks-table">
        <thead><tr><th>Task ID</th><th>Prompt</th><th>Status</th><th>Time</th><th>Duration (s)</th></tr></thead>
        <tbody></tbody>
    </table>
    <script>
        async function refresh() {
            const res = await fetch('/list');
            const tasks = await res.json();
            const tbody = document.querySelector('#tasks-table tbody');
            tbody.innerHTML = '';
            for (const t of tasks) {
                const row = tbody.insertRow();
                row.className = t.status;
                row.insertCell(0).textContent = t.task_id;
                row.insertCell(1).textContent = t.prompt || '';
                row.insertCell(2).textContent = t.status;
                row.insertCell(3).textContent = t.created_at || '';
                row.insertCell(4).textContent = t.duration !== undefined ? t.duration : '';
            }
        }
        refresh();
        setInterval(refresh, 3000);
    </script>
</body>
</html>
"""

@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return HTML_MONITOR

# ---------- Точка входа ----------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")