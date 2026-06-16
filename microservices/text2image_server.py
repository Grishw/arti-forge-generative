#!/usr/bin/env python3
"""
Микросервис генерации изображений по тексту.
Загружает модель при старте и держит её в памяти.

Поддерживает:
- синхронную генерацию (возвращает PNG сразу) – /generate
- асинхронную генерацию (возвращает task_id) – /generate_async
- просмотр списка готовых изображений с превью – /images
- скачивание изображения – /download/{image_id}
- просмотр превью – /preview/{image_id}
"""
import argparse
import logging
import time
import uuid
from io import BytesIO
from collections import deque
from datetime import datetime
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response, HTMLResponse, JSONResponse
from pydantic import BaseModel

from services.text2image_service import Text2ImageService
from config import settings

# ----------------------------------------------------------------------
# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text2image_service")

# ----------------------------------------------------------------------
# Инициализация FastAPI
app = FastAPI(title="Text2Image Microservice")
service = None

# История синхронных запросов (последние 30)
history = deque(maxlen=30)

# Асинхронные задачи (статус генерации)
TASKS: dict[str, dict] = {}          # task_id -> {status, image_path, error, preview_path}

# Хранилище готовых изображений (успешно завершённые)
IMAGES: dict[str, dict] = {}         # image_id -> {id, name, prompt, image_path, preview_path, created_at}

# Директория для сохранения изображений и превью
TEMP_DIR = Path("../generated_images")
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# Вспомогательная функция сохранения превью
def save_preview(image: Image.Image, output_path: Path, size=(256, 256)) -> str:
    """Сохраняет уменьшенную копию изображения как превью."""
    preview = image.copy()
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    preview.save(output_path, "PNG")
    return str(output_path)

# ----------------------------------------------------------------------
# Pydantic модели
class GenerateRequest(BaseModel):
    prompt: str
    seed: int = 1234

class GenerateAsyncRequest(BaseModel):
    prompt: str
    seed: int = 1234
    # name: str = None  # опциональное имя для модели (можно добавить позже)

# ----------------------------------------------------------------------
# Эндпоинты
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

# ----------------------------------------------------------------------
# Синхронная генерация (возвращает PNG сразу, не сохраняет в IMAGES)
@app.post("/generate")
async def generate(req: GenerateRequest):
    if not service or not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    task_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    try:
        image = await service.generate(req.prompt, seed=req.seed)
        duration = round(time.time() - start_time, 2)
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
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": f"attachment; filename=image_{task_id}.png",
            "X-Task-ID": task_id,
            "X-Duration-Sec": str(duration)
        }
    )

# ----------------------------------------------------------------------
# Асинхронная генерация (возвращает task_id, сохраняет результат)
@app.post("/generate_async")
async def generate_async(req: GenerateAsyncRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "pending",
        "image_path": None,
        "preview_path": None,
        "error": None
    }

    # Запускаем генерацию в фоне
    background_tasks.add_task(run_generation, task_id, req.prompt, req.seed)

    return {"task_id": task_id}

async def run_generation(task_id: str, prompt: str, seed: int):
    """Фоновая задача: генерация изображения, сохранение PNG и превью."""
    try:
        # Генерация изображения
        image = await service.generate(prompt, seed=seed)

        # Сохраняем полное изображение
        image_path = TEMP_DIR / f"{task_id}.png"
        image.save(image_path, "PNG")

        # Сохраняем превью (уменьшенная копия)
        preview_path = TEMP_DIR / f"{task_id}_preview.png"
        save_preview(image, preview_path)

        # Обновляем статус задачи
        TASKS[task_id]["status"] = "succeeded"
        TASKS[task_id]["image_path"] = str(image_path)
        TASKS[task_id]["preview_path"] = str(preview_path)

        # Добавляем изображение в общий список готовых
        IMAGES[task_id] = {
            "id": task_id,
            "name": prompt[:50] + ("..." if len(prompt) > 50 else ""),  # короткое имя на основе prompt
            "prompt": prompt,
            "image_path": str(image_path),
            "preview_path": str(preview_path),
            "created_at": datetime.now().isoformat(timespec='seconds')
        }
        logger.info(f"Изображение {task_id} успешно создано и добавлено в список")

    except Exception as e:
        logger.exception(f"Ошибка генерации изображения {task_id}")
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)

# ----------------------------------------------------------------------
# Статус асинхронной задачи
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    if task["status"] == "succeeded":
        return {
            "status": "succeeded",
            "image_urls": {
                "download": f"/download/{task_id}",
                "preview": f"/preview/{task_id}"
            }
        }
    elif task["status"] == "failed":
        return {"status": "failed", "error": task["error"]}
    else:
        return {"status": "pending"}

# ----------------------------------------------------------------------
# Скачивание готового изображения (PNG)
@app.get("/download/{image_id}")
async def download_image(image_id: str):
    # Сначала ищем среди готовых изображений
    if image_id in IMAGES:
        image_path = IMAGES[image_id]["image_path"]
    # Если ещё не добавлена в IMAGES, но задача завершена – тоже можно отдать
    elif image_id in TASKS and TASKS[image_id]["status"] == "succeeded":
        image_path = TASKS[image_id]["image_path"]
    else:
        raise HTTPException(404, "Image not ready or not found")

    if not Path(image_path).exists():
        raise HTTPException(404, "Image file missing")
    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"image_{image_id}.png"
    )

# ----------------------------------------------------------------------
# Превью изображения (PNG)
@app.get("/preview/{image_id}")
async def get_preview(image_id: str):
    if image_id in IMAGES:
        preview_path = IMAGES[image_id]["preview_path"]
    elif image_id in TASKS and TASKS[image_id]["status"] == "succeeded":
        preview_path = TASKS[image_id]["preview_path"]
    else:
        raise HTTPException(404, "Preview not available")

    if not Path(preview_path).exists():
        raise HTTPException(404, "Preview file missing")
    return FileResponse(
        path=preview_path,
        media_type="image/png",
        filename=f"preview_{image_id}.png"
    )

# ----------------------------------------------------------------------
# Список всех доступных изображений (с превью и именами)
@app.get("/images")
async def list_images():
    """
    Возвращает список готовых изображений.
    Каждая запись содержит:
    - id (идентификатор)
    - name (короткое имя на основе prompt)
    - prompt (полный текст запроса)
    - preview_url (ссылка на превью)
    - download_url (ссылка на скачивание PNG)
    - created_at (дата создания)
    """
    items = []
    for img_id, info in IMAGES.items():
        items.append({
            "id": img_id,
            "name": info["name"],
            "prompt": info["prompt"],
            "preview_url": f"/preview/{img_id}",
            "download_url": f"/download/{img_id}",
            "created_at": info["created_at"]
        })
    # Сортируем по дате (новые сверху)
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items

# ----------------------------------------------------------------------
# История синхронных запросов (для обратной совместимости)
@app.get("/list")
async def list_history():
    return list(history)

# ----------------------------------------------------------------------
# Веб-монитор
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
    <h2>Synchronous requests</h2>
    <table id="tasks-table">
        <thead><tr><th>Task ID</th><th>Prompt</th><th>Status</th><th>Time</th><th>Duration (s)</th></tr></thead>
        <tbody></tbody>
    </table>
    <h2>Async generated images</h2>
    <div id="images-list"></div>
    <script>
        async function refreshHistory() {
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
        async function refreshImages() {
            const res = await fetch('/images');
            const images = await res.json();
            const container = document.getElementById('images-list');
            container.innerHTML = '';
            for (const img of images) {
                const div = document.createElement('div');
                div.style.border = '1px solid #ccc';
                div.style.margin = '10px';
                div.style.padding = '10px';
                div.style.display = 'inline-block';
                div.style.width = '200px';
                div.innerHTML = `
                    <strong>${img.name}</strong><br>
                    <small>${img.created_at}</small><br>
                    <a href="${img.preview_url}" target="_blank">
                        <img src="${img.preview_url}" style="max-width:100%; max-height:150px;" alt="preview">
                    </a><br>
                    <a href="${img.download_url}">Download PNG</a>
                `;
                container.appendChild(div);
            }
        }
        refreshHistory();
        refreshImages();
        setInterval(() => {
            refreshHistory();
            refreshImages();
        }, 5000);
    </script>
</body>
</html>
"""

@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    return HTML_MONITOR

# ----------------------------------------------------------------------
# Точка входа
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")