#!/usr/bin/env python3
"""
Микросервис генерации 3D-меша по изображению.
Загружает модель при старте и держит её в памяти.
Требует профиль 'high' (GPU).


python -m microservices.image2shape_server

Поддерживает:
- синхронную генерацию (возвращает GLB сразу) – /generate
- асинхронную генерацию (возвращает task_id) – /generate_async
- просмотр списка готовых моделей с превью – /models
- скачивание модели – /download/{model_id}
- просмотр превью – /preview/{model_id}
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
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, Response, HTMLResponse, JSONResponse
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from services.image2shape_service import Image2ShapeService
from config import settings
import sys

# ----------------------------------------------------------------------
# Настройка логгирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("image2shape_service")

# ----------------------------------------------------------------------
# Инициализация FastAPI
app = FastAPI(title="Image2Shape Microservice")
service = None

# История синхронных запросов (последние 30)
history = deque(maxlen=30)

# Асинхронные задачи (статус генерации)
TASKS: dict[str, dict] = {}          # task_id -> {status, glb_path, error, preview_path}

# Хранилище готовых моделей (успешно завершённые)
MODELS: dict[str, dict] = {}         # model_id -> {name, glb_path, preview_path, created_at}

# Директория для сохранения моделей и превью
TEMP_DIR = Path("../generated_models")
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# Вспомогательная функция сохранения превью (исходное изображение)
def save_preview(image: Image.Image, output_path: Path, size=(256, 256)) -> str:
    """Сохраняет уменьшенную копию изображения как превью."""
    preview = image.copy()
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    preview.save(output_path, "PNG")
    return str(output_path)

# ----------------------------------------------------------------------
# Эндпоинты
@app.on_event("startup")
async def startup():
    global service
    if settings.profile != "high":
        raise RuntimeError("Image2Shape микросервис требует профиль 'high'")
    logger.info("Запуск Image2Shape микросервиса...")
    service = Image2ShapeService(settings.profile)
    await service.load_model()
    logger.info("Модель загружена и готова")

@app.get("/health")
async def health():
    if service and service.is_loaded():
        return {"status": "ready"}
    return JSONResponse({"status": "not loaded"}, status_code=503)

# ----------------------------------------------------------------------
# Синхронная генерация (старый метод, возвращает GLB сразу)
@app.post("/generate")
async def generate(
    file: UploadFile = File(..., description="PNG/JPEG изображение"),
    seed: int = Form(1234, description="Seed для генерации")
):
    if not service or not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, detail="Only JPEG/PNG images are supported")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img_size = f"{image.width}x{image.height}"
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid image file: {e}")

    task_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    try:
        mesh = await service.generate(image, seed=seed)
        duration = round(time.time() - start_time, 2)
        history.appendleft({
            "task_id": task_id,
            "image": img_size,
            "status": "completed",
            "created_at": datetime.now().isoformat(timespec='seconds'),
            "duration": duration
        })
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        history.appendleft({
            "task_id": task_id,
            "image": img_size,
            "status": "failed",
            "created_at": datetime.now().isoformat(timespec='seconds'),
            "duration": duration,
            "error": str(e)
        })
        logger.exception("Ошибка генерации 3D")
        raise HTTPException(status_code=500, detail=str(e))

    buf = BytesIO()
    mesh.export(buf, file_type='glb')
    glb_data = buf.getvalue()

    return Response(
        content=glb_data,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f"attachment; filename=model_{task_id}.glb",
            "X-Task-ID": task_id,
            "X-Duration-Sec": str(duration)
        }
    )

# ----------------------------------------------------------------------
# Асинхронная генерация (возвращает task_id)
@app.post("/generate_async")
async def generate_async(
    file: UploadFile = File(...),
    seed: int = Form(1234),
    background_tasks: BackgroundTasks = None
):
    # Создаём задачу со статусом "pending"
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "pending",
        "glb_path": None,
        "preview_path": None,
        "error": None
    }

    # Сохраняем оригинальное имя файла для отображения в списке моделей
    original_name = file.filename or f"model_{task_id}"

    # Запускаем генерацию в фоне
    background_tasks.add_task(run_generation, task_id, file, seed, original_name)

    return {"task_id": task_id}

async def run_generation(task_id: str, file: UploadFile, seed: int, original_name: str):
    """Фоновая задача: генерация меша, сохранение GLB и превью."""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Генерация 3D меша
        mesh = await service.generate(image, seed=seed)

        # Сохраняем GLB
        glb_path = TEMP_DIR / f"{task_id}.glb"
        mesh.export(glb_path, file_type='glb')

        # Сохраняем превью (уменьшенная копия исходного изображения)
        preview_path = TEMP_DIR / f"{task_id}.png"
        save_preview(image, preview_path)

        # Обновляем статус задачи
        TASKS[task_id]["status"] = "succeeded"
        TASKS[task_id]["glb_path"] = str(glb_path)
        TASKS[task_id]["preview_path"] = str(preview_path)

        # Добавляем модель в общий список готовых моделей
        MODELS[task_id] = {
            "id": task_id,
            "name": original_name,
            "glb_path": str(glb_path),
            "preview_path": str(preview_path),
            "created_at": datetime.now().isoformat(timespec='seconds')
        }
        logger.info(f"Модель {task_id} успешно создана и добавлена в список")

    except Exception as e:
        logger.exception(f"Ошибка генерации модели {task_id}")
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
            "model_urls": {
                "glb": f"/download/{task_id}",
                "preview": f"/preview/{task_id}"
            }
        }
    elif task["status"] == "failed":
        return {"status": "failed", "error": task["error"]}
    else:
        return {"status": "pending"}

# ----------------------------------------------------------------------
# Скачивание готовой модели (GLB)
@app.get("/download/{model_id}")
async def download_model(model_id: str):
    # Сначала ищем среди готовых моделей
    if model_id in MODELS:
        glb_path = MODELS[model_id]["glb_path"]
    # Если ещё не добавлена в MODELS, но задача завершена – тоже можно отдать
    elif model_id in TASKS and TASKS[model_id]["status"] == "succeeded":
        glb_path = TASKS[model_id]["glb_path"]
    else:
        raise HTTPException(404, "Model not ready or not found")

    if not Path(glb_path).exists():
        raise HTTPException(404, "Model file missing")
    return FileResponse(
        path=glb_path,
        media_type="model/gltf-binary",
        filename=f"model_{model_id}.glb"
    )

# ----------------------------------------------------------------------
# Превью модели (PNG)
@app.get("/preview/{model_id}")
async def get_preview(model_id: str):
    if model_id in MODELS:
        preview_path = MODELS[model_id]["preview_path"]
    elif model_id in TASKS and TASKS[model_id]["status"] == "succeeded":
        preview_path = TASKS[model_id]["preview_path"]
    else:
        raise HTTPException(404, "Preview not available")

    if not Path(preview_path).exists():
        raise HTTPException(404, "Preview file missing")
    return FileResponse(
        path=preview_path,
        media_type="image/png",
        filename=f"preview_{model_id}.png"
    )

# ----------------------------------------------------------------------
# Список всех доступных моделей (с превью и именами)
@app.get("/models")
async def list_models():
    """
    Возвращает список готовых моделей.
    Каждая запись содержит:
    - id (идентификатор)
    - name (имя модели, взятое из исходного файла)
    - preview_url (ссылка на превью)
    - model_url (ссылка на скачивание GLB)
    - created_at (дата создания)
    """
    items = []
    for mid, info in MODELS.items():
        items.append({
            "id": mid,
            "name": info["name"],
            "preview_url": f"/preview/{mid}",
            "model_url": f"/download/{mid}",
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
# Веб-монитор (без изменений)
HTML_MONITOR = """
<!DOCTYPE html>
<html>
<head>
    <title>Image2Shape Monitor</title>
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
    <h1>🧊 Image2Shape Service Monitor</h1>
    <table id="tasks-table">
        <thead><tr><th>Task ID</th><th>Image</th><th>Status</th><th>Time</th><th>Duration (s)</th></tr></thead>
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
                row.insertCell(1).textContent = t.image || '';
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

# ----------------------------------------------------------------------
# Точка входа
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")