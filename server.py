import asyncio
import uuid
import os
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

from config import settings
from services import create_text2image_service, create_image2shape_service
from services.text2shape_service import Text2ShapeService

# ---------- Инициализация ----------
SAVE_DIR = "generated_models"
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("server")

app = FastAPI(title="ARTi Forge – Generative Services", version="1.0")

# Сервисы
text2img_svc = None
img2shape_svc = None
text2shape_svc = None

# Хранилища задач
tasks_2d = {}
tasks_3d = {}

# ---------- Модели данных ----------
class PromptRequest(BaseModel):
    prompt: str
    seed: int = 1234

# ---------- Фоновые задачи ----------
async def process_2d(task_id: str, req: PromptRequest):
    try:
        tasks_2d[task_id]["status"] = "generating"
        image = await text2img_svc.generate(req.prompt)
        path = os.path.join(SAVE_DIR, f"{task_id}.png")
        image.save(path)
        tasks_2d[task_id]["status"] = "completed"
        tasks_2d[task_id]["path"] = path
    except Exception as e:
        logger.exception("2D generation failed")
        tasks_2d[task_id]["status"] = "failed"
        tasks_2d[task_id]["error"] = str(e)

async def process_3d(task_id: str, req: PromptRequest):
    try:
        tasks_3d[task_id]["status"] = "generating"
        mesh = await text2shape_svc.generate(req.prompt, seed=req.seed)
        path = os.path.join(SAVE_DIR, f"{task_id}.glb")
        mesh.export(path)
        tasks_3d[task_id]["status"] = "completed"
        tasks_3d[task_id]["path"] = path
    except Exception as e:
        logger.exception("3D generation failed")
        tasks_3d[task_id]["status"] = "failed"
        tasks_3d[task_id]["error"] = str(e)

# ---------- События ----------
@app.on_event("startup")
async def startup():
    global text2img_svc, img2shape_svc, text2shape_svc

    text2img_svc = create_text2image_service()
    t2i_type = "remote" if settings.text2image_url else "local"
    logger.info(f"Text2Image service: {t2i_type}")

    img2shape_svc = create_image2shape_service()
    if img2shape_svc is not None:
        i2s_type = "remote" if settings.image2shape_url else "local"
        logger.info(f"Image2Shape service: {i2s_type}")
        # Предзагружаем локальную модель, если она используется
        if not settings.image2shape_url:
            await img2shape_svc.load_model()
        text2shape_svc = Text2ShapeService(text2img_svc, img2shape_svc)
    else:
        logger.warning("3D generation NOT available (low profile, no remote URL)")

# ---------- Эндпоинты ----------
@app.post("/generate/2d")
async def generate_2d(req: PromptRequest):
    task_id = str(uuid.uuid4())
    tasks_2d[task_id] = {
        "status": "queued",
        "created_at": datetime.now(),
        "prompt": req.prompt
    }
    asyncio.create_task(process_2d(task_id, req))
    return {"task_id": task_id}

@app.post("/generate/3d")
async def generate_3d(req: PromptRequest):
    if text2shape_svc is None:
        raise HTTPException(
            status_code=400,
            detail="3D generation is not available on this device. "
                   "Set IMAGE2SHAPE_SERVICE_URL or run on a high-end GPU."
        )
    task_id = str(uuid.uuid4())
    tasks_3d[task_id] = {
        "status": "queued",
        "created_at": datetime.now(),
        "prompt": req.prompt
    }
    asyncio.create_task(process_3d(task_id, req))
    return {"task_id": task_id}

@app.get("/status/2d/{task_id}")
async def status_2d(task_id: str):
    task = tasks_2d.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    resp = {"status": task["status"]}
    if task["status"] == "completed":
        resp["download_url"] = f"/download/2d/{task_id}"
    elif task["status"] == "failed":
        resp["error"] = task.get("error")
    return resp

@app.get("/status/3d/{task_id}")
async def status_3d(task_id: str):
    task = tasks_3d.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    resp = {"status": task["status"]}
    if task["status"] == "completed":
        resp["download_url"] = f"/download/3d/{task_id}"
    elif task["status"] == "failed":
        resp["error"] = task.get("error")
    return resp

@app.get("/download/2d/{task_id}")
async def download_2d(task_id: str):
    task = tasks_2d.get(task_id)
    if not task or task["status"] != "completed":
        raise HTTPException(404, "Not ready")
    return FileResponse(task["path"], media_type="image/png")

@app.get("/download/3d/{task_id}")
async def download_3d(task_id: str):
    task = tasks_3d.get(task_id)
    if not task or task["status"] != "completed":
        raise HTTPException(404, "Not ready")
    return FileResponse(task["path"], media_type="application/octet-stream")

# ---------- Веб-монитор ----------
@app.get("/monitor", response_class=HTMLResponse)
async def monitor():
    html = """
    <html>
    <head><title>ARTi Forge Monitor</title>
    <style>
        body { font-family: sans-serif; margin: 2rem; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        .completed { background-color: #e6ffe6; }
        .failed { background-color: #ffe6e6; }
        .generating { background-color: #fffbe6; }
        .queued { background-color: #e6f3ff; }
    </style>
    </head>
    <body>
        <h1>🎨 ARTi Forge – Статус генерации</h1>
        <table id="tasks"></table>
        <script>
            async function refresh() {
                const resp2d = await fetch('/list/2d');
                const resp3d = await fetch('/list/3d');
                const tasks2d = await resp2d.json();
                const tasks3d = await resp3d.json();
                const all = [...tasks2d.map(t => ({...t, type:'2D'})), ...tasks3d.map(t => ({...t, type:'3D'}))];
                const tbody = document.querySelector('#tasks');
                tbody.innerHTML = '<tr><th>ID</th><th>Тип</th><th>Промпт</th><th>Статус</th><th>Скачать</th></tr>';
                for (const t of all) {
                    const tr = document.createElement('tr');
                    tr.className = t.status;
                    tr.innerHTML = `<td>${t.task_id.slice(0,8)}</td><td>${t.type}</td><td>${t.prompt||''}</td><td>${t.status}</td>`;
                    const td = tr.insertCell(4);
                    if (t.status === 'completed') {
                        td.innerHTML = `<a href="/download/${t.type=='2D'?'2d':'3d'}/${t.task_id}">Скачать</a>`;
                    }
                    tbody.appendChild(tr);
                }
            }
            refresh();
            setInterval(refresh, 3000);
        </script>
    </body>
    </html>
    """
    return html

@app.get("/list/2d")
async def list_2d():
    return [
        {"task_id": tid, **info}
        for tid, info in tasks_2d.items()
    ]

@app.get("/list/3d")
async def list_3d():
    return [
        {"task_id": tid, **info}
        for tid, info in tasks_3d.items()
    ]

# ---------- Точка входа ----------
if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)