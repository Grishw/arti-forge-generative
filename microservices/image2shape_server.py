#!/usr/bin/env python3
"""
Микросервис генерации 3D-меша по изображению.
Загружает модель при старте и держит её в памяти.
Требует профиль 'high' (GPU).
"""
import argparse
import logging
import base64
import trimesh
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.image2shape_service import Image2ShapeService
from config import settings

# ---------- Инициализация ----------
app = FastAPI(title="Image2Shape Microservice")
logger = logging.getLogger("image2shape_service")
logging.basicConfig(level=logging.INFO)

service = None

class GenerateRequest(BaseModel):
    image: str   # base64-кодированное PNG
    seed: int = 1234
    # Дополнительные параметры при необходимости

# ---------- Эндпоинты ----------
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

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not service or not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Декодируем изображение
    try:
        img_bytes = base64.b64decode(req.image)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    try:
        mesh = await service.generate(image, seed=req.seed)
    except Exception as e:
        logger.exception("Ошибка генерации 3D")
        raise HTTPException(status_code=500, detail=str(e))

    # Сериализуем меш в GLB и возвращаем base64
    buf = BytesIO()
    mesh.export(buf, file_type='glb')
    mesh_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"mesh_base64": mesh_b64}

# ---------- Точка входа ----------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")