#!/usr/bin/env python3
"""
Микросервис генерации изображений по тексту.
Загружает модель при старте и держит её в памяти.
"""
import argparse
import logging
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from services.text2image_service import Text2ImageService
from config import settings

# ---------- Инициализация ----------
app = FastAPI(title="Text2Image Microservice")
logger = logging.getLogger("text2image_service")
logging.basicConfig(level=logging.INFO)

service = None

class GenerateRequest(BaseModel):
    prompt: str
    seed: int = 1234
    # Можно добавить любые параметры, специфичные для модели

# ---------- Эндпоинты ----------
@app.on_event("startup")
async def startup():
    global service
    logger.info(f"Запуск Text2Image микросервиса, профиль {settings.profile}")
    # Создаём локальный сервис с нужным профилем (high/low)
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
    try:
        image = await service.generate(req.prompt, seed=req.seed)
    except Exception as e:
        logger.exception("Ошибка генерации")
        raise HTTPException(status_code=500, detail=str(e))

    # Сохраняем изображение в буфер и отдаём PNG
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

# ---------- Точка входа ----------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")