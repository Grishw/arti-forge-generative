import asyncio
import torch
from services.base import BaseImage2ShapeService
from pipelines.Hunyuan.Hunyuan3DShapePipeline import Hunyuan3DShapePipeline
from config import settings

class Image2ShapeService(BaseImage2ShapeService):
    def __init__(self, profile: str):
        if profile != "high":
            raise ValueError("Image2ShapeService требует профиль 'high'")
        self.model = None
        self._lock = asyncio.Lock()

    def _create_model(self):
        return Hunyuan3DShapePipeline(
            shape_model_path=settings.shape_model,
            tex_model_path=settings.tex_model,
            subfolder=settings.subfolder,
            device=settings.device,
            enable_texture=True,
            max_faces=40000
        )

    async def load_model(self):
        async with self._lock:
            if self.model is None:
                self.model = self._create_model()

    async def unload_model(self):
        async with self._lock:
            if self.model:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None

    async def maybe_offload_if_needed(self):
        """Выгружает модель, если свободной видеопамяти < 4 ГБ."""
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free < 4 * 1024**3:
                await self.unload_model()

    async def generate(self, image, **kwargs):
        await self.load_model()
        return await asyncio.to_thread(self.model.generate, image, **kwargs)