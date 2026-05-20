import asyncio
import torch
from services.base import BaseText2ImageService
from pipelines.SD.SDTurboText2Image import SDTurboText2Image
from pipelines.PixArt.PixArtSigmaText2Image import PixArtSigmaText2Image
from config import settings

class Text2ImageService(BaseText2ImageService):
    def __init__(self, profile: str):
        self.profile = profile
        self.model = None
        self._lock = asyncio.Lock()

    def _create_model(self):
        if self.profile == "high":
            return PixArtSigmaText2Image(
                model_path=settings.text2img_model,
                device=settings.device,
                offload=False
            )
        else:
            return SDTurboText2Image(device=settings.device)

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

    async def generate(self, prompt: str, **kwargs):
        await self.load_model()
        try:
            # Выполняем генерацию в отдельном потоке, чтобы не блокировать event loop
            return await asyncio.to_thread(self.model.generate, prompt, **kwargs)
        finally:
            if self.profile == "high":
                # В высоком профиле сразу выгружаем, чтобы освободить VRAM для 3D
                await self.unload_model()