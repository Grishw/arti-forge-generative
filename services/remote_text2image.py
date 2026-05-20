import aiohttp
from PIL import Image
from io import BytesIO
from services.base import BaseText2ImageService

class RemoteText2ImageService(BaseText2ImageService):
    def __init__(self, url: str):
        self.url = url.rstrip('/')

    async def load_model(self):
        pass  # удалённый сервис управляет своей моделью

    async def unload_model(self):
        pass

    def is_loaded(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> Image.Image:
        payload = {"prompt": prompt, **kwargs}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/generate", json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Remote text2image error {resp.status}: {error_text}")
                img_bytes = await resp.read()
        return Image.open(BytesIO(img_bytes))