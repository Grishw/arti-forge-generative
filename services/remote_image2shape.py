import aiohttp
import base64
import trimesh
from PIL import Image
from io import BytesIO
from services.base import BaseImage2ShapeService

class RemoteImage2ShapeService(BaseImage2ShapeService):
    def __init__(self, url: str):
        self.url = url.rstrip('/')

    async def load_model(self): pass
    async def unload_model(self): pass
    def is_loaded(self) -> bool: return True
    async def maybe_offload_if_needed(self): pass

    async def generate(self, image: Image.Image, **kwargs) -> trimesh.Trimesh:
        # Конвертируем изображение в base64
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {"image": img_b64, **kwargs}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/generate", json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Remote image2shape error {resp.status}: {error_text}")
                data = await resp.json()
                mesh_b64 = data["mesh_base64"]
                mesh_bytes = base64.b64decode(mesh_b64)
        return trimesh.load(BytesIO(mesh_bytes), file_type='glb')