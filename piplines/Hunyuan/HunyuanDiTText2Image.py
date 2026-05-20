import torch
from PIL import Image
from pipelines.base import BaseText2Image


# ---------- HunyuanDiT (качественная модель от Tencent) ----------
class HunyuanDiTText2Image(BaseText2Image):
    """
    Обёртка над официальным пайплайном HunyuanDiT из пакета hy3dgen.
    Подходит для профиля high, когда PixArt недоступен или нужна вариативность.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        from hy3dgen.text2image import HunyuanDiTPipeline
        
        self.pipeline = HunyuanDiTPipeline(
            model_path=model_path,
            device=device
        )

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        # HunyuanDiTPipeline генерирует одно изображение
        return self.pipeline(prompt)