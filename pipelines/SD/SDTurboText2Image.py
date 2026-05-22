import torch
from PIL import Image
from pipelines.base import BaseText2Image

# ---------- SD Turbo (лёгкая модель для CPU / слабых GPU) ----------
class SDTurboText2Image(BaseText2Image):
    """
    Облегчённая модель Stable Diffusion Turbo (512px, 1 шаг).
    Идеально для профиля low (CPU или Mac).
    """
    def __init__(self, device: str = "cpu"):
        from diffusers import AutoPipelineForText2Image

        self.device = device
        # SD Turbo оптимизирована для 1–4 шагов
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float32,  # на CPU float16 может не работать
            safety_checker=None,
            requires_safety_checker=False
        )
        if device != "cpu":
            self.pipe.to(device)

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        # Рекомендованные параметры для Turbo: guidance_scale=0.0, steps=1
        defaults = {
            "num_inference_steps": 1,
            "guidance_scale": 0.0,
        }
        params = {**defaults, **kwargs}
        return self.pipe(prompt, **params).images[0]