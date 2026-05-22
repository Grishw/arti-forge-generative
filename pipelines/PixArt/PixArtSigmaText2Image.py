import torch
from PIL import Image
from pipelines.base import BaseText2Image
from diffusers import PixArtSigmaPipeline

# ---------- PixArt-Sigma (высокое качество, требует GPU) ----------
class PixArtSigmaText2Image(BaseText2Image):
    """
    Использует модель PixArt-Sigma XL 1024px через diffusers.
    По умолчанию загружается в torch.float16 на CUDA.
    """
    def __init__(
        self,
        model_path: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        device: str = "cuda",
        offload: bool = False
    ):
       
        self.device = device
        self.offload = offload
        self.model_path = model_path

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        # Параметры по умолчанию можно переопределить
        defaults = {
            "num_inference_steps": 20,
            "guidance_scale": 4.5,
        }
        params = {**defaults, **kwargs}
        return self.pipe(prompt, **params).images[0]