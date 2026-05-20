import os
import torch

class Settings:
    def __init__(self):
        # Аппаратное обеспечение
        self.device = self._detect_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Профиль (автоопределение или через переменную окружения)
        self.profile = os.getenv("PROFILE", "auto")
        if self.profile == "auto":
            self.profile = self._determine_profile()

        # URL удалённых сервисов (если заданы, используются вместо локальных)
        self.text2image_url = os.getenv("TEXT2IMAGE_SERVICE_URL")
        self.image2shape_url = os.getenv("IMAGE2SHAPE_SERVICE_URL")

        # Пути к моделям для локального запуска
        self.text2img_model = os.getenv(
            "TEXT2IMG_MODEL",
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        )
        self.shape_model = os.getenv("SHAPE_MODEL", "tencent/Hunyuan3D-2mini")
        self.tex_model = os.getenv("TEX_MODEL", "tencent/Hunyuan3D-2")
        self.subfolder = os.getenv("SUBFOLDER", "hunyuan3d-dit-v2-0")

    def _detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _determine_profile(self):
        if self.device == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram > 10 * 1024**3:  # >10 ГБ VRAM
                return "high"
        return "low"

settings = Settings()