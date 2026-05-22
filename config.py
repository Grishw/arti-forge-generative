"""
ARTi Forge Configuration Module
--------------------------------
Загружает настройки из YAML-файла, с возможностью переопределения
через переменные окружения. Поддерживает:
  - автоопределение профиля (high/low) по видеопамяти
  - раскрытие ~ и $VAR в путях к моделям
  - singleton-доступ через `from config import settings`
"""
import os
import yaml
import torch
from pathlib import Path

class Settings:
    _instance = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        if self._initialized:
            return
        self._initialized = True

        # 1. Определяем путь к конфигурационному файлу
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "config.yaml")
        self._config_path = config_path
        self._raw = {}
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self._raw = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Configuration file '{config_path}' not found. Using defaults.")

        # 2. Аппаратное обеспечение
        self.device = self._get_str("device") or self._detect_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 3. Профиль
        self.profile = self._get_str("profile") or "auto"
        if self.profile == "auto":
            self.profile = self._determine_profile()

        # 4. Модели
        # Text2Image
        if self.profile == "high":
            self.text2image_model_name = self._get_str("models.text2image.high", "pixart_sigma")
            self.text2img_model = self._expand_path(
                self._get_str("models.text2image.high_path", "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
            )
        else:
            self.text2image_model_name = self._get_str("models.text2image.low", "sd_turbo")
            self.text2img_model = self._expand_path(
                self._get_str("models.text2image.low_path", "stabilityai/sd-turbo")
            )
        # Image2Shape
        self.image2shape_model_name = self._get_str("models.image2shape.name", "hunyuan_3d")
        self.shape_model = self._expand_path(
            self._get_str("models.image2shape.shape_path", "tencent/Hunyuan3D-2mini")
        )
        self.tex_model = self._expand_path(
            self._get_str("models.image2shape.tex_path", "tencent/Hunyuan3D-2")
        )
        self.subfolder = self._get_str("models.image2shape.subfolder", "hunyuan3d-dit-v2-0")
        self.enable_texture = self._get_bool("models.image2shape.enable_texture", True)
        self.max_faces = self._get_int("models.image2shape.max_faces", 40000)

        # 5. Удалённые сервисы
        self.text2image_url = self._expand_path(
            self._get_str("remote_services.text2image_url")
        )
        self.image2shape_url = self._expand_path(
            self._get_str("remote_services.image2shape_url")
        )

        # 6. Сервер
        self.host = self._get_str("server.host", "0.0.0.0")
        self.port = self._get_int("server.port", 8000)
        self.save_dir = self._expand_path(
            self._get_str("server.save_dir", "generated_models")
        )
        self.cleanup_hours = self._get_int("server.cleanup_hours", 24)

        # 7. Микросервисы
        self.text2image_port = self._get_int("microservices.text2image_port", 8001)
        self.image2shape_port = self._get_int("microservices.image2shape_port", 8002)

        # 8. Логирование
        self.log_level = self._get_str("logging.level", "INFO")
        self.log_file = self._get_str("logging.file", "server.log")

        # Применяем переопределения из переменных окружения
        self._apply_env_overrides()

        # Создаём папку для результатов, если её нет
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    def _detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _determine_profile(self):
        if self.device == "cuda":
            # Получаем общий объём памяти GPU
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory
                if total_vram > 10 * 1024**3:   # >10 ГБ
                    return "high"
            except:
                pass
        return "low"

    def _get_str(self, key, default=None):
        keys = key.split('.')
        val = self._raw
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return str(val) if val is not None else default

    def _get_int(self, key, default=None):
        val = self._get_str(key)
        try:
            return int(val) if val is not None else default
        except ValueError:
            return default

    def _get_bool(self, key, default=None):
        val = self._get_str(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return val.lower() in ('true', '1', 'yes')

    def _expand_path(self, value):
        if value and isinstance(value, str):
            value = os.path.expanduser(value)
            value = os.path.expandvars(value)
            if value.startswith('./') or value.startswith('.\\'):
                value = os.path.abspath(value)
            return value
        return value

    def _apply_env_overrides(self):
        """
        Отдельные переменные окружения могут переопределить загруженные значения.
        Приоритет: аргументы командной строки (в start.py) > переменные окружения > YAML.
        """
        mapping = {
            "PROFILE":                     ("profile", str),
            "DEVICE":                      ("device", str),
            "TEXT2IMAGE_SERVICE_URL":      ("text2image_url", str),
            "IMAGE2SHAPE_SERVICE_URL":     ("image2shape_url", str),
            "TEXT2IMG_MODEL":              ("text2img_model", str),
            "SHAPE_MODEL":                 ("shape_model", str),
            "TEX_MODEL":                   ("tex_model", str),
            "SUBFOLDER":                   ("subfolder", str),
            "TEXT2IMG_MODEL_NAME":         ("text2image_model_name", str),
            "IMAGE2SHAPE_MODEL_NAME":      ("image2shape_model_name", str),
        }
        for env_key, (attr_name, cast) in mapping.items():
            val = os.getenv(env_key)
            if val is not None:
                setattr(self, attr_name, cast(val))

# Синглтон
settings = Settings()