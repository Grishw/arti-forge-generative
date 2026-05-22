from abc import ABC, abstractmethod
from PIL import Image
import trimesh

class BaseText2Image(ABC):
    """
    Интерфейс для моделей, генерирующих изображение по текстовому описанию.
    """
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Image.Image:
        """
        Генерирует изображение на основе текста.

        Args:
            prompt: Текстовое описание желаемого изображения.
            **kwargs: Дополнительные параметры, специфичные для модели
                      (например, num_inference_steps, guidance_scale, seed).

        Returns:
            Объект PIL.Image.
        """
        pass

class BaseImage2Shape(ABC):
    """
    Интерфейс для моделей, генерирующих 3D-меш по входному изображению.
    """
    @abstractmethod
    def generate(self, image: Image.Image, **kwargs) -> trimesh.Trimesh:
        """
        Генерирует 3D-меш (trimesh.Trimesh) на основе изображения.

        Args:
            image: PIL Image (обычно с уже удалённым фоном, но допустим любой).
            **kwargs: Параметры генерации (seed, octree_resolution,
                      num_inference_steps, guidance_scale, face_count и т.п.).

        Returns:
            trimesh.Trimesh.
        """
        pass