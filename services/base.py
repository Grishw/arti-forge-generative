from abc import ABC, abstractmethod
from PIL import Image
import trimesh

class BaseText2ImageService(ABC):
    """Интерфейс сервиса генерации изображений по тексту."""
    @abstractmethod
    async def load_model(self): pass
    @abstractmethod
    async def unload_model(self): pass
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Image.Image: pass
    @abstractmethod
    def is_loaded(self) -> bool: pass

class BaseImage2ShapeService(ABC):
    """Интерфейс сервиса генерации 3D-меша по изображению."""
    @abstractmethod
    async def load_model(self): pass
    @abstractmethod
    async def unload_model(self): pass
    @abstractmethod
    async def generate(self, image: Image.Image, **kwargs) -> trimesh.Trimesh: pass
    @abstractmethod
    def is_loaded(self) -> bool: pass
    @abstractmethod
    async def maybe_offload_if_needed(self): pass

class BaseText2ShapeService(ABC):
    """Композитный сервис: текст -> 3D."""
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> trimesh.Trimesh: pass