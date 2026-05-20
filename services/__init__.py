from config import settings
from .text2image_service import Text2ImageService as LocalText2Image
from .image2shape_service import Image2ShapeService as LocalImage2Shape
from .remote_text2image import RemoteText2ImageService
from .remote_image2shape import RemoteImage2ShapeService
from .base import BaseText2ImageService, BaseImage2ShapeService

def create_text2image_service() -> BaseText2ImageService:
    if settings.text2image_url:
        return RemoteText2ImageService(settings.text2image_url)
    return LocalText2Image(settings.profile)

def create_image2shape_service() -> BaseImage2ShapeService:
    if settings.image2shape_url:
        return RemoteImage2ShapeService(settings.image2shape_url)
    if settings.profile == "high":
        return LocalImage2Shape(settings.profile)
    return None