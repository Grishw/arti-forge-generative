# pipelines/__init__.py
from .base import BaseText2Image, BaseImage2Shape

# Реестр доступных моделей
TEXT2IMAGE_MODELS = {
    "pixart_sigma": "pipelines.PixArt.PixArtSigmaText2Image.PixArtSigmaText2Image",
    "hunyuan_dit": "pipelines.Hunyuan.HunyuanDiTText2Image.HunyuanDiTText2Image",
    "sd_turbo": "pipelines.SD.SDTurboText2Image.SDTurboText2Image",
}

IMAGE2SHAPE_MODELS = {
    "hunyuan_3d": "pipelines.Hunyuan.Hunyuan3DShapePipeline.Hunyuan3DShapePipeline",
}

def get_text2image_class(name: str):
    import importlib
    full_path = TEXT2IMAGE_MODELS.get(name, name)   # если имя есть в реестре, берём полный путь
    module_path, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_image2shape_class(name: str):
    import importlib
    full_path = IMAGE2SHAPE_MODELS.get(name, name)   # аналогично
    module_path, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)