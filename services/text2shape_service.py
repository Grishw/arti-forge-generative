from services.base import BaseText2ShapeService, BaseText2ImageService, BaseImage2ShapeService

class Text2ShapeService(BaseText2ShapeService):
    def __init__(self, text2img: BaseText2ImageService, img2shape: BaseImage2ShapeService):
        self.text2img = text2img
        self.img2shape = img2shape

    async def generate(self, prompt: str, **kwargs):
        # Перед генерацией 2D освобождаем память от 3D модели, если необходимо
        await self.img2shape.maybe_offload_if_needed()
        image = await self.text2img.generate(prompt)
        mesh = await self.img2shape.generate(image, **kwargs)
        return mesh