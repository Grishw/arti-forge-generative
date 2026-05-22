import torch
import trimesh
from PIL import Image
from pipelines.base import BaseImage2Shape
from hy3dgen.shapegen.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer
        

class Hunyuan3DShapePipeline(BaseImage2Shape):
    """
    Полный пайплайн создания 3D-объекта по RGB-изображению на базе Hunyuan3D-2.
    Включает:
      - удаление фона (BackgroundRemover)
      - генерацию сетки (Hunyuan3DDiTFlowMatchingPipeline)
      - постобработку: удаление плавающих граней, дегенератов, упрощение
      - опциональное текстурирование (Hunyuan3DPaintPipeline)

    Args:
        shape_model_path (str): путь или название HuggingFace-модели для геометрии.
        tex_model_path (str, optional): путь к модели текстурирования. Если None, текстура не накладывается.
        subfolder (str): подпапка внутри репозитория модели (обычно "hunyuan3d-dit-v2-0").
        device (str): устройство ("cuda" или "cpu").
        enable_texture (bool): если False, текстурная модель не загружается даже при указанном пути.
        max_faces (int): целевое количество граней после упрощения.
    """

    def __init__(
        self,
        shape_model_path: str,
        tex_model_path: str = None,
        subfolder: str = "",
        device: str = "cuda",
        enable_texture: bool = True,
        max_faces: int = 40000
    ):
        # Отложенный импорт, чтобы не требовать hy3dgen при отсутствии
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline 
        

        self.device = device
        self.max_faces = max_faces

        # Инструменты
        self.rembg = BackgroundRemover()

        # Пайплайн формы
        self.shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            shape_model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        # Ускорение (MC-алгоритм)
        self.shape_pipe.enable_flashvdm(mc_algo='mc')

        # Текстурирование
        self.tex_pipe = None
        if enable_texture and tex_model_path:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            self.tex_pipe = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)

    def generate(self, image: Image.Image, **kwargs) -> trimesh.Trimesh:
        """
        Генерирует 3D-меш с текстурами (если включено) по изображению.

        Args:
            image: PIL Image.
            **kwargs: возможные ключи
                seed (int): зерно генератора (по умолчанию 1234)
                octree_resolution (int): детализация (128-256)
                num_inference_steps (int): шаги диффузии (5)
                guidance_scale (float): сила следования изображению (5.0)
                texture (bool): переопределяет, накладывать ли текстуру

        Returns:
            trimesh.Trimesh.
        """
        # 1. Удаление фона
        image_no_bg = self.rembg(image)

        # 2. Параметры генерации
        seed = kwargs.get("seed", 1234)
        generator = torch.Generator(self.device).manual_seed(seed)
        params = {
            "image": image_no_bg,
            "generator": generator,
            "octree_resolution": kwargs.get("octree_resolution", 128),
            "num_inference_steps": kwargs.get("num_inference_steps", 5),
            "guidance_scale": kwargs.get("guidance_scale", 5.0),
            "mc_algo": "mc"
        }

        # 3. Генерация меша
        mesh = self.shape_pipe(**params)[0]

        # 4. Постобработка (всегда)
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=self.max_faces)

        # 5. Текстурирование (если требуется)
        if kwargs.get("texture", True) and self.tex_pipe is not None:
            mesh = self.tex_pipe(mesh, image_no_bg)

        return mesh