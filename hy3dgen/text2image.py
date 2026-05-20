import random
import torch
from diffusers import HunyuanDiTPipeline as _HunyuanDiTPipeline
from transformers import T5EncoderModel
import gc

NEGATIVE_PROMPT = ""

class HunyuanDiTPipeline:
    def __init__(self, model_path="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled", device="cuda"):
        self.model_path = model_path
        self.device = device
        self.default_negative_prompt = NEGATIVE_PROMPT
        self.pipeline = None  # будет создан при первом вызове

    def _flush(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _get_text_emb(self, prompts):
        # Загружаем T5 энкодер в 8 бит для экономии памяти
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.model_path,
            subfolder="text_encoder_2",
            load_in_8bit=True,
            device_map="auto",
        )
        encoder_pipeline = _HunyuanDiTPipeline.from_pretrained(
            self.model_path,
            text_encoder_2=text_encoder_2,
            transformer=None,
            vae=None,
            torch_dtype=torch.float16,
            device_map="balanced",
        )
        # Параметры для второго энкодера
        text_encoder_conf = {
            "negative_prompt": self.default_negative_prompt,
            "prompt_embeds": None,
            "negative_prompt_embeds": None,
            "prompt_attention_mask": None,
            "negative_prompt_attention_mask": None,
            "max_sequence_length": 256,
            "text_encoder_index": 1,
        }
        prompt_emb1 = encoder_pipeline.encode_prompt(prompts, negative_prompt=self.default_negative_prompt)
        prompt_emb2 = encoder_pipeline.encode_prompt(prompts, **text_encoder_conf)
        del text_encoder_2, encoder_pipeline
        self._flush()
        return prompt_emb1, prompt_emb2

    def _load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = _HunyuanDiTPipeline.from_pretrained(
                self.model_path,
                text_encoder=None,
                text_encoder_2=None,
                torch_dtype=torch.float16,
            ).to(self.device)

    @torch.no_grad()
    def __call__(self, prompt, seed=0):
        # Метод для совместимости с api_server.py
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Получаем эмбеддинги
        prompt_emb1, prompt_emb2 = self._get_text_emb(prompt)
        (prompt_embeds, neg_embeds, prompt_mask, neg_mask) = prompt_emb1
        (prompt_embeds_2, neg_embeds_2, prompt_mask_2, neg_mask_2) = prompt_emb2

        # Загружаем основной пайплайн (без энкодеров)
        self._load_pipeline()

        images = self.pipeline(
            prompt_embeds=prompt_embeds,
            prompt_embeds_2=prompt_embeds_2,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_embeds_2=neg_embeds_2,
            prompt_attention_mask=prompt_mask,
            prompt_attention_mask_2=prompt_mask_2,
            negative_prompt_attention_mask=neg_mask,
            negative_prompt_attention_mask_2=neg_mask_2,
            num_images_per_prompt=1,
            guidance_scale=6.0,
            num_inference_steps=25,
            generator=generator,
        ).images[0]
        return images