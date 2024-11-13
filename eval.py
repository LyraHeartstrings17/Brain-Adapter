import os

import numpy as np
from einops import rearrange

from dc_ldm.models.diffusion.plms import PLMSSampler

from dc_ldm.util import instantiate_from_config
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionImg2ImgPipeline

import torch
from PIL import Image
from ip_adapter.ip_adapter import IPAdapter
from omegaconf import OmegaConf

device = "cuda"


#
# pipe = StableDiffusionPipeline.from_single_file(
#     '/pretrains/models/v1-5-pruned.ckpt',
#     torch_dtype=torch.float16)
# pipe.safety_checker = None
# pipe.feature_extractor = None
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.to(device)
# ip_adapter = IPAdapter(pipe, "/id/encoder", 
#                                                                               "/id/ip"
#                                                                               "-adapter_sd15.bin", device=device)
# img = Image.open('/datasets/imageNet_images/n02106662/n02106662_1152.JPEG')
# prompt_embeds, negative_prompt_embeds = ip_adapter.get_image_embeds(
#     img
# )
# print(negative_prompt_embeds)
# for k in range(3, 6):
#     print('/img/4layers-sbj' + str(k + 1) + '/only_f_pres')
#     test_prompt_embeds = torch.load('/img/4layers-sbj' + str(k + 1) + '/pres')
#     for i in range(0, 4):
#         generator = torch.Generator().manual_seed(i + 1)
#         for j in range(0, len(test_prompt_embeds)):
#             print(i + 1, j)
#             image = pipe(
#                 prompt_embeds=test_prompt_embeds[j],
#                 negative_prompt_embeds=negative_prompt_embeds,
#                 num_inference_steps=100,
#                 guidance_scale=7.5,
#                 generator=generator,
#             ).images[0]
#             image.save('/img/4layers-sbj' + str(k + 1) + '/stage1/val_' + str(
#                 j) + '_img_' + str(
#                 i + 1) + '.jpg', lossless=True,
#                        quality=100)


def gen_img_stage2():
    model2 = StableDiffusionImg2ImgPipeline.from_single_file(
        '/project/DreamDiffusion-main/pretrains/models/v1-5-pruned.ckpt',
        torch_dtype=torch.float16).to(device)
    # test_prompts = torch.load('/img/4layers-sbj6/prompts')
    # ip_model = IPAdapter(model2, "/id/encoder", "/id/ip"
    #                                                                               "-adapter_sd15.bin", device=device)
    # test_prompts2 = torch.load('./eval/img/4layers-sbj5/prompts')
    # 找出不同项的索引
    # different_indices = [index for index, (item1, item2) in enumerate(zip(test_prompts, test_prompts2)) if
    #                      item1 != item2]
    for k in range(0, 6):
        print('/img/4layers-sbj' + str(k + 1) + '/prompts')
        test_prompts = torch.load('/img/4layers-sbj' + str(k + 1) + '/prompts')
        for i in range(len(test_prompts)):
            print("prompts is :", test_prompts[i])
            for j in range(4):
                print(j)
                image = Image.open(
                    '/img/4layers-sbj' + str(k + 1) + '/stage1/val_' + str(
                        i) + '_img_' + str(
                        j + 1) + '.jpg')
                generated_image = model2(prompt=test_prompts[i], image=image, strength=0.75, guidance_scale=7.5,
                                         num_inference_steps=100,
                                         ).images[0]
                generated_image.save(
                    '/img/4layers-sbj' + str(k + 1) + '/stage2/val_' + str(
                        i) + '_enhance_' + str(
                        j + 1) + '.jpg',
                    lossless=True, quality=100)


gen_img_stage2()
