import os

import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionImg2ImgPipeline
import pandas as pd
import torch
from PIL import Image
from ip_adapter.ip_adapter import IPAdapter

device = "cuda"
from diffusers import StableDiffusionPipeline

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

pipe = StableDiffusionPipeline.from_single_file(
    '/v1-5-pruned-emaonly.ckpt',
    torch_dtype=torch.float32)
pipe.safety_checker = None
pipe.feature_extractor = None
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
ip_adapter = IPAdapter(pipe, "id/encoder", 
                                                                           "/id/ip"
                                                                           "-adapter_sd15.bin", device=device)
# image = Image.open('./eval/dog.jpg')
# embs, _ = ip_adapter.get_image_embeds(image)
root_dir = 'datasets/imageNet_images'
# 遍历根目录及其子目录
i = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # 检查文件扩展名是否为图片格式
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构建完整的文件路径
            file_path = os.path.join(subdir, file)
            name, _ = os.path.splitext(file)
            # 打印图片文件的路径
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    embs, _ = ip_adapter.get_image_embeds(img)
                    torch.save(embs, "/data/ip_emb/" + name)
                    print(i)
                    i += 1
            except Exception as e:
                print(f"Error resizing {file_path}: {e}")
