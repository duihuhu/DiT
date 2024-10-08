# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#python sample.py  --image-size 256  --seed 1
"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import time
from PIL import Image
import numpy as np

def save_tensor_as_image(tensor, filename):
    # 确保张量的维度是 (C, H, W) 或 (H, W, C)
    if tensor.dim() == 4 and tensor.size(1) == 4:
        tensor = tensor[0]  # 选择第一个样本
    
    # 将张量转换为 numpy 数组
    tensor = tensor.squeeze().cpu().numpy()  # 去掉多余的维度，并移动到 CPU 上
    if tensor.shape[0] == 4:
        # 如果有4个通道，选择其中一个通道，通常情况下 RGB 图像有3个通道
        tensor = tensor[:3]
    
    # 归一化到 [0, 255]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype(np.uint8)
    
    # 转换为图像并保存
    if tensor.shape[0] == 1:
        # 单通道图像
        img = Image.fromarray(tensor[0], mode='L')
    else:
        # 多通道图像
        img = Image.fromarray(np.transpose(tensor, (1, 2, 0)), mode='RGB')
    
    img.save(filename)

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/home/jovyan/models/sd-vae-ft-mse/models/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493").to(device)
    
    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    class_labels = [146]
    
    class_labels_6 = [186]
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    
    save_tensor_as_image(z[0], 'noise_image.png')

    y = torch.tensor(class_labels, device=device)
    
    y6 = torch.tensor(class_labels_6, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    
    y6 = torch.cat([y6, y_null], 0)
    
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    
    model_kwargs6 = dict(y=y6, cfg_scale=args.cfg_scale)

    # Sample images:
    t1 = time.time()
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, model_kwargs6=model_kwargs6, progress=True, device=device
    )
    t2 = time.time()
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    t3 = time.time()
    print("diffusion p_sample_loop ", t3-t2, t2-t1)
    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
