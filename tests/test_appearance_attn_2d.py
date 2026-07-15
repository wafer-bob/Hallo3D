"""2D sanity check for Module A (Multi-view Appearance Alignment).

Generates a batch of 4 images with plain SD sampling vs. with the
appearance-aligned attention processor enabled (K/V from view 0), same
seeds. If alignment is implemented correctly the aligned batch should look
plausible (not corrupted) and share color/texture with the focal image.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "threestudio-hallo3d"))
from utils.appearance_attn import AlignmentState, install_appearance_alignment

from diffusers import DDIMScheduler, StableDiffusionPipeline

MODEL = "stabilityai/stable-diffusion-2-1-base"
PROMPT = "a majestic lion standing on a rock, 3d render"
NEG = "unnatural colors, poor lighting, low quality, artifacts"


@torch.no_grad()
def main():
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    state = AlignmentState(enabled=False, num_views=4, focal_idx=0)
    install_appearance_alignment(pipe.unet, state)

    def gen(tag):
        g = torch.Generator(device).manual_seed(42)
        imgs = pipe(
            [PROMPT] * 4,
            negative_prompt=[NEG] * 4,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=g,
        ).images
        w, h = imgs[0].size
        from PIL import Image

        canvas = Image.new("RGB", (w * 4, h))
        for i, im in enumerate(imgs):
            canvas.paste(im, (i * w, 0))
        canvas.save(f"/tmp/attn2d_{tag}.png")
        print("saved", tag)

    gen("plain")
    state.enabled = True
    gen("aligned")


if __name__ == "__main__":
    main()
