"""Standalone Prompt-Enhanced Re-consistency module (Hallo3D Sec. 3.4).

For frameworks whose guidance model is not text-conditioned (Zero-1-to-3,
DreamGaussian's zero123), L_CG still follows Fig. 3: the rendered views
are DDIM-inverted and re-sampled by a *text-conditioned* 2D diffusion
model (SD 2.1-base) whose null prompt is replaced by the enhanced
negative prompt P_E^- (Eq. 6), with multi-view appearance alignment
active inside the denoiser.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)

from .appearance_attn import AlignmentState, install_appearance_alignment


class SDReconsistency:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        align_mode: str = "extend",
    ):
        from transformers import AutoTokenizer, CLIPTextModel

        self.device, self.dtype = device, dtype
        path = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
        self.text_encoder = (
            CLIPTextModel.from_pretrained(
                path, subfolder="text_encoder", torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        self.vae = (
            AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
            .to(device)
            .eval()
        )
        self.unet = (
            UNet2DConditionModel.from_pretrained(
                path, subfolder="unet", torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        for m in (self.text_encoder, self.vae, self.unet):
            for p in m.parameters():
                p.requires_grad_(False)
        self.scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler")
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            path, subfolder="scheduler"
        )
        self.align_state = AlignmentState(
            enabled=False, num_views=4, focal_idx=0, mode=align_mode
        )
        install_appearance_alignment(self.unet, self.align_state)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            [text],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return self.text_encoder(tokens.input_ids)[0]

    @torch.no_grad()
    def reconsist(
        self,
        images: torch.Tensor,  # [B,3,H,W] in [0,1]
        positive_prompt: str,
        negative_prompt: str,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        strength: float = 1.0,
        focal_idx: int = 0,
    ) -> torch.Tensor:
        """DDIM invert -> CFG re-sample with P_E^- (Eq. 6); returns x_hat_0."""
        batch_size = images.shape[0]
        imgs_512 = F.interpolate(
            images, (512, 512), mode="bilinear", align_corners=False
        ).to(self.dtype)

        latents = (
            self.vae.encode(imgs_512 * 2 - 1).latent_dist.sample()
            * self.vae.config.scaling_factor
        )

        cond = self.encode_text(positive_prompt).expand(batch_size, -1, -1)
        neg = self.encode_text(negative_prompt).expand(batch_size, -1, -1)

        self.align_state.enabled = batch_size > 1
        self.align_state.num_views = batch_size
        self.align_state.focal_idx = focal_idx
        try:
            k = max(1, min(num_steps, int(round(num_steps * strength))))
            self.inverse_scheduler.set_timesteps(num_steps, device=self.device)
            lat = latents
            for t in self.inverse_scheduler.timesteps[:k]:
                eps = self.unet(lat, t, encoder_hidden_states=cond).sample
                lat = self.inverse_scheduler.step(eps, t, lat).prev_sample

            self.scheduler.set_timesteps(num_steps, device=self.device)
            for t in self.scheduler.timesteps[num_steps - k :]:
                eps = self.unet(
                    torch.cat([lat] * 2),
                    t,
                    encoder_hidden_states=torch.cat([cond, neg]),
                ).sample
                eps_cond, eps_neg = eps.chunk(2)
                eps = eps_neg + guidance_scale * (eps_cond - eps_neg)
                lat = self.scheduler.step(eps, t, lat).prev_sample
        finally:
            self.align_state.enabled = False

        image = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        return ((image + 1) / 2).clamp(0, 1).float()
