"""Hallo3D guidance (NeurIPS 2024).

Extends threestudio's StableDiffusionGuidance with the three Hallo3D
modules, following the generation-detection-correction paradigm:

  A. Multi-view Appearance Alignment (Sec. 3.2, Eq. 4)
     Self-attention K/V of the diffusion UNet are taken from a focal view
     (selected by Fovy, Appendix A) for all views in the rendered batch,
     both in the SDS noise prediction and in the L_CG denoising.

  B. Multi-modal Hallucination Detection (Sec. 3.3, Eq. 5)
     One rendered view plus the 3D-aware inquiry P_I is sent to a locally
     deployed LMM which returns the Enhanced Negative Prompt P_E^- or None
     when the rendering lacks complete semantic structure.

  C. Prompt-Enhanced Re-consistency (Sec. 3.4, Eq. 6-8)
     The rendering x_0 is DDIM-inverted to x_T and re-sampled with CFG in
     which the null/negative prompt is replaced by P_E^-, using the aligned
     denoiser; L_CG = MSE(x_hat_0, x_0) in image space, added to L_SDS with
     weight w (0.1) only when the detector returns a prompt, starting late
     in training and computed every `cg_interval` iterations (Appendix B).

Tuning-free and plug-and-play: select this guidance in any threestudio
config that uses "stable-diffusion-guidance" (incl. SJC via use_sjc=true).
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMInverseScheduler, DDIMScheduler

import threestudio
from threestudio.models.guidance.stable_diffusion_guidance import (
    StableDiffusionGuidance,
)
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.typing import *

from ..utils.appearance_attn import (
    AlignmentState,
    install_appearance_alignment,
    select_focal_view,
)
from ..utils.lmm_client import LMMDetectorClient


@threestudio.register("hallo3d-sd-guidance")
class Hallo3DStableDiffusionGuidance(StableDiffusionGuidance):
    @dataclass
    class Config(StableDiffusionGuidance.Config):
        # --- Module A: Multi-view Appearance Alignment ---
        enable_appearance_alignment: bool = True
        # 120% of the baselines' default minimum Fovy (40 deg) -> 48 deg
        focal_fovy_threshold_deg: float = 48.0
        # which self-attention layers to align: "all" (Eq. 4 literal) or
        # "decoder" (up_blocks only; keeps structure formation unaligned)
        align_layers: str = "all"
        # "replace" = Eq. 4 literal (K/V from focal view only);
        # "extend" = mutual attention, K/V = [own; focal] concatenation —
        # keeps every view's own structure anchor while sharing appearance.
        # "extend" is the robust plug-and-play default: at SDS guidance
        # scales (~100) "replace" collapses all views onto the focal view,
        # which prevents structure formation when the 3D representation is
        # trained from scratch (see docs/DIAGNOSIS.md). "replace" remains a
        # valid stronger choice when a 3D-consistent initialization exists
        # (e.g. GaussianDreamer with shap-e init).
        align_mode: str = "extend"
        # apply AAttn only at appearance-dominant (low-noise) timesteps:
        # aligned iff sampled t <= ratio * num_train_timesteps
        align_max_timestep_ratio: float = 1.0
        # training step from which alignment is applied
        align_start_step: int = 0

        # --- Module B: Multi-modal Hallucination Detection ---
        enable_hallucination_detection: bool = True
        lmm_server_url: str = "http://127.0.0.1:39121"
        lmm_query_timeout: float = 90.0
        # general prompt used as universal enhancement (Appendix A)
        general_negative_prompt: str = (
            "unnatural colors, poor lighting, low quality, artifacts, smooth texture"
        )

        # --- Module C: Prompt-Enhanced Re-consistency ---
        enable_cg_loss: bool = True
        cg_start_step: int = 1000  # delayed introduction of L_CG (Appendix B)
        cg_interval: int = 4  # compute L_CG every 4 iterations (Appendix B)
        cg_ddim_steps: int = 20
        cg_guidance_scale: float = 7.5
        # fraction of the DDIM chain used for inversion (1.0 = invert to x_T)
        cg_strength: float = 1.0
        # when detection is disabled ("w/o C & P_E^-" uses False+False; the
        # "w/o C" ablation applies P_E^- to the SDS CFG instead of L_CG)
        apply_pe_neg_to_sds: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Module A: patch UNet attention
        self.align_state = AlignmentState(
            enabled=False, num_views=4, focal_idx=0, mode=self.cfg.align_mode
        )
        self._align_requested = False
        if self.cfg.enable_appearance_alignment:
            install_appearance_alignment(
                self.unet, self.align_state, layers=self.cfg.align_layers
            )
            threestudio.info(
                "[Hallo3D] Multi-view Appearance Alignment installed on UNet "
                f"(layers={self.cfg.align_layers})."
            )

        # Module B: LMM detector client
        self.detector = (
            LMMDetectorClient(self.cfg.lmm_server_url, self.cfg.lmm_query_timeout)
            if self.cfg.enable_hallucination_detection
            else None
        )
        self._pe_neg: Optional[str] = None
        self._pe_neg_emb: Optional[torch.Tensor] = None

        # Module C: text encoder (the base class deletes it) + DDIM schedulers
        if self.cfg.enable_cg_loss or self.cfg.apply_pe_neg_to_sds:
            from transformers import AutoTokenizer, CLIPTextModel

            self.cg_tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
            )
            self.cg_text_encoder = (
                CLIPTextModel.from_pretrained(
                    self.cfg.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    torch_dtype=self.weights_dtype,
                )
                .to(self.device)
                .eval()
            )
            for p in self.cg_text_encoder.parameters():
                p.requires_grad_(False)

            self.cg_scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="scheduler"
            )
            self.cg_inverse_scheduler = DDIMInverseScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="scheduler"
            )

        self._global_step = 0

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        super().update_step(epoch, global_step, on_load_weights)
        self._global_step = global_step

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        # Module A gate: alignment is applied per UNet call, restricted to
        # appearance-dominant (low-noise) timesteps when
        # align_max_timestep_ratio < 1.
        if self._align_requested:
            t_mean = float(t.float().mean())
            self.align_state.enabled = t_mean <= (
                self.cfg.align_max_timestep_ratio * self.num_train_timesteps
            )
        else:
            self.align_state.enabled = False
        try:
            return super().forward_unet(latents, t, encoder_hidden_states)
        finally:
            self.align_state.enabled = False

    # ------------------------------------------------------------------ #
    # Module B helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.cg_tokenizer(
            [text],
            padding="max_length",
            max_length=self.cg_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return self.cg_text_encoder(tokens.input_ids)[0]

    def detect_hallucination(
        self, rgb_BCHW: torch.Tensor, focal_idx: int, prompt: str
    ) -> Optional[str]:
        """Query the LMM with the focal-view rendering; returns P_E^- or None."""
        if self.detector is None:
            return None
        return self.detector.query(rgb_BCHW[focal_idx], prompt)

    # ------------------------------------------------------------------ #
    # Module C: DDIM inversion + prompt-enhanced re-sampling (Eq. 6)
    # ------------------------------------------------------------------ #
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def reconsist_images(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        cond_embeddings: Float[Tensor, "B N Nf"],
        pe_neg_embedding: Float[Tensor, "1 N Nf"],
    ) -> Float[Tensor, "B 3 512 512"]:
        batch_size = latents.shape[0]
        cond_embeddings = cond_embeddings.to(self.weights_dtype)
        neg_embeddings = pe_neg_embedding.to(self.weights_dtype).expand(
            batch_size, -1, -1
        )

        n_steps = self.cfg.cg_ddim_steps
        k = max(1, min(n_steps, int(round(n_steps * self.cfg.cg_strength))))

        # --- DDIM inversion x_0 -> x_T (guidance scale 1, cond embedding) ---
        self.cg_inverse_scheduler.set_timesteps(n_steps, device=self.device)
        inv_timesteps = self.cg_inverse_scheduler.timesteps[:k]
        lat = latents.to(self.weights_dtype)
        for t in inv_timesteps:
            noise_pred = self.forward_unet(
                lat, t.repeat(batch_size), encoder_hidden_states=cond_embeddings
            )
            lat = self.cg_inverse_scheduler.step(noise_pred, t, lat).prev_sample

        # --- DDIM sampling with CFG; null prompt replaced by P_E^- (Eq. 6) ---
        self.cg_scheduler.set_timesteps(n_steps, device=self.device)
        sample_timesteps = self.cg_scheduler.timesteps[n_steps - k :]
        for t in sample_timesteps:
            latent_model_input = torch.cat([lat] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                t.repeat(batch_size * 2),
                encoder_hidden_states=torch.cat(
                    [cond_embeddings, neg_embeddings], dim=0
                ),
            )
            noise_pred_cond, noise_pred_neg = noise_pred.chunk(2)
            noise_pred = noise_pred_neg + self.cfg.cg_guidance_scale * (
                noise_pred_cond - noise_pred_neg
            )
            lat = self.cg_scheduler.step(noise_pred, t, lat).prev_sample

        return self.decode_latents(lat.to(latents.dtype))

    def compute_loss_cg(
        self,
        rgb_BCHW_512: Float[Tensor, "B 3 512 512"],
        prompt_utils: PromptProcessorOutput,
        elevation,
        azimuth,
        camera_distances,
        focal_idx: int,
    ) -> Optional[Float[Tensor, ""]]:
        batch_size = rgb_BCHW_512.shape[0]

        if self.cfg.enable_hallucination_detection:
            # Module B: P_E^- = D_psi(x, P_I) (Eq. 5)
            pe_neg = self.detect_hallucination(
                rgb_BCHW_512.detach(), focal_idx, prompt_utils.prompt
            )
            if pe_neg != self._pe_neg:
                msg = f"[Hallo3D] step {self._global_step} P_E^-: {pe_neg}"
                threestudio.info(msg)
                try:
                    import os

                    with open(
                        os.environ.get(
                            "HALLO3D_PE_LOG", "/tmp/hallo3d_pe_neg.log"
                        ),
                        "a",
                    ) as f:
                        f.write(msg + "\n")
                except OSError:
                    pass
            if pe_neg is None:
                # detector judged the rendering semantically incomplete (Eq. 8)
                self._pe_neg = None
                return None
            self._pe_neg = pe_neg
            neg_text = f"{pe_neg}, {self.cfg.general_negative_prompt}"
        else:
            # ablation "w/o B": L_CG guided by the general prompt only
            neg_text = self.cfg.general_negative_prompt
        pe_neg_emb = self.encode_text(neg_text)

        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        cond_embeddings = text_embeddings[:batch_size]

        with torch.no_grad():
            latents = self.encode_images(rgb_BCHW_512.detach())
            xhat_0 = self.reconsist_images(latents, cond_embeddings, pe_neg_emb)

        # L_CG in image space (Eq. 7); gradients flow into the renderings
        return (
            F.mse_loss(rgb_BCHW_512, xhat_0.detach(), reduction="sum") / batch_size
        )

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        # focal view from camera Fovy (Appendix A)
        fovy = kwargs.get("fovy", None)
        focal_idx = (
            select_focal_view(fovy, self.cfg.focal_fovy_threshold_deg)
            if fovy is not None
            else 0
        )
        self.align_state.num_views = batch_size
        self.align_state.focal_idx = focal_idx

        step = self._global_step
        on_cg_schedule = step >= self.cfg.cg_start_step and (
            (step - self.cfg.cg_start_step) % self.cfg.cg_interval == 0
        )

        # Ablation "w/o C": the LMM output P_E^- is applied to the SDS CFG
        # (as the unconditional/negative embedding) instead of through L_CG.
        if self.cfg.apply_pe_neg_to_sds and not rgb_as_latents:
            if on_cg_schedule:
                rgb_512_query = F.interpolate(
                    rgb.permute(0, 3, 1, 2),
                    (512, 512),
                    mode="bilinear",
                    align_corners=False,
                )
                self._pe_neg = self.detect_hallucination(
                    rgb_512_query.detach(), focal_idx, prompt_utils.prompt
                )
            if self._pe_neg is not None:
                emb = self.encode_text(
                    f"{self._pe_neg}, {self.cfg.general_negative_prompt}"
                ).to(prompt_utils.text_embeddings.dtype)
                n_vd = prompt_utils.uncond_text_embeddings_vd.shape[0]
                prompt_utils = replace(
                    prompt_utils,
                    uncond_text_embeddings=emb[0],
                    uncond_text_embeddings_vd=emb.expand(n_vd, -1, -1),
                )

        # Module A active during the SDS noise prediction (and inside the
        # L_CG denoising, both via forward_unet)
        self._align_requested = (
            self.cfg.enable_appearance_alignment
            and batch_size > 1
            and self._global_step >= self.cfg.align_start_step
        )
        try:
            guidance_out = super().__call__(
                rgb,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                rgb_as_latents=rgb_as_latents,
                guidance_eval=guidance_eval,
                **kwargs,
            )

            # Module C on schedule (Appendix B)
            if (
                self.cfg.enable_cg_loss
                and not rgb_as_latents
                and on_cg_schedule
            ):
                rgb_BCHW = rgb.permute(0, 3, 1, 2)
                rgb_BCHW_512 = F.interpolate(
                    rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
                )
                loss_cg = self.compute_loss_cg(
                    rgb_BCHW_512,
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    focal_idx,
                )
                if loss_cg is not None:
                    guidance_out["loss_cg"] = loss_cg
        finally:
            self._align_requested = False
            self.align_state.enabled = False

        return guidance_out
