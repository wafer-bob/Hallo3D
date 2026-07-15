"""Hallo3D guidance for image-to-3D (Zero-1-to-3 family).

Wraps threestudio's StableZero123Guidance with the three Hallo3D modules:

  A. Multi-view Appearance Alignment — the LDM UNet's self-attention
     layers are patched so all views in the rendered batch share the focal
     view's K/V (mode "extend" by default, see docs/DIAGNOSIS.md).
  B. Multi-modal Hallucination Detection — LLaVA inquiry on the focal
     rendering; since image-driven generation has no text prompt, the
     scene description used in P_I comes from `scene_prompt` (or a one-off
     LLaVA caption of the reference image when left empty).
  C. Prompt-Enhanced Re-consistency — L_CG is computed with a separate
     text-conditioned SD 2.1 denoiser (utils/sd_reconsist.py), because the
     enhanced negative prompt P_E^- requires text conditioning (Eq. 6);
     this matches Fig. 3 where a generic 2D diffusion model performs the
     re-consistency step, bridging text-driven and image-driven pipelines.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.guidance.stable_zero123_guidance import (
    StableZero123Guidance,
)
from threestudio.utils.typing import *

from ..utils.appearance_attn import AlignmentState, select_focal_view
from ..utils.appearance_attn_ldm import install_ldm_appearance_alignment
from ..utils.lmm_client import LMMDetectorClient, image_tensor_to_b64png
from ..utils.sd_reconsist import SDReconsistency


@threestudio.register("hallo3d-stable-zero123-guidance")
class Hallo3DStableZero123Guidance(StableZero123Guidance):
    @dataclass
    class Config(StableZero123Guidance.Config):
        # --- Module A ---
        enable_appearance_alignment: bool = True
        align_mode: str = "extend"
        focal_fovy_threshold_deg: float = 48.0

        # --- Module B ---
        enable_hallucination_detection: bool = True
        lmm_server_url: str = "http://127.0.0.1:39122"
        lmm_query_timeout: float = 90.0
        # object description used in the inquiry P_I; empty -> caption the
        # reference image once with the LMM
        scene_prompt: str = ""
        general_negative_prompt: str = (
            "unnatural colors, poor lighting, low quality, artifacts, smooth texture"
        )

        # --- Module C ---
        enable_cg_loss: bool = True
        sd_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        cg_start_step: int = 400  # zero123 system trains 600 steps by default
        cg_interval: int = 4
        cg_ddim_steps: int = 20
        cg_guidance_scale: float = 7.5
        cg_strength: float = 1.0

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.align_state = AlignmentState(
            enabled=False, num_views=4, focal_idx=0, mode=self.cfg.align_mode
        )
        if self.cfg.enable_appearance_alignment:
            n = install_ldm_appearance_alignment(self.model, self.align_state)
            threestudio.info(
                f"[Hallo3D] LDM appearance alignment installed on {n} "
                f"self-attention layers (mode={self.cfg.align_mode})."
            )

        self.detector = (
            LMMDetectorClient(self.cfg.lmm_server_url, self.cfg.lmm_query_timeout)
            if self.cfg.enable_hallucination_detection
            else None
        )
        self._scene_prompt = self.cfg.scene_prompt
        self._pe_neg: Optional[str] = None

        self.reconsist: Optional[SDReconsistency] = None  # lazy (VRAM)
        self._global_step = 0

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        super().update_step(epoch, global_step, on_load_weights)
        self._global_step = global_step

    # ------------------------------------------------------------------ #
    def _resolve_scene_prompt(self, image_chw: torch.Tensor) -> str:
        """Caption the reference image once when no prompt is provided."""
        if self._scene_prompt:
            return self._scene_prompt
        caption = None
        if self.detector is not None:
            try:
                response = self.detector._post(
                    {
                        "image_b64": image_tensor_to_b64png(image_chw),
                        "prompt": (
                            "Name the main object in this image with a short "
                            "noun phrase (at most six words). Answer with the "
                            "phrase only."
                        ),
                    }
                )
                caption = response.strip().strip(".\"'")
            except Exception:
                caption = None
        self._scene_prompt = caption or "an object"
        threestudio.info(f"[Hallo3D] scene prompt: {self._scene_prompt}")
        return self._scene_prompt

    def compute_loss_cg(
        self, rgb_BCHW: Float[Tensor, "B 3 H W"], focal_idx: int
    ) -> Optional[Float[Tensor, ""]]:
        batch_size = rgb_BCHW.shape[0]
        scene_prompt = self._resolve_scene_prompt(rgb_BCHW[focal_idx].detach())

        if self.detector is not None:
            pe_neg = self.detector.query(
                rgb_BCHW[focal_idx].detach(), scene_prompt
            )
            if pe_neg != self._pe_neg:
                threestudio.info(
                    f"[Hallo3D] step {self._global_step} P_E^-: {pe_neg}"
                )
            self._pe_neg = pe_neg
            if pe_neg is None:
                return None  # Eq. 8
            neg_text = f"{pe_neg}, {self.cfg.general_negative_prompt}"
        else:
            neg_text = self.cfg.general_negative_prompt

        if self.reconsist is None:
            self.reconsist = SDReconsistency(
                self.cfg.sd_model_name_or_path,
                device=self.device,
                align_mode=self.cfg.align_mode,
            )

        with torch.no_grad():
            xhat = self.reconsist.reconsist(
                rgb_BCHW.detach(),
                positive_prompt=f"a 3d rendering of {scene_prompt}, high quality",
                negative_prompt=neg_text,
                num_steps=self.cfg.cg_ddim_steps,
                guidance_scale=self.cfg.cg_guidance_scale,
                strength=self.cfg.cg_strength,
                focal_idx=focal_idx,
            )
        xhat = F.interpolate(
            xhat, rgb_BCHW.shape[-2:], mode="bilinear", align_corners=False
        )
        return F.mse_loss(rgb_BCHW, xhat.detach(), reduction="sum") / batch_size

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        fovy = kwargs.get("fovy", None)
        focal_idx = (
            select_focal_view(fovy, self.cfg.focal_fovy_threshold_deg)
            if fovy is not None
            else 0
        )
        self.align_state.num_views = batch_size
        self.align_state.focal_idx = focal_idx
        self.align_state.enabled = (
            self.cfg.enable_appearance_alignment and batch_size > 1
        )
        try:
            guidance_out = super().__call__(
                rgb,
                elevation,
                azimuth,
                camera_distances,
                rgb_as_latents=rgb_as_latents,
                guidance_eval=guidance_eval,
                **kwargs,
            )

            step = self._global_step
            if (
                self.cfg.enable_cg_loss
                and not rgb_as_latents
                and step >= self.cfg.cg_start_step
                and (step - self.cfg.cg_start_step) % self.cfg.cg_interval == 0
            ):
                loss_cg = self.compute_loss_cg(
                    rgb.permute(0, 3, 1, 2), focal_idx
                )
                if loss_cg is not None:
                    guidance_out["loss_cg"] = loss_cg
        finally:
            self.align_state.enabled = False

        return guidance_out
