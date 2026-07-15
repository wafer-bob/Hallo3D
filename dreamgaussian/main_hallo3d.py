"""DreamGaussian + Hallo3D (NeurIPS 2024), image-to-3D.

Plug-and-play integration of the three Hallo3D modules into the official
DreamGaussian stage-1 trainer, without modifying main.py:

  A. Multi-view Appearance Alignment — the diffusers zero123 UNet's
     self-attention is patched (mode "extend"); active during the SDS
     batch of `opt.batch_size` novel views.
  B. Multi-modal Hallucination Detection — LLaVA two-round inquiry on a
     rendered novel view; the object description in P_I comes from a
     one-off LLaVA caption of the input image.
  C. Prompt-Enhanced Re-consistency — every `cg_interval` steps after
     `cg_start_iter`, four novel views are re-rendered, DDIM-inverted and
     re-sampled by SD 2.1 with the enhanced negative prompt (Eq. 6), and
     L_CG = w * MSE(x_hat_0, x_0) is applied in an extra optimizer step.

Usage (dg4d conda env):
    python main_hallo3d.py --config configs/image.yaml \
        input=data/name_rgba.png save_path=name_hallo3d batch_size=4
Extra keys (all optional): hallo3d_lmm_url, hallo3d_cg_start_ratio,
hallo3d_cg_interval, hallo3d_align_mode, hallo3d_scene_prompt.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# NOTE: appended (not inserted) so DreamGaussian's own `guidance/` package
# keeps priority over threestudio-hallo3d's `guidance/`.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "threestudio-hallo3d")
)
from utils.appearance_attn import (  # noqa: E402
    AlignmentState,
    install_appearance_alignment,
)
from utils.lmm_client import LMMDetectorClient  # noqa: E402
from utils.sd_reconsist import SDReconsistency  # noqa: E402

# headless: stub dearpygui (broken libstdc++ in this env; GUI unused)
import types  # noqa: E402


class _DPGStub(types.ModuleType):
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


_dpg_parent = types.ModuleType("dearpygui")
_dpg_child = _DPGStub("dearpygui.dearpygui")
_dpg_parent.dearpygui = _dpg_child
sys.modules.setdefault("dearpygui", _dpg_parent)
sys.modules.setdefault("dearpygui.dearpygui", _dpg_child)

import main as dg_main  # noqa: E402  (original DreamGaussian)
from cam_utils import orbit_camera  # noqa: E402
from main import MiniCam  # noqa: E402


class Hallo3DGUI(dg_main.GUI):
    def __init__(self, opt):
        super().__init__(opt)
        self.h3d_enable = bool(getattr(opt, "hallo3d_enable", True))
        self.h3d_state = AlignmentState(
            enabled=False,
            num_views=max(1, int(opt.batch_size)),
            focal_idx=0,
            mode=getattr(opt, "hallo3d_align_mode", "extend"),
        )
        self.h3d_detector = LMMDetectorClient(
            getattr(opt, "hallo3d_lmm_url", "http://127.0.0.1:39122"), timeout=90.0
        )
        self.h3d_sd = None  # lazy SDReconsistency
        self.h3d_patched = False
        self.h3d_caption = getattr(opt, "hallo3d_scene_prompt", "") or None
        self.h3d_cg_start = int(
            getattr(opt, "hallo3d_cg_start_ratio", 0.8) * opt.iters
        )
        self.h3d_cg_interval = int(getattr(opt, "hallo3d_cg_interval", 4))
        self.h3d_cg_weight = float(getattr(opt, "hallo3d_cg_weight", 0.1))
        print(
            f"[Hallo3D] align={self.h3d_state.mode} "
            f"L_CG from step {self.h3d_cg_start} every {self.h3d_cg_interval}"
        )

    # ------------------------------------------------------------------ #
    def _ensure_patched(self):
        if not self.h3d_patched and self.guidance_zero123 is not None:
            install_appearance_alignment(
                self.guidance_zero123.unet, self.h3d_state
            )
            self.h3d_patched = True
            print("[Hallo3D] appearance alignment installed on zero123 UNet.")

    def _get_caption(self, image_chw: torch.Tensor) -> str:
        if self.h3d_caption:
            return self.h3d_caption
        try:
            from utils.lmm_client import image_tensor_to_b64png

            response = self.h3d_detector._post(
                {
                    "image_b64": image_tensor_to_b64png(image_chw),
                    "prompt": (
                        "Name the main object in this image with a short noun "
                        "phrase (at most six words). Answer with the phrase only."
                    ),
                }
            )
            self.h3d_caption = response.strip().strip(".\"'") or "an object"
        except Exception:
            self.h3d_caption = "an object"
        print(f"[Hallo3D] scene prompt: {self.h3d_caption}")
        return self.h3d_caption

    @torch.no_grad()
    def _render_views(self, n_views: int, resolution: int):
        raise NotImplementedError  # unused

    def _loss_cg_step(self):
        """Extra optimization step applying L_CG (Eq. 7-8)."""
        resolution = 256
        min_ver = max(
            min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation),
            -80 - self.opt.elevation,
        )
        max_ver = min(
            max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation),
            80 - self.opt.elevation,
        )
        images = []
        for _ in range(max(2, self.h3d_state.num_views)):
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius)
            cam = MiniCam(
                pose,
                resolution,
                resolution,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            out = self.renderer.render(cam)
            images.append(out["image"].unsqueeze(0))
        images = torch.cat(images, dim=0)  # [V,3,H,W], requires grad

        caption_src = (
            self.input_img_torch[0]
            if self.input_img_torch is not None
            else images[0].detach()
        )
        caption = self._get_caption(caption_src)
        pe_neg = self.h3d_detector.query(images[0].detach(), caption)
        if pe_neg is None:
            return  # Eq. 8: skip when semantic structure incomplete
        if self.step % 40 == 0:
            print(f"[Hallo3D] step {self.step} P_E^-: {pe_neg}")

        if self.h3d_sd is None:
            self.h3d_sd = SDReconsistency(
                device=self.device, align_mode=self.h3d_state.mode
            )
        neg_text = (
            f"{pe_neg}, unnatural colors, poor lighting, low quality, "
            "artifacts, smooth texture"
        )
        with torch.no_grad():
            xhat = self.h3d_sd.reconsist(
                images.detach(),
                positive_prompt=f"a 3d rendering of {caption}, high quality",
                negative_prompt=neg_text,
            )
        xhat = F.interpolate(
            xhat, images.shape[-2:], mode="bilinear", align_corners=False
        )
        loss_cg = (
            self.h3d_cg_weight
            * F.mse_loss(images, xhat.detach(), reduction="sum")
            / images.shape[0]
        )
        loss_cg.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------ #
    def train_step(self):
        if not self.h3d_enable:  # plain DreamGaussian baseline
            return super().train_step()
        self._ensure_patched()
        self.h3d_state.num_views = max(1, int(self.opt.batch_size))
        self.h3d_state.focal_idx = 0  # fixed fovy -> Appendix A degenerates
        self.h3d_state.enabled = (
            self.h3d_patched and self.h3d_state.num_views > 1
        )
        try:
            super().train_step()
            if (
                self.step >= self.h3d_cg_start
                and (self.step - self.h3d_cg_start) % self.h3d_cg_interval == 0
            ):
                self._loss_cg_step()
        finally:
            self.h3d_state.enabled = False


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = Hallo3DGUI(opt)
    gui.train(opt.iters)
