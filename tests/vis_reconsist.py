"""Offline visualization of Module C (Prompt-Enhanced Re-consistency).

Takes 4 rendered views of a trained asset, runs the full detection +
DDIM-inversion + prompt-enhanced re-sampling path of the guidance, and
saves [x_0 | x_hat_0] side by side. Use to verify that x_hat_0 preserves
semantics while correcting appearance (Eq. 6).

Usage:
  python tests/vis_reconsist.py --frames-dir <it1200-test dir> \
      --prompt "..." --out /tmp/reconsist_vis.png [--gpu 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/data222/hongbo.wang/threestudio")

import threestudio  # noqa: E402
from threestudio.utils.config import parse_structured  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frames", type=int, nargs=4, default=[0, 30, 60, 90])
    parser.add_argument("--out", default="/tmp/reconsist_vis.png")
    parser.add_argument(
        "--lmm-url", default="http://127.0.0.1:39122"
    )
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    # register the extension
    import importlib.util

    ext = "/data222/hongbo.wang/threestudio/custom/threestudio-hallo3d"
    spec = importlib.util.spec_from_file_location(
        "threestudio-hallo3d",
        ext + "/__init__.py",
        submodule_search_locations=[ext],
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["threestudio-hallo3d"] = m
    spec.loader.exec_module(m)

    guidance_cls = threestudio.find("hallo3d-sd-guidance")
    cfg = parse_structured(
        guidance_cls.Config,
        {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1-base",
            "guidance_scale": 100.0,
            "lmm_server_url": args.lmm_url,
            "cg_ddim_steps": args.ddim_steps,
            "cg_strength": args.strength,
            "cg_start_step": 0,
        },
    )
    guidance = guidance_cls(cfg)

    # load 4 views
    imgs = []
    for i in args.frames:
        im = Image.open(Path(args.frames_dir) / f"{i}.png").convert("RGB")
        # threestudio test frames may contain [rgb | extra] panels; crop square
        w, h = im.size
        if w > h:
            im = im.crop((0, 0, h, h))
        imgs.append(im.resize((512, 512)))
    x0 = (
        torch.stack(
            [torch.from_numpy(np.array(im)).float() / 255.0 for im in imgs]
        )
        .permute(0, 3, 1, 2)
        .to(guidance.device)
    )

    # Module B
    pe_neg = guidance.detector.query(x0[0], args.prompt)
    print("P_E^- =", repr(pe_neg))
    neg_text = (
        f"{pe_neg}, {cfg.general_negative_prompt}"
        if pe_neg
        else cfg.general_negative_prompt
    )

    # simple non-view-dependent embeddings
    cond = guidance.encode_text(args.prompt).expand(4, -1, -1)
    neg = guidance.encode_text(neg_text)

    guidance.align_state.enabled = True
    guidance.align_state.num_views = 4
    guidance.align_state.focal_idx = 0
    with torch.no_grad():
        latents = guidance.encode_images(x0)
        xhat0 = guidance.reconsist_images(latents, cond, neg)
    guidance.align_state.enabled = False

    rows = []
    for row_t in [x0, xhat0]:
        arr = (
            (row_t.clamp(0, 1) * 255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        rows.append([Image.fromarray(a).resize((256, 256)) for a in arr])
    canvas = Image.new("RGB", (256 * 4, 256 * 2))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            canvas.paste(im, (c * 256, r * 256))
    canvas.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
