"""Appearance alignment patch for LDM (CompVis) UNets, used by the
Zero-1-to-3 / Stable-Zero123 checkpoints loaded through
extern/ldm_zero123 (threestudio's zero123 guidances).

Same semantics as utils/appearance_attn.py but implemented by wrapping the
`forward` of every self-attention module (BasicTransformerBlock.attn1):
when alignment is enabled, the attention context (the K/V source) is
replaced ("replace", Eq. 4 literal) or extended ("extend", mutual
attention) with the focal view's hidden states.
"""

import torch

from .appearance_attn import AlignmentState


def _make_aligned_forward(orig_forward, state: AlignmentState):
    def forward(x, context=None, mask=None, **kwargs):
        if (
            context is None  # self-attention only
            and state.enabled
            and state.num_views > 1
            and x.shape[0] % state.num_views == 0
        ):
            batch_size, seq_len, channels = x.shape
            n_groups = batch_size // state.num_views
            focal = x.view(n_groups, state.num_views, seq_len, channels)[
                :, state.focal_idx
            ]
            focal = (
                focal.unsqueeze(1)
                .expand(n_groups, state.num_views, seq_len, channels)
                .reshape(batch_size, seq_len, channels)
            )
            if state.mode == "extend":
                context = torch.cat([x, focal], dim=1)
            else:
                context = focal
            return orig_forward(x, context=context, mask=mask, **kwargs)
        return orig_forward(x, context=context, mask=mask, **kwargs)

    return forward


def install_ldm_appearance_alignment(model, state: AlignmentState) -> int:
    """Wrap every `*.attn1` module of an LDM UNet. Returns #patched."""
    n = 0
    for name, module in model.named_modules():
        if name.endswith("attn1") and hasattr(module, "to_q"):
            if getattr(module, "_hallo3d_patched", False):
                continue
            module.forward = _make_aligned_forward(module.forward, state)
            module._hallo3d_patched = True
            n += 1
    return n
