"""Multi-view Appearance Alignment (Hallo3D, Sec. 3.2).

AAttn(Q, K_i, V_i) = Softmax(Q K_i^T / sqrt(d)) V_i   (Eq. 4)

The self-attention layers of the 2D diffusion UNet are patched so that,
when alignment is enabled, the key/value features of every view are taken
from a single "focal view" i while queries remain per-view. This turns
self-attention into a cross-attention onto the focal view, aligning color
and texture across the batch of rendered views.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AlignmentState:
    """Shared, mutable switch consulted by all patched attention layers.

    The UNet input batch is laid out as G contiguous groups of `num_views`
    view-images (e.g. [cond x B_views, uncond x B_views] for CFG, G=2).
    Alignment is applied independently within each group.

    mode:
      "replace" - K/V come from the focal view only (Eq. 4 literal)
      "extend"  - K/V are the concatenation of each view's own features and
                  the focal view's features (mutual attention); keeps every
                  view's own structure anchor while sharing appearance
    """

    enabled: bool = False
    num_views: int = 4
    focal_idx: int = 0
    mode: str = "replace"


class AppearanceAlignedAttnProcessor:
    """Drop-in diffusers attention processor implementing AAttn.

    Behaves exactly like AttnProcessor2_0 for cross-attention and for
    self-attention while `state.enabled` is False or this layer is not
    selected for alignment (`active=False`).
    """

    def __init__(self, state: AlignmentState, active: bool = True):
        self.state = state
        self.active = active

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        is_self_attention = encoder_hidden_states is None

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        kv_source = encoder_hidden_states
        state = self.state
        if (
            is_self_attention
            and self.active
            and state.enabled
            and state.num_views > 1
            and batch_size % state.num_views == 0
        ):
            # replace or extend every view's K/V source with the focal view's
            n_groups = batch_size // state.num_views
            seq_len, channels = kv_source.shape[1], kv_source.shape[2]
            focal = kv_source.view(n_groups, state.num_views, seq_len, channels)[
                :, state.focal_idx
            ]  # [G, L, C]
            focal = (
                focal.unsqueeze(1)
                .expand(n_groups, state.num_views, seq_len, channels)
                .reshape(batch_size, seq_len, channels)
            )
            if state.mode == "extend":
                kv_source = torch.cat([kv_source, focal], dim=1)
            else:
                kv_source = focal

        key = attn.to_k(kv_source)
        value = attn.to_v(kv_source)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def install_appearance_alignment(
    unet, state: AlignmentState, layers: str = "all"
) -> None:
    """Patch attention layers of `unet` with the aligned processor.

    layers:
      "all"     - align every self-attention layer (Eq. 4 literal)
      "decoder" - align only up_blocks self-attention; encoder/mid layers
                  keep vanilla attention so per-view structure formation is
                  not disturbed (MasaCtrl-style restriction)
    """
    procs = {}
    for name in unet.attn_processors.keys():
        active = layers == "all" or name.startswith("up_blocks")
        procs[name] = AppearanceAlignedAttnProcessor(state, active=active)
    unet.set_attn_processor(procs)


def select_focal_view(
    fovy_rad: torch.Tensor, threshold_deg: float
) -> int:
    """Focal view selection (Appendix A).

    The first view whose Fovy exceeds 120% of the baseline default (the
    threshold, in degrees) becomes the focal view; if none qualifies, the
    view with the largest Fovy is used, as a wider vertical view captures
    more object details.
    """
    fovy_deg = fovy_rad * 180.0 / torch.pi
    exceed = (fovy_deg > threshold_deg).nonzero(as_tuple=True)[0]
    if len(exceed) > 0:
        return int(exceed[0].item())
    return int(fovy_deg.argmax().item())
