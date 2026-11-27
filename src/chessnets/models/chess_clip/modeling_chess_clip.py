# SPDX-License-Identifier: MIT AND Apache-2.0
# Portions adapted from OpenCLIP (https://github.com/mlfoundations/open_clip),
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini,
# Rohan Taori, Achal Dave, Vaishaal Shankar, John Miller, Hongseok Namkoong,
# Hannaneh Hajishirzi, Ali Farhadi, Ludwig Schmidt. See LICENSES/OpenCLIP-MIT.txt.
# Portions adapted from the ChessCLIP fork in ChessGPT
# (https://github.com/waterhorse1/ChessGPT), Copyright 2022-2023 Xidong Feng,
# Yicheng Luo, Ziyan Wang. See LICENSES/ChessGPT-Apache-2.0.txt.
# Modified for ChessNets by Sherman Siu; modifications are under Apache-2.0.

"""ChessCLIP model implementation adapted from OpenCLIP with FiLM-modified ResNet vision tower.

The original implementation is a fork of open_clip, which is available at https://github.com/waterhorse1/ChessGPT/tree/main/chessclip

This file contains the changes made to open_clip to implement the ChessCLIP model, so that it can be loaded without installing the forked repository.
"""

import dataclasses
import logging
import os
import pathlib
import typing as tp
import warnings
from collections import OrderedDict

import huggingface_hub
import numpy as np
import torch
import torch.jit
from torch import nn
from torch.nn import functional as F

from chessnets._optional import import_optional

open_clip = import_optional("open_clip", "chess-clip")
freeze_batch_norm_2d = import_optional(
    "open_clip.utils", "chess-clip"
).freeze_batch_norm_2d


class ChessCLIPModel(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        chess_vision_cfg: open_clip.CLIPVisionCfg,
        text_cfg: open_clip.CLIPTextCfg | None = None,
        pretrain_model=None,
        quick_gelu: bool = False,
        cast_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(
            embed_dim, chess_vision_cfg, quick_gelu, cast_dtype
        )
        if text_cfg is None and pretrain_model:
            import copy

            self.text = copy.deepcopy(pretrain_model.text)
            for name, params in self.text.named_parameters():
                if "10" in name or "11" in name:
                    pass
                else:
                    params.requires_grad = False
            del pretrain_model
            logging.debug("finish copy")
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = None

        logging.debug(
            "# Trainable parameters:",
            sum([p.numel() for p in self.parameters() if p.requires_grad]),
        )

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, state, normalize: bool = False):
        image, action = state
        features = self.visual(image, action)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "Waterhorse/ChessCLIP",
        **kwargs: tp.Any,
    ):
        """Load the published ChessCLIP checkpoint by repository ID."""
        if pretrained_model_name_or_path != "Waterhorse/ChessCLIP":
            raise ValueError(
                "ChessCLIPModel currently supports only 'Waterhorse/ChessCLIP'."
            )
        return load_pretrained_chessclip(**kwargs)


@dataclasses.dataclass
class CLIPVisionCfg:
    layers: tuple[int, int, int, int] | int = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: tuple[int, int] | int = 224

    ls_init_value: float | None = None  # layer scale initial value
    patch_dropout: float = 0.0  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = "learnable"
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = "tok"
    output_tokens: bool = False
    act_kwargs: dict | None = None
    norm_kwargs: dict | None = None

    in_channels: int = 3
    film: bool = False
    stride: int = 1
    action_dim: int = 1858

    # Custom attention block settings
    block_type: str | None = (
        None  # attention block type ('default', 'custom'), auto-selects 'custom' if any below features enabled
    )
    qk_norm: bool = False  # apply layer norm to q and k in attention
    scaled_cosine_attn: bool = False  # use scaled cosine attention
    scale_heads: bool = (
        False  # learnable head-specific scale applied to attention logits
    )
    scale_attn_inner: bool = (
        False  # apply layer norm on attention context, before output projection
    )
    scale_attn: bool = False  # apply layer norm after full attention block
    scale_fc: bool = False  # apply layer norm in MLP block

    timm_model_name: str | None = (
        None  # a valid model name overrides layers, width, patch_size
    )
    timm_model_pretrained: bool = (
        False  # use (imagenet) pretrained weights for named model
    )
    timm_pool: str = (
        "avg"  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    )
    timm_proj: str = (
        "linear"  # linear projection for timm model output ('linear', 'mlp', '')
    )
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.0  # head dropout
    timm_drop_path: float | None = None  # backbone stochastic depth


def build_model_from_openai_state_dict_chess(
    state_dict: dict,
    quick_gelu=True,
    cast_dtype=torch.float16,
    chess_vision_cfg=None,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
        )
    )

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = open_clip.CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    openai_model = open_clip.CustomTextCLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    state_dict = open_clip.model.convert_to_custom_text_state_dict(state_dict)

    open_clip.convert_weights_to_fp16(
        openai_model
    )  # OpenAI state dicts are partially converted to float16
    openai_model.load_state_dict(state_dict)

    model = ChessCLIPModel(
        embed_dim,
        chess_vision_cfg=chess_vision_cfg,
        text_cfg=None,
        pretrain_model=openai_model,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    # convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    return model.eval()


def load_openai_model(
    name: str,
    precision: str | None = None,
    device: str | torch.device | None = None,
    jit: bool = True,
    cache_dir: str | None = None,
    chess_vision_cfg=None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = "fp32" if device == "cpu" else "fp16"

    if open_clip.get_pretrained_url(name, "openai"):
        model_path = open_clip.download_pretrained_from_url(
            open_clip.get_pretrained_url(name, "openai"), cache_dir=cache_dir
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {open_clip.list_openai_models()}"
        )

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead",
                stacklevel=2,
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        # Build a non-jit model from the OpenAI jitted model state dict
        cast_dtype = open_clip.get_cast_dtype(precision)
        if chess_vision_cfg is not None:
            model = build_model_from_openai_state_dict_chess(
                state_dict or model.state_dict(),
                cast_dtype=cast_dtype,
                chess_vision_cfg=chess_vision_cfg,
            )
        else:
            try:
                model = open_clip.build_model_from_openai_state_dict(
                    state_dict or model.state_dict(), cast_dtype=cast_dtype
                )
            except KeyError:
                sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
                model = open_clip.build_model_from_openai_state_dict(
                    sd, cast_dtype=cast_dtype
                )

        # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
        model = model.to(device)
        if precision.startswith("amp") or precision == "fp32":
            model.float()
        elif precision == "bf16":
            open_clip.convert_weights_to_lp(model, dtype=torch.bfloat16)

        return model

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 (typically for CPU)
    if precision == "fp32":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model


def create_model(
    model_name: str,
    pretrained: str | None = None,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: float | None = None,
    pretrained_image: bool = False,
    pretrained_hf: bool = True,
    cache_dir: str | None = None,
):
    model_name = model_name.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names
    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == "openai":
        logging.info(f"Loading pretrained {model_name} from OpenAI.")
        model_cfg = open_clip.get_model_config(model_name)
        if "chess_vision_cfg" in model_cfg.keys():
            chess_vision_cfg = model_cfg["chess_vision_cfg"]
        else:
            chess_vision_cfg = None
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            jit=jit,
            cache_dir=cache_dir,
            chess_vision_cfg=chess_vision_cfg,
        )
    else:
        model_cfg = open_clip.get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f"Loaded {model_name} model config.")
        else:
            logging.error(
                f"Model config for {model_name} not found; available models {open_clip.list_models()}."
            )
            raise RuntimeError(f"Model config for {model_name} not found.")

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if pretrained_image:
            if "timm_model_name" in model_cfg.get("vision_cfg", {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg["vision_cfg"]["timm_model_pretrained"] = True
            else:
                raise ValueError(
                    "pretrained image towers currently only supported for timm models"
                )

        cast_dtype = open_clip.get_cast_dtype(precision)
        custom_text = (
            model_cfg.pop("custom_text", False)
            or force_custom_text
            or ("hf_model_name" in model_cfg.get("text_cfg", {}))
        )

        if custom_text:
            if "hf_model_name" in model_cfg.get("text_cfg", {}):
                model_cfg["text_cfg"]["hf_model_pretrained"] = pretrained_hf
            model = open_clip.CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = open_clip.CLIP(**model_cfg, cast_dtype=cast_dtype)

        pretrained_cfg = {}
        if pretrained:
            checkpoint_path = ""
            pretrained_cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = open_clip.download_pretrained(
                    pretrained_cfg, cache_dir=cache_dir
                )
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f"Loading pretrained {model_name} weights ({pretrained}).")
                open_clip.load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                    f"Available pretrained tags ({open_clip.list_pretrained_tags_by_model(model_name)}."
                )
                logging.warning(error_str)
                raise RuntimeError(error_str)

        model.to(device=device)
        if precision in ("fp16", "bf16"):
            open_clip.convert_weights_to_lp(
                model, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
            )

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = (
            pretrained_cfg.get("mean", None) or open_clip.OPENAI_DATASET_MEAN
        )
        model.visual.image_std = (
            pretrained_cfg.get("std", None) or open_clip.OPENAI_DATASET_STD
        )

        if jit:
            model = torch.jit.script(model)

    return model


class FiLMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(
            inplanes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes * self.expansion, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * FiLMBasicBlock.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        out = gamma * out + beta

        out += identity
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class FiLMModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(
        self,
        layers,
        output_dim,
        heads,
        image_size=224,
        width=64,
        in_channels=3,
        stride=1,
        action_dim=1858,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        self.width = width

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            in_channels, width // 2, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=stride)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=stride)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=stride)

        self.film = nn.Sequential(
            nn.Linear(action_dim, width * 4),
            nn.ReLU(inplace=True),
            nn.Linear(
                width * 4,
                (
                    width * layers[0]
                    + width * 2 * layers[1]
                    + width * 4 * layers[2]
                    + width * 8 * layers[3]
                )
                * 2,
            ),
        )

        embed_dim = width * 8  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [FiLMBasicBlock(self._inplanes, planes, stride)]

        self._inplanes = planes * FiLMBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(FiLMBasicBlock(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, (
            "partial locking not currently supported for this model"
        )
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # Not supported for this ResNet tower, as in OpenCLIP's ModifiedResNet.
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x, action):
        x = self.stem(x)
        film = self.film(action)
        offset = 0
        for i in range(len(self.layer1)):
            x = self.layer1[i](
                x,
                film[:, offset : offset + self.width],
                film[:, offset + self.width : offset + self.width * 2],
            )
            offset += self.width * 2

        for i in range(len(self.layer2)):
            x = self.layer2[i](
                x,
                film[:, offset : offset + self.width * 2],
                film[:, offset + self.width * 2 : offset + self.width * 4],
            )
            offset += self.width * 4

        for i in range(len(self.layer3)):
            x = self.layer3[i](
                x,
                film[:, offset : offset + self.width * 4],
                film[:, offset + self.width * 4 : offset + self.width * 8],
            )
            offset += self.width * 8

        for i in range(len(self.layer4)):
            x = self.layer4[i](
                x,
                film[:, offset : offset + self.width * 8],
                film[:, offset + self.width * 8 : offset + self.width * 16],
            )
            offset += self.width * 16

        x = self.attnpool(x)

        return x


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
    quick_gelu: bool = False,
    cast_dtype: torch.dtype | None = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = open_clip.model.QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = open_clip.model.TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
        act_layer = (
            nn.GELU
        )  # so that text transformer doesn't use QuickGELU w/ timm models
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        if vision_cfg.film:
            visual = FiLMModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width,
                in_channels=vision_cfg.in_channels,
                stride=vision_cfg.stride,
                action_dim=vision_cfg.action_dim,
            )
        else:
            visual = open_clip.model.ModifiedResNet(
                layers=list(vision_cfg.layers),
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width,
            )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = (
            open_clip.model.LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else open_clip.model.LayerNorm
        )
        visual = open_clip.model.VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            pool_type=vision_cfg.pool_type,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


open_clip.pretrained._PRETRAINED["chessclip-quickgelu"] = open_clip.pretrained._VITB32
open_clip.add_model_config(
    pathlib.Path(__file__)
    .parent.joinpath("model_configs/chessclip-quickgelu.json")
    .resolve()
)


def load_pretrained_chessclip(
    model_name: str = "chessclip-quickgelu",
    pretrained: str = "openai",
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    jit: bool = False,
    cache_dir: str | None = None,
):
    model = create_model(
        model_name,
        pretrained=pretrained,
        precision=precision,
        device=device,
        jit=jit,
        cache_dir=cache_dir,
    )
    if model_name == "chessclip-quickgelu":
        chess_clip_checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id="Waterhorse/ChessCLIP", filename="epoch_latest.pt"
        )
        open_clip.load_checkpoint(model, chess_clip_checkpoint_path, weights_only=False)
    return model


__all__ = [
    "CLIPVisionCfg",
    "ChessCLIPModel",
    "FiLMModifiedResNet",
    "build_model_from_openai_state_dict_chess",
    "create_model",
    "load_openai_model",
    "load_pretrained_chessclip",
]
