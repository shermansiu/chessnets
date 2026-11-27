"""Lc0 attention-body (BT/CF) network for PyTorch.

Original ChessNets implementation. The architecture was derived by exporting Lc0
checkpoints to ONNX, inspecting the graph, and validating outputs numerically
against ONNX Runtime. No lczero-training source code is included.
"""

from __future__ import annotations

import functools
import typing as tp

import attrs
import torch
from torch import nn
from torch.nn import functional as F

from chessnets.models import lc0_policy_map
from chessnets.models.lc0_senet import configuration_lc0_senet as leela_config

BOARD_SIZE = 64
NUM_PLANES = 112
BOARD_EMBED_SIZE = BOARD_SIZE * 12

TensorCallable = tp.Callable[[torch.Tensor], torch.Tensor]


class MaGating(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.mul_gate = nn.Parameter(torch.zeros((BOARD_SIZE, embed_size)))
        self.add_gate = nn.Parameter(torch.ones((BOARD_SIZE, embed_size)))

    def forward(self, x):
        return self.mul_gate.unsqueeze(0) * x + self.add_gate.unsqueeze(0)


@attrs.define
class DeepNorm:
    """DeepNorm. Allows transformers with many layers.

    See https://arxiv.org/abs/2203.00555 for more details.

    Note:
        LayerNorm(x ∗ alpha + f(x)) not implemented as a method, to prevent code from being over-engineered.

        Unlike the paper, in the official repo's training code, alpha is defined as 2N^-0.25, not 2N^0.25 because LayerNorm(x + f(x)*alpha) is used instead
            (pay attention to what alpha is multiplied by)
            and the quotient between the two values is offloaded to be learned by the LayerNorm.
        Note that it hurts training stability somewhat (https://github.com/Ergodice/lczero-training/commit/931aa8adc75f475cd9d088f10dd4ff947b2afbba)
            but is kept for compatibility with existing nets.
        FYI: t82 uses the original paper's DeepNorm, but t3, BT3, and BT4 use the lczero-training version.
            Regardless, they are mathematically equivalent due to the scale invariance of layer normalization.

    Discussion from the Lc0 Discord:
        borg — 12/08/2023 3:08 AM
        In this case there is no issue, the sign change was accompanied with the change in how it is used that you noticed.
        As masterkni6 points out this is followed by a layer normalization that takes care of the overall scaling.
        Doing it this way seems to be more stable in fp16 in my tests.
        (There is a small difference in the following layer normalization epsilon,
        that also needs scaling to be exactly equivalent when the same formula is not used in training).

        Daniel Monroe — 12/08/2023 1:12 PM
        The result is equivalent since both types of normalization we use (layer norm and rmsnorm) are scale invariant

        Regardless of which method is used, you generally get the exact same output (with some minor rounding errors),
        even with the same weights.
    """

    n_embed_layers: int
    alpha: float = attrs.field(init=False)
    beta: float = attrs.field(init=False)
    init_: tp.Callable = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.alpha = (
            2.0 * self.n_embed_layers
        ) ** 0.25  # Original paper uses '** 0.25', lczero-training uses '** -0.25'
        self.beta = (8.0 * self.n_embed_layers) ** -0.25
        self.init_ = functools.partial(nn.init.xavier_normal_, gain=self.beta)

    def apply_norm(self, norm: tp.Callable, fn: tp.Callable, x: torch.Tensor):
        """Apply the layer norm/RMSNorm.

        The original paper uses norm(x ∗ alpha + f(x)) with alpha = 2N^0.25, whereas the lczero-training repo uses
        norm(x + f(x) * alpha) with alpha = 2N^-0.25.
        """
        return norm(x * self.alpha + fn(x))

    def apply_norm_transformed(
        self, norm: tp.Callable, input: torch.Tensor, transformed: torch.Tensor
    ):
        return norm(self.apply_affine(input, transformed))

    def apply_affine(self, input: torch.Tensor, transformed: torch.Tensor):
        return input * self.alpha + transformed


class FFN(nn.Module):
    def __init__(self, config: leela_config.LeelaNetConfig, deep_norm: DeepNorm):
        super().__init__()
        self.dense1 = nn.Linear(
            config.model.embedding_size, config.model.encoder_dff, bias=True
        )
        self.act_fn = config.model.ffn_activation_fn
        self.dense2 = nn.Linear(
            config.model.encoder_dff, config.model.embedding_size, bias=True
        )

        deep_norm.init_(self.dense1.weight)
        deep_norm.init_(self.dense2.weight)

    def forward(self, x):
        return self.dense2(self.act_fn(self.dense1(x)))


class AttentionArcEncodingInputBlock(nn.Module):
    """Input block with arc encoding positional embeddings.

    Used in T82 model.
    """

    def __init__(self, config: leela_config.LeelaNetConfig):
        super().__init__()
        self.register_buffer(
            "pos_encoding",
            torch.from_numpy(lc0_policy_map.arc_position_encoding()),
            persistent=False,
        )
        self.embedding_size = config.model.embedding_size
        self.embed = nn.Linear(
            BOARD_SIZE + NUM_PLANES,
            config.model.embedding_size,
            bias=True,
        )
        self.default_activation_fn = config.model.default_activation_fn
        self.ma_gating = MaGating(config.model.embedding_size)

        nn.init.xavier_normal_(self.embed.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # batch, board_size, n_planes
        x = torch.cat([x, self.pos_encoding.expand(x.size(0), -1, -1)], dim=2)

        # Square embedding
        # https://github.com/Ergodice/lczero-training/blob/c2dc768953b52dd5926476a5e348d9334ffebb7e/tf/tfprocess.py#L1792-L1798
        x = self.default_activation_fn(self.embed(x))
        x = self.ma_gating(x)

        return x


class AttentionInputBlock(nn.Module):
    def __init__(self, config: leela_config.LeelaNetConfig, deep_norm: DeepNorm):
        super().__init__()
        self.preprocess = nn.Linear(
            BOARD_EMBED_SIZE, BOARD_SIZE * config.model.embedding_dense_sz, bias=True
        )
        self.embedding_dense_sz = config.model.embedding_dense_sz
        self.embed = nn.Linear(
            NUM_PLANES + config.model.embedding_dense_sz,
            config.model.embedding_size,
            bias=True,
        )
        self.ffn_activation_fn = config.model.ffn_activation_fn
        self.ln = nn.LayerNorm(config.model.embedding_size, bias=True, eps=0.001)
        self.ma_gating = MaGating(config.model.embedding_size)
        self.deep_norm = deep_norm
        self.ffn = FFN(config, deep_norm)
        self.ffn_ln = nn.LayerNorm(config.model.embedding_size, bias=True, eps=0.001)

        nn.init.xavier_normal_(self.embed.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # batch, board_size, n_planes
        cur_board_embed = (
            x[..., :12].reshape(-1, BOARD_EMBED_SIZE).contiguous()
        )  # The current board
        cur_board_embed = self.preprocess(cur_board_embed).reshape(
            -1, BOARD_SIZE, self.embedding_dense_sz
        )
        x = torch.cat((x, cur_board_embed), dim=2)

        # Square embedding
        # https://github.com/Ergodice/lczero-training/blob/c2dc768953b52dd5926476a5e348d9334ffebb7e/tf/tfprocess.py#L1792-L1798
        x = self.ln(self.ffn_activation_fn(self.embed(x)))
        x = self.ma_gating(x)

        return self.deep_norm.apply_norm(norm=self.ffn_ln, fn=self.ffn, x=x)


class Smolgen(nn.Module):
    """The attention bias used in BT models.

    Geometric Attention Bias (GAB) is the academic name for Smolgen in the official paper.
    """

    def __init__(self, config: leela_config.LeelaNetConfig):
        super().__init__()
        self.n_heads = config.model.encoder_heads
        self.smolgen_hidden_channels = config.model.smolgen_hidden_channels
        self.smolgen_hidden_sz = config.model.smolgen_hidden_sz
        self.smolgen_gen_sz = config.model.smolgen_gen_sz
        self.smolgen_activation_fn = config.model.smolgen_activation_fn
        self.compress = nn.Linear(
            config.model.embedding_size, self.smolgen_hidden_channels, bias=False
        )
        self.hidden_dense = nn.Linear(
            BOARD_SIZE * self.smolgen_hidden_channels, self.smolgen_hidden_sz, bias=True
        )
        self.hidden_ln = nn.LayerNorm(self.smolgen_hidden_sz, bias=True, eps=0.001)
        self.gen_from = nn.Linear(
            self.smolgen_hidden_sz, self.n_heads * self.smolgen_gen_sz, bias=True
        )
        self.gen_ln = nn.LayerNorm(
            self.n_heads * self.smolgen_gen_sz, bias=True, eps=0.001
        )
        self.smol_weight_gen_dense = nn.Linear(
            self.smolgen_gen_sz, BOARD_SIZE * BOARD_SIZE, bias=False
        )

    def forward(self, x):
        x = self.compress(x).view(-1, BOARD_SIZE * self.smolgen_hidden_channels)
        x = self.smolgen_activation_fn(self.hidden_dense(x))
        x = self.hidden_ln(x)

        x = self.smolgen_activation_fn(self.gen_from(x))
        x = self.gen_ln(x).reshape(-1, self.n_heads, self.smolgen_gen_sz)

        return self.smol_weight_gen_dense(x).reshape(
            -1, self.n_heads, BOARD_SIZE, BOARD_SIZE
        )


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        config: leela_config.LeelaNetConfig,
        deepnorm_init_: tp.Callable,
    ):
        super().__init__()
        assert config.model.encoder_d_model % config.model.encoder_heads == 0
        depth = config.model.encoder_d_model // config.model.encoder_heads
        use_bias = not config.model.omit_qkv_biases

        self.depth = depth
        self.num_heads = config.model.encoder_heads
        self.d_model = config.model.encoder_d_model
        self.smolgen = Smolgen(config) if config.model.use_smolgen else None

        emb_size = config.model.embedding_size
        self.q_proj = nn.Linear(emb_size, self.d_model, bias=use_bias)
        self.k_proj = nn.Linear(emb_size, self.d_model, bias=use_bias)
        self.v_proj = nn.Linear(emb_size, self.d_model, bias=use_bias)
        self.out_proj = nn.Linear(self.d_model, emb_size, bias=use_bias)

        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)
        deepnorm_init_(self.v_proj.weight)
        deepnorm_init_(self.out_proj.weight)

    def split_heads(self, vec: torch.Tensor, batch_size: int):
        if self.num_heads < 2:
            return vec
        return vec.view(batch_size, BOARD_SIZE, self.num_heads, self.depth)

    def forward(self, x: torch.Tensor):
        # query, key, and value vectors for self-attention
        # inputs b, 64, sz
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split q, k and v into smaller vectors of size "depth" -- one for each head in multi-head attention
        batch_size = q.size(0)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        if self.smolgen is not None:
            attn_bias = self.smolgen(x)
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_bias
        ).transpose(1, 2)

        if self.num_heads > 1:
            x = x.reshape(batch_size, -1, self.d_model)  # concatenate heads

        # Final dense layer
        return self.out_proj(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        config: leela_config.LeelaNetConfig,
        deep_norm: DeepNorm,
    ):
        super().__init__()
        self.ffn_activation_fn = config.model.ffn_activation_fn
        self.deep_norm = deep_norm
        self.skip_first_ln = config.model.skip_first_ln
        self.mha = MultiHeadedAttention(config, deep_norm.init_)
        self.dropout1 = nn.Dropout(config.model.dropout_rate)
        self.ln1 = nn.LayerNorm(config.model.embedding_size, bias=True, eps=0.001)
        self.ffn = FFN(config, deep_norm)
        self.dropout2 = nn.Dropout(config.model.dropout_rate)
        self.ln2 = nn.LayerNorm(config.model.embedding_size, bias=True, eps=0.001)

    def forward(self, x):
        # Multihead attention
        attn_output = self.mha(x)

        # dropout for weight regularization
        attn_output = self.dropout1(attn_output)
        # skip connection + layernorm
        x = self.deep_norm.apply_affine(input=x, transformed=attn_output)
        if not self.skip_first_ln:
            x = self.ln1(x)

        # feed-forward network
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)

        return self.deep_norm.apply_norm_transformed(
            norm=self.ln2, input=x, transformed=ffn_output
        )


class GatedMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = True,
    ):
        if not batch_first:
            raise NotImplementedError("Only batch_first=True is supported")
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        self.k_head_dim = self.kdim // num_heads
        self.v_head_dim = self.vdim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        if kdim is not None:
            assert self.k_head_dim * num_heads == self.kdim, (
                "kdim must be divisible by num_heads"
            )
        if vdim is not None:
            assert self.v_head_dim * num_heads == self.vdim, (
                "vdim must be divisible by num_heads"
            )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if need_weights:
            raise NotImplementedError("need_weights=True is not supported")

        input_shape = query.size()[:-1]
        q_shape = query.size()[:-1] + (self.num_heads, self.head_dim)
        k_shape = key.size()[:-1] + (self.num_heads, self.head_dim)
        v_shape = value.size()[:-1] + (self.num_heads, self.head_dim)
        gate = self.gate_proj(query)
        query = self.q_proj(query).view(q_shape).transpose(1, 2)
        key = self.k_proj(key).view(k_shape).transpose(1, 2)
        value = self.v_proj(value).view(v_shape).transpose(1, 2)

        # Merge key padding mask and attention mask
        # https://github.com/pytorch/pytorch/blob/a60570acc59c52ff1aa21158292784cd74e63f13/torch/nn/functional.py#L6607-L6619
        if key_padding_mask is not None:
            batch_size = q_shape[0]
            source_length = k_shape[1]
            key_padding_mask = key_padding_mask.view(
                batch_size, 1, 1, source_length
            ).expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.out_proj(attn_output)

        return attn_output


class CrossEncoderLayer(EncoderLayer):
    def __init__(
        self,
        config: leela_config.LeelaNetConfig,
        text_encoder_config: tp.Any,
        deep_norm: DeepNorm,
        should_use_gated_attention: bool = True,
    ):
        super().__init__(config, deep_norm)
        if should_use_gated_attention:
            cross_mha_class = GatedMultiheadAttention
        else:
            cross_mha_class = torch.nn.MultiheadAttention
        # For T5Gemma 2
        if hasattr(text_encoder_config, "text_config"):
            text_encoder_config = text_encoder_config.text_config
        self.cross_mha = cross_mha_class(
            embed_dim=config.model.embedding_size,
            num_heads=config.model.encoder_heads,
            dropout=config.model.dropout_rate,
            bias=not config.model.omit_qkv_biases,
            kdim=text_encoder_config.hidden_size,
            vdim=text_encoder_config.hidden_size,
            batch_first=True,
        )
        zero_init_module(self.cross_mha.out_proj)
        self.cross_mha_ln = nn.LayerNorm(
            config.model.embedding_size, bias=True, eps=0.001
        )
        self.should_use_gated_attention: bool = should_use_gated_attention

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ):
        # Multihead attention
        attn_output = self.mha(x)

        # dropout for weight regularization
        attn_output = self.dropout1(attn_output)
        # skip connection + layernorm
        x = self.deep_norm.apply_affine(input=x, transformed=attn_output)
        if not self.skip_first_ln:
            x = self.ln1(x)

        # Cross-attention with text embeddings
        cross_attn_output = self.cross_mha(
            x,
            text_embeddings,
            text_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.cross_mha_ln(x + cross_attn_output)  # skip connection + layernorm

        # feed-forward network
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)

        return self.deep_norm.apply_norm_transformed(
            norm=self.ln2, input=x, transformed=ffn_output
        )


class ApplyAttentionPolicyMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "idx",
            torch.from_numpy(lc0_policy_map.attention_policy_indices()),
            persistent=False,
        )

    def forward(self, logits, pp_logits):
        logits = torch.cat(
            [logits.view(-1, 64 * 64), pp_logits.view(-1, 8 * 24)], dim=1
        )
        return logits.index_select(1, self.idx)


class AttentionPolicyHead(nn.Module):
    def __init__(
        self,
        config: leela_config.LeelaNetConfig,
        depth: int | None = None,
        opponent=False,
    ):
        super().__init__()
        if depth is None:
            self.depth = config.model.policy_d_model
        else:
            self.depth = depth
        self.opponent = opponent
        self.q_proj = nn.Linear(
            config.model.policy_embedding_size, self.depth, bias=True
        )
        self.k_proj = nn.Linear(
            config.model.policy_embedding_size, self.depth, bias=True
        )
        self.promotion_offsets_proj = nn.Linear(self.depth, 4, bias=False)
        self.activation = leela_config.identity_function
        self.attention_policy_map = ApplyAttentionPolicyMap()

        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.promotion_offsets_proj.weight)

    def forward(self, x):
        # reverse the tokens along the square (second) dimension to get the opponent's perspective
        x = torch.flip(x, (1,)) if self.opponent else x

        # create queries and keys for policy self-attention
        queries = self.q_proj(x)
        keys = self.k_proj(x)

        # POLICY SELF-ATTENTION: self-attention weights are interpreted as from->to policy
        # Bx64x64 (from 64 queries, 64 keys)
        policy_attn_logits = queries @ keys.transpose(-2, -1)

        # PAWN PROMOTION: create promotion logits using scalar offsets generated from the promotion-rank keys
        # constant for scaling
        sqrt_attn_dim = queries.shape[-1] ** 0.5
        promotion_x = keys[:, -8:]  # Backrank embeddings
        # queen, rook, bishop, knight order
        promotion_x = self.promotion_offsets_proj(promotion_x)
        promotion_x = promotion_x.permute(0, 2, 1) * sqrt_attn_dim  # Bx4x8
        # knight offset is added to the other three
        promotion_x = promotion_x[:, :3] + promotion_x[:, 3:4]

        # q, r, and b promotions are offset from the default promotion logit (knight)
        # default traversals from penultimate rank to promotion rank
        n_promo_logits = policy_attn_logits[:, -16:-8, -8:]
        q_promo_logits = (n_promo_logits + promotion_x[:, 0:1]).unsqueeze(3)  # Bx8x8x1
        r_promo_logits = (n_promo_logits + promotion_x[:, 1:2]).unsqueeze(3)  # Bx8x8x1
        b_promo_logits = (n_promo_logits + promotion_x[:, 2:3]).unsqueeze(3)  # Bx8x8x1
        del promotion_x
        promotion_logits = torch.concat(
            [q_promo_logits, r_promo_logits, b_promo_logits], dim=3
        )  # Bx8x8x3
        del q_promo_logits, r_promo_logits, b_promo_logits
        # logits now alternate a7a8q,a7a8r,a7a8b,...,
        promotion_logits = promotion_logits.view(-1, 8, 24)

        # scale the logits by dividing them by sqrt(d_model) to stabilize gradients
        # Bx8x24 (8 from-squares, 3x8 promotions)
        promotion_logits = promotion_logits / sqrt_attn_dim
        # Bx64x64 (64 from-squares, 64 to-squares)
        policy_attn_logits = policy_attn_logits / sqrt_attn_dim

        # APPLY POLICY MAP: output becomes Bx1856
        x = self.attention_policy_map(policy_attn_logits, promotion_logits)

        if self.activation is not None:
            x = self.activation(x)

        return x


class AttentionValueHead(nn.Module):
    def __init__(
        self, config: leela_config.LeelaNetConfig, wdl=True, use_err=True, use_cat=True
    ):
        super().__init__()
        self.embed = nn.Linear(
            config.model.embedding_size, config.model.value_embedding_size, bias=True
        )
        self.default_activation = config.model.default_activation_fn
        self.dense = nn.Linear(
            BOARD_SIZE * config.model.value_embedding_size, 128, bias=True
        )
        self.val_proj = nn.Linear(128, 3 if wdl else 1, bias=True)
        self.val_err_proj = nn.Linear(128, 1, bias=True) if use_err else None
        self.val_cat_proj = (
            nn.Linear(128, config.model.categorical_value_buckets, bias=True)
            if use_cat
            else None
        )

        self.wdl = wdl
        self.use_err = use_err
        self.use_cat = use_cat
        self.categorical_value_buckets = config.model.categorical_value_buckets

        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.xavier_normal_(self.val_proj.weight)
        if use_err:
            nn.init.xavier_normal_(self.val_err_proj.weight)
        if use_cat:
            nn.init.xavier_normal_(self.val_cat_proj.weight)

    def forward(self, x):
        x = self.default_activation(self.embed(x)).flatten(start_dim=1)
        x = self.default_activation(self.dense(x))

        # WDL head
        if self.wdl:
            value = self.val_proj(x)
        else:
            value = torch.tanh(self.val_proj(x))

        if self.use_err:
            # Shouldn't be more than 1
            value_err = torch.sigmoid(self.val_err_proj(x))
        else:
            value_err = None

        if self.use_cat:
            value_cat = self.val_cat_proj(x)
        else:
            value_cat = None

        return value, value_err, value_cat


class AttentionMovesLeftHead(nn.Module):
    def __init__(self, config: leela_config.LeelaNetConfig):
        super().__init__()
        self.embed = nn.Linear(
            config.model.embedding_size,
            config.model.moves_left_embedding_size,
            bias=True,
        )
        self.activation = config.model.default_activation_fn
        self.dense1 = nn.Linear(
            config.model.moves_left_embedding_size * BOARD_SIZE, 128, bias=True
        )
        self.dense2 = nn.Linear(128, 1, bias=True)

        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.dense1.weight)
        nn.init.xavier_normal_(self.dense2.weight)

    def forward(self, x):
        x = self.activation(self.embed(x)).flatten(start_dim=1)
        x = self.activation(self.dense1(x))

        return torch.relu(x=self.activation(self.dense2(x)))


class AttentionOutcomeHead(nn.Module):
    def __init__(self, config: leela_config.LeelaNetConfig, num_categories: int = 12):
        super().__init__()
        self.embed = nn.Linear(
            config.model.embedding_size, config.model.value_embedding_size, bias=True
        )
        self.default_activation = config.model.default_activation_fn
        self.dense = nn.Linear(
            BOARD_SIZE * config.model.value_embedding_size, 128, bias=True
        )
        self.outcome_proj = nn.Linear(128, num_categories, bias=True)

        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.xavier_normal_(self.outcome_proj.weight)

    def forward(self, x):
        x = self.default_activation(self.embed(x)).flatten(start_dim=1)
        x = self.default_activation(self.dense(x))

        outcome = self.outcome_proj(x)

        return outcome


class SwiGLU(torch.nn.Module):
    """Implement SwiGLU FFN.

    Note: for the GPT-NeoX module we're using,

    Sources:
    - https://github.com/huggingface/transformers/blob/6dfd561d9cd722dfc09f702355518c6d09b9b4e3/src/transformers/models/dinov2/modeling_dinov2.py#L380-L394
    - https://dev.to/mshojaei77/swiglu-the-ffn-upgrade-i-use-to-get-free-performance-33jc
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        is_intermediate_size_for_swiglu: bool = False,
    ):
        super().__init__()
        if is_intermediate_size_for_swiglu:
            ffn_size = intermediate_size
        else:
            # The SwiGLU paper recommends that the hidden dimension of a SwiGLU FFN be 2/3
            # that of the FFN dimension to keep parameter sizes comparable
            # See the last paragraph of section 2: https://arxiv.org/abs/2002.05202
            ffn_size = int(2 * intermediate_size / 3)
            # Rounds up to the nearest multiple of 8.
            ffn_size = (ffn_size + 7) // 8 * 8
        self.weights_in = nn.Linear(hidden_size, 2 * ffn_size, bias=True)
        self.weights_out = nn.Linear(ffn_size, hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


def zero_init_module(module: nn.Module) -> nn.Module:
    """Zero-initializes all parameters in a module.

    Returns the module in case it hasn't been assigned to a variable yet.
    """
    with torch.no_grad():
        for param in module.parameters():
            nn.init.zeros_(param)
    return module
