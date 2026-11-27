"""Load PyTorch weights from a gzipped Lc0 protobuf checkpoint.

This ChessNets implementation contains original code; its checkpoint-loading
and conversion logic was adapted from the following interoperability
references:

* https://github.com/LeelaChessZero/lc0/blob/master/src/neural/onnx/converter.cc
* https://github.com/Ergodice/lczero-training/blob/master/tf/net.py
* https://github.com/Ergodice/lczero-training/blob/master/tf/tfprocess.py
"""

import numpy as np
import torch
import warnings
from . import net_pb2 as pb
from chessnets.models.lc0 import layers_lc0_seresnet
from chessnets.models.lc0 import modeling_lc0_seresnet as sln
from chessnets.models.lc0 import layers_lc0_chessformer as layers
from chessnets.models.lc0 import modeling_lc0_chessformer as aln


def inverse_permutation(permutation: np.ndarray):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(permutation.size)
    return inv


def pb_layer_to_tensor(layer: pb.Weights.Layer) -> np.ndarray:
    params = np.frombuffer(layer.params, np.uint16).astype(np.float32)
    params /= 0xFFFF
    return params * (layer.max_val - layer.min_val) + layer.min_val


def load_param(
    layer: pb.Weights.Layer,
    pt_layer: torch.nn.Parameter,
    permutation: list[int] | None = None,
    permissive: bool = False,
):
    """Load a PyTorch parameter from a Protobuf."""
    if len(layer.params) == 0:
        error_message = f"Protobuf layer for {pt_layer} has no params!"
        if permissive:
            warnings.warn(error_message)
            return
        raise ValueError(error_message)

    weights = pb_layer_to_tensor(layer)
    shape = pt_layer.shape
    if permutation is not None:
        perm = np.asanyarray(permutation)
        inv = inverse_permutation(perm)
        inv_shape = [shape[i] for i in inv]
        weights = weights.reshape(inv_shape).transpose(*permutation)
    with torch.no_grad():
        pt_layer.copy_(torch.from_numpy(weights).reshape(shape))


def load_batchnorm(convblock: pb.Weights.ConvBlock, pt_batchnorm: torch.nn.BatchNorm2d):
    """Load a squeeze-excitation convolution block from a Protobuf."""
    load_param(convblock.bn_gammas, pt_batchnorm.weight)
    load_param(convblock.bn_betas, pt_batchnorm.bias)
    load_param(convblock.bn_means, pt_batchnorm.running_mean)
    load_param(convblock.bn_stddivs, pt_batchnorm.running_var)
    # with torch.no_grad():
    #     pt_batchnorm.running_var.square_()


def load_conv_block(
    convblock: pb.Weights.ConvBlock,
    pt_convblock: layers_lc0_seresnet.ConvBatchNormReLU,
):
    """Denormalize a squeeze-excitation convolution block from a Protobuf."""
    load_plain_conv(convblock, pt_convblock.conv_layer, bias=False)
    load_batchnorm(convblock, pt_convblock.batchnorm)


def load_plain_conv(
    convblock: pb.Weights.ConvBlock, pt_convblock: torch.nn.Conv2d, bias: bool = False
):
    """Denormalize a plain convolution from a Protobuf."""
    load_param(convblock.weights, pt_convblock.weight)
    if bias:
        load_param(convblock.biases, pt_convblock.bias)


def load_se_unit(
    convblock: pb.Weights.SEunit, pt_convblock: layers_lc0_seresnet.SqueezeExcite
):
    """Denormalize a SE-unit from a Protobuf."""
    load_param(convblock.w1, pt_convblock.squeeze[0].weight)
    load_param(convblock.b1, pt_convblock.squeeze[0].bias)
    load_param(convblock.w2, pt_convblock.expand.weight)
    load_param(convblock.b2, pt_convblock.expand.bias)


def load_residual_layers(
    residual_layers: list[pb.Weights.Residual], pt_residual_layers: torch.nn.Sequential
):
    if len(residual_layers) != len(pt_residual_layers):
        raise ValueError(
            f"Different numbers of residual layers: {len(residual_layers)} != {len(pt_residual_layers)}"
        )
    for reslay, pt_reslay in zip(residual_layers, pt_residual_layers):
        load_plain_conv(reslay.conv1, pt_reslay.conv1, bias=False)
        load_batchnorm(reslay.conv1, pt_reslay.batch_norm1)
        load_plain_conv(reslay.conv2, pt_reslay.conv2, bias=False)
        load_batchnorm(reslay.conv2, pt_reslay.batch_norm2)
        load_se_unit(reslay.se, pt_reslay.squeeze_excite)


def load_conv_policy_head(
    weights: pb.Weights, pt_policy_head: layers_lc0_seresnet.ConvPolicyHead
):
    load_conv_block(weights.policy1, pt_policy_head.conv_block)
    load_plain_conv(weights.policy, pt_policy_head.conv, bias=True)


def load_conv_value_head(
    weights: pb.Weights,
    pt_value_head: layers_lc0_seresnet.ConvValueHead,
):
    load_conv_block(weights.value, pt_value_head.conv_block)
    load_param(weights.ip1_val_w, pt_value_head.fc2.weight)
    load_param(weights.ip1_val_b, pt_value_head.fc2.bias)
    load_param(weights.ip2_val_w, pt_value_head.fc_out.weight)
    load_param(weights.ip2_val_b, pt_value_head.fc_out.bias)


def fix_conv_50_rule_move(convblock: layers_lc0_seresnet.ConvBatchNormReLU):
    """Rescale the 50-move rule weights.

    Normalization is done during training but not inference.

    https://github.com/LeelaChessZero/lczero-training/blob/fae46e6892fd1a7fc1e0b8bff38b538b4626dff2/tf/tfprocess.py#L678
    """
    with torch.no_grad():
        convblock.conv_layer.weight[:, 109] *= 99


def fix_attn_50_rule_move(attn_input_block: layers.AttentionInputBlock):
    """Rescale the 50-move rule weights.

    Normalization is done during training but not inference.

    https://github.com/LeelaChessZero/lczero-training/blob/fae46e6892fd1a7fc1e0b8bff38b538b4626dff2/tf/tfprocess.py#L678
    """
    with torch.no_grad():
        attn_input_block.embed.weight[:, 109] *= 99


def load_se_model(
    weights: pb.Weights,
    pt_model: sln.Lc0SeResNetModel,
    is_hectoplies_format: bool = False,
):
    load_conv_block(weights.input, pt_model.input_block)
    if not is_hectoplies_format:
        fix_conv_50_rule_move(pt_model.input_block)
    load_residual_layers(weights.residual, pt_model.residual_blocks)
    load_conv_policy_head(weights, pt_model.policy_head)
    load_conv_value_head(weights, pt_model.value_head)


def load_ffn(ffn: pb.Weights.FFN, pt_ffn: layers.FFN):
    load_param(ffn.dense1_w, pt_ffn.dense1.weight)
    load_param(ffn.dense1_b, pt_ffn.dense1.bias)
    load_param(ffn.dense2_w, pt_ffn.dense2.weight)
    load_param(ffn.dense2_b, pt_ffn.dense2.bias)


def load_attn_input_block(
    weights: pb.Weights,
    pt_input_block: layers.AttentionInputBlock | layers.AttentionArcEncodingInputBlock,
):
    if isinstance(pt_input_block, layers.AttentionArcEncodingInputBlock):
        load_arc_encoding_attn_input_block(weights, pt_input_block)
    else:
        load_new_attn_input_block(weights, pt_input_block)


def load_arc_encoding_attn_input_block(
    weights: pb.Weights, pt_input_block: layers.AttentionArcEncodingInputBlock
):
    load_param(weights.ip_emb_w, pt_input_block.embed.weight)
    load_param(weights.ip_emb_b, pt_input_block.embed.bias)

    load_param(
        weights.ip_mult_gate, pt_input_block.ma_gating.mul_gate, permutation=(1, 0)
    )
    load_param(
        weights.ip_add_gate, pt_input_block.ma_gating.add_gate, permutation=(1, 0)
    )


def load_new_attn_input_block(
    weights: pb.Weights, pt_input_block: layers.AttentionInputBlock
):
    load_param(weights.ip_emb_preproc_w, pt_input_block.preprocess.weight)
    load_param(weights.ip_emb_preproc_b, pt_input_block.preprocess.bias)

    load_param(weights.ip_emb_w, pt_input_block.embed.weight)
    load_param(weights.ip_emb_b, pt_input_block.embed.bias)

    load_param(weights.ip_emb_ln_gammas, pt_input_block.ln.weight)
    load_param(weights.ip_emb_ln_betas, pt_input_block.ln.bias)

    load_param(
        weights.ip_mult_gate, pt_input_block.ma_gating.mul_gate, permutation=(1, 0)
    )
    load_param(
        weights.ip_add_gate, pt_input_block.ma_gating.add_gate, permutation=(1, 0)
    )

    load_ffn(weights.ip_emb_ffn, pt_input_block.ffn)

    load_param(weights.ip_emb_ffn_ln_gammas, pt_input_block.ffn_ln.weight)
    load_param(weights.ip_emb_ffn_ln_betas, pt_input_block.ffn_ln.bias)


def load_smolgen(smolgen: pb.Weights.Smolgen, pt_smolgen: layers.Smolgen | None):
    if pt_smolgen is None:
        return
    load_param(smolgen.compress, pt_smolgen.compress.weight)
    load_param(smolgen.dense1_w, pt_smolgen.hidden_dense.weight)
    load_param(smolgen.dense1_b, pt_smolgen.hidden_dense.bias)
    load_param(smolgen.ln1_gammas, pt_smolgen.hidden_ln.weight)
    load_param(smolgen.ln1_betas, pt_smolgen.hidden_ln.bias)
    load_param(smolgen.dense2_w, pt_smolgen.gen_from.weight)
    load_param(smolgen.dense2_b, pt_smolgen.gen_from.bias)
    load_param(smolgen.ln2_gammas, pt_smolgen.gen_ln.weight)
    load_param(smolgen.ln2_betas, pt_smolgen.gen_ln.bias)


def load_mha(mha: pb.Weights.MHA, pt_mha: layers.MultiHeadedAttention):
    load_param(mha.q_w, pt_mha.q_proj.weight)
    load_param(mha.q_b, pt_mha.q_proj.bias)
    load_param(mha.k_w, pt_mha.k_proj.weight)
    load_param(mha.k_b, pt_mha.k_proj.bias)
    load_param(mha.v_w, pt_mha.v_proj.weight)
    load_param(mha.v_b, pt_mha.v_proj.bias)
    load_param(mha.dense_w, pt_mha.out_proj.weight)
    load_param(mha.dense_b, pt_mha.out_proj.bias)

    load_smolgen(mha.smolgen, pt_mha.smolgen)


def load_encoder_layers(weights: pb.Weights, pt_encoder_layers: torch.nn.Sequential):
    encoder_layers: list[pb.Weights.EncoderLayer] = weights.encoder
    if len(encoder_layers) != len(pt_encoder_layers):
        raise ValueError(
            f"Different numbers of encoder layers: {len(encoder_layers)} != {len(pt_encoder_layers)}"
        )
    for enclay, pt_enclay in zip(encoder_layers, pt_encoder_layers):
        load_param(
            weights.smolgen_w, pt_enclay.mha.smolgen.smol_weight_gen_dense.weight
        )  # Global smolgen weights
        load_mha(enclay.mha, pt_enclay.mha)

        load_param(enclay.ln1_gammas, pt_enclay.ln1.weight)
        load_param(enclay.ln1_betas, pt_enclay.ln1.bias)

        load_ffn(enclay.ffn, pt_enclay.ffn)

        load_param(enclay.ln2_gammas, pt_enclay.ln2.weight)
        load_param(enclay.ln2_betas, pt_enclay.ln2.bias)


def load_attn_policy_head(
    policy_head: pb.Weights.PolicyHead,
    pt_policy_head: layers.AttentionPolicyHead | None,
):
    if pt_policy_head is None:
        return
    load_param(policy_head.ip2_pol_w, pt_policy_head.q_proj.weight)
    load_param(policy_head.ip2_pol_b, pt_policy_head.q_proj.bias)
    load_param(policy_head.ip3_pol_w, pt_policy_head.k_proj.weight)
    load_param(policy_head.ip3_pol_b, pt_policy_head.k_proj.bias)
    load_param(policy_head.ip4_pol_w, pt_policy_head.promotion_offsets_proj.weight)


def load_attn_value_head(
    value_head: pb.Weights.ValueHead,
    pt_value_head: layers.AttentionValueHead | None,
    permissive: bool = False,
):
    if pt_value_head is None:
        return
    load_param(value_head.ip_val_w, pt_value_head.embed.weight)
    load_param(value_head.ip_val_b, pt_value_head.embed.bias)
    load_param(value_head.ip1_val_w, pt_value_head.dense.weight)
    load_param(value_head.ip1_val_b, pt_value_head.dense.bias)
    load_param(value_head.ip2_val_w, pt_value_head.val_proj.weight)
    load_param(value_head.ip2_val_b, pt_value_head.val_proj.bias)
    if pt_value_head.val_err_proj is not None:
        load_param(value_head.ip_val_err_w, pt_value_head.val_err_proj.weight)
        load_param(value_head.ip_val_err_b, pt_value_head.val_err_proj.bias)
    if pt_value_head.val_cat_proj is not None:
        load_param(
            value_head.ip_val_cat_w,
            pt_value_head.val_cat_proj.weight,
            permissive=permissive,
        )
        load_param(
            value_head.ip_val_cat_b,
            pt_value_head.val_cat_proj.bias,
            permissive=permissive,
        )


def load_attn_moves_left_head(
    weights: pb.Weights, pt_moves_left_head: layers.AttentionMovesLeftHead | None
):
    if pt_moves_left_head is None:
        return
    load_param(weights.ip_mov_w, pt_moves_left_head.embed.weight)
    load_param(weights.ip_mov_b, pt_moves_left_head.embed.bias)
    load_param(weights.ip1_mov_w, pt_moves_left_head.dense1.weight)
    load_param(weights.ip1_mov_b, pt_moves_left_head.dense1.bias)
    load_param(weights.ip2_mov_w, pt_moves_left_head.dense2.weight)
    load_param(weights.ip2_mov_b, pt_moves_left_head.dense2.bias)


def load_bt_model(
    weights: pb.Weights,
    pt_model: aln.Lc0ChessFormerModel,
    is_hectoplies_format: bool = False,
):
    load_attn_input_block(weights, pt_model.input_block)
    if not is_hectoplies_format:
        fix_attn_50_rule_move(pt_model.input_block)
    load_encoder_layers(weights, pt_model.encoder_layers)

    # Hacky way to detect T82 vs t3/BT3/BT4
    if isinstance(pt_model.input_block, layers.AttentionArcEncodingInputBlock):
        load_param(weights.ip_pol_w, pt_model.policy_embed.weight)
        load_param(weights.ip_pol_b, pt_model.policy_embed.bias)
        load_attn_policy_head(weights, pt_model.policy_vanilla)
        load_attn_value_head(weights, pt_model.value_winner)
        load_attn_moves_left_head(weights, pt_model.moves_left_head)
    else:
        load_param(weights.policy_heads.ip_pol_w, pt_model.policy_embed.weight)
        load_param(weights.policy_heads.ip_pol_b, pt_model.policy_embed.bias)
        load_attn_policy_head(weights.policy_heads.vanilla, pt_model.policy_vanilla)
        load_attn_policy_head(
            weights.policy_heads.optimistic_st, pt_model.policy_optimistic_st
        )
        load_attn_policy_head(weights.policy_heads.soft, pt_model.policy_soft)
        load_attn_policy_head(weights.policy_heads.opponent, pt_model.policy_opponent)

        load_attn_value_head(weights.value_heads.winner, pt_model.value_winner)
        load_attn_value_head(weights.value_heads.q, pt_model.value_q, permissive=True)
        load_attn_value_head(weights.value_heads.st, pt_model.value_st)

        load_attn_moves_left_head(weights, pt_model.moves_left_head)
