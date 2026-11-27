"""UniMaia model components."""

import logging
import typing as tp
from collections import OrderedDict

import torch
from torch import nn

from chessnets._optional import optional_module
from chessnets.models.lc0_cf import layers_lc0_cf as layers
from chessnets.models.lc0_cf.modeling_lc0_cf import (
    BTLeelaZeroNet,
    MultiValueModelOutput,
    optional_apply,
    optional_apply_triple,
)
from chessnets.models.lc0_senet import configuration_lc0_senet as leela_config
from chessnets.models.unimaia import configuration_unimaia as unimaia_config
from chessnets.utils import decode_policy

peft = optional_module("peft", "unimaia")
transformers = optional_module("transformers", "unimaia")


class UniMaiaModel(BTLeelaZeroNet):
    def __init__(
        self,
        config: leela_config.LeelaNetConfig,
        umt_config: unimaia_config.UniMaiaConfig,
        text_encoder: tp.Any,
    ):
        super().__init__(config=config)
        self.umt_config = umt_config

        if umt_config.text_encoder_type == unimaia_config.TextEncoderType.frozen:
            for name, param in text_encoder.named_parameters():
                # Debugging
                if (
                    "final_layer_norm" in name
                    and isinstance(text_encoder, transformers.GPTNeoXModel)
                ) or (
                    "norm" in name
                    and "layers" not in name
                    and isinstance(
                        text_encoder,
                        (transformers.Gemma3TextModel, transformers.Qwen3Model),
                    )
                ):
                    logging.info(f"Excluding layer {name} from being frozen")
                else:
                    param.requires_grad = False
        elif umt_config.text_encoder_type == unimaia_config.TextEncoderType.lora:
            peft_config = peft.LoraConfig(
                # peft_config = peft_singlora.SingLoRAConfig(
                task_type=peft.TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=umt_config.lora_rank,
                target_modules=umt_config.target_lora_modules,
                lora_alpha=umt_config.lora_alpha,
                lora_dropout=umt_config.lora_dropout,
                use_rslora=umt_config.use_rslora,
                use_dora=umt_config.use_dora,
                # ramp_up_steps=1,
            )
            text_encoder = peft.get_peft_model(text_encoder, peft_config)
            # # We use bf16 training, and the ramp-up is unstable for it.
            # # https://github.com/huggingface/peft/issues/2652#issuecomment-3105685693
            # peft_singlora.update_singlora_global_step(text_encoder, 1)

        self.text_encoder = text_encoder

        self.vlm_architecture_type = umt_config.vlm_architecture_type

        MHAClass: type[nn.Module]
        if umt_config.should_use_gated_attention:
            MHAClass = layers.GatedMultiheadAttention
        else:
            MHAClass = nn.MultiheadAttention

        # For T5Gemma 2
        if hasattr(text_encoder.config, "text_config"):
            text_encoder_hidden_size = text_encoder.config.text_config.hidden_size
        else:
            text_encoder_hidden_size = text_encoder.config.hidden_size

        if self.vlm_architecture_type == unimaia_config.VLMModelArchitectureType.vlm:
            self.input_block_ln = nn.LayerNorm(
                [layers.BOARD_SIZE, self.config.model.embedding_dense_sz]
            )
            self.fusion_layer = MHAClass(
                embed_dim=self.config.model.embedding_dense_sz,
                num_heads=umt_config.fusion_num_attn_heads,
                kdim=text_encoder_hidden_size,
                vdim=text_encoder_hidden_size,
                dropout=umt_config.fusion_attn_dropout,
                batch_first=True,
            )

            layers.zero_init_module(self.fusion_layer.out_proj)

            self.fusion_layer_ffn_ln = nn.LayerNorm(
                [layers.BOARD_SIZE, self.config.model.embedding_dense_sz]
            )
            self.fusion_layer_ffn = layers.SwiGLU(
                self.config.model.embedding_dense_sz,
                umt_config.fusion_ffn_intermediate_dim_multiplier
                * self.config.model.embedding_dense_sz,
            )
            layers.zero_init_module(self.fusion_layer_ffn.weights_out)

            self.register_parameter("encoder_clone_layers", None)
            self.register_parameter("zero_proj_encoder_layers", None)
            self.register_parameter("policy_embed_clone", None)
            self.register_parameter("policy_vanilla_clone", None)
            self.register_parameter("zero_proj_policy_vanilla", None)
        elif (
            self.vlm_architecture_type
            == unimaia_config.VLMModelArchitectureType.controlnet
            or self.vlm_architecture_type
            == unimaia_config.VLMModelArchitectureType.controlnet_controllable_policy_head
        ):
            self.register_parameter("input_block_ln", None)
            self.register_parameter("fusion_layer", None)
            self.register_parameter("fusion_layer_ffn_ln", None)
            self.register_parameter("fusion_layer_ffn", None)

            deep_norm = layers.DeepNorm(config.model.encoder_layers)
            encoder_clone_layers = OrderedDict(
                [
                    (
                        f"encoder_clone_layer_{i}",
                        layers.CrossEncoderLayer(
                            config,
                            text_encoder.config,
                            deep_norm,
                            should_use_gated_attention=umt_config.should_use_gated_attention,
                        ),
                    )
                    for i in range(config.model.encoder_layers)
                ]
            )
            self.encoder_clone_layers = nn.Sequential(encoder_clone_layers)

            encoder_dim = config.model.encoder_d_model
            zero_proj_encoder_layers = OrderedDict(
                [
                    (
                        f"zero_proj_encoder_layer_{i}",
                        layers.zero_init_module(
                            nn.Linear(encoder_dim, encoder_dim, bias=True)
                        ),
                    )
                    for i in range(config.model.encoder_layers)
                ]
            )
            self.zero_proj_encoder_layers = nn.Sequential(zero_proj_encoder_layers)

            self.policy_embed_clone = nn.Linear(
                config.model.embedding_size, config.model.policy_embedding_size
            )
            if (
                self.vlm_architecture_type
                == unimaia_config.VLMModelArchitectureType.controlnet_controllable_policy_head
            ):
                self.policy_embed_clone_mha = MHAClass(
                    embed_dim=config.model.policy_embedding_size,
                    num_heads=config.model.encoder_heads,
                    dropout=config.model.dropout_rate,
                    bias=not config.model.omit_qkv_biases,
                    kdim=text_encoder_hidden_size,
                    vdim=text_encoder_hidden_size,
                    batch_first=True,
                )
                layers.zero_init_module(self.policy_embed_clone_mha.out_proj)
                self.policy_embed_clone_mha_ln = nn.LayerNorm(
                    config.model.policy_embedding_size, bias=True, eps=0.001
                )
            else:
                self.register_parameter("policy_embed_clone_mha", None)
                self.register_parameter("policy_embed_clone_mha_ln", None)
            self.policy_vanilla_clone = layers.AttentionPolicyHead(config)
            self.zero_proj_policy_vanilla = layers.zero_init_module(
                nn.Linear(len(decode_policy.MOVES), len(decode_policy.MOVES))
            )
        elif (
            self.vlm_architecture_type
            == unimaia_config.VLMModelArchitectureType.cross_encoder
            or self.vlm_architecture_type
            == unimaia_config.VLMModelArchitectureType.cross_encoder_controllable_policy_head
        ):
            self.register_parameter("input_block_ln", None)
            self.register_parameter("fusion_layer", None)
            self.register_parameter("fusion_layer_ffn_ln", None)
            self.register_parameter("fusion_layer_ffn", None)

            deep_norm = layers.DeepNorm(config.model.encoder_layers)
            encoder_layers = OrderedDict(
                [
                    (
                        f"encoder_layer_{i}",
                        layers.CrossEncoderLayer(
                            config,
                            text_encoder.config,
                            deep_norm,
                            should_use_gated_attention=umt_config.should_use_gated_attention,
                        ),
                    )
                    for i in range(config.model.encoder_layers)
                ]
            )
            # Replace the regular encoder layers with cross-encoder layers
            self.encoder_layers = nn.Sequential(encoder_layers)

            self.policy_embed_clone = nn.Linear(
                config.model.embedding_size, config.model.policy_embedding_size
            )
            if (
                self.vlm_architecture_type
                == unimaia_config.VLMModelArchitectureType.cross_encoder_controllable_policy_head
            ):
                self.policy_embed_clone_mha = MHAClass(
                    embed_dim=config.model.policy_embedding_size,
                    num_heads=config.model.encoder_heads,
                    dropout=config.model.dropout_rate,
                    bias=not config.model.omit_qkv_biases,
                    kdim=text_encoder_hidden_size,
                    vdim=text_encoder_hidden_size,
                    batch_first=True,
                )
                layers.zero_init_module(self.policy_embed_clone_mha.out_proj)
                self.policy_embed_clone_mha_ln = nn.LayerNorm(
                    config.model.policy_embedding_size, bias=True, eps=0.001
                )
            else:
                self.register_parameter("policy_embed_clone_mha", None)
                self.register_parameter("policy_embed_clone_mha_ln", None)
            # Why not just use the original policy head!?
            # I mean, the initial ablation was to see if a lightweight change to the ControlNet ablation would improve results
            # The initial one didn't, but the later experiment had better results
            # Keeping for reproducibility's sake
            self.policy_vanilla_clone = layers.AttentionPolicyHead(config)
            self.zero_proj_policy_vanilla = layers.zero_init_module(
                nn.Linear(len(decode_policy.MOVES), len(decode_policy.MOVES))
            )

            self.register_parameter("encoder_clone_layers", None)
            self.register_parameter("zero_proj_encoder_layers", None)
        else:
            raise ValueError(
                f"Unsupported VLM architecture type: {self.vlm_architecture_type}"
            )

        if self.vlm_architecture_type in (
            unimaia_config.VLMModelArchitectureType.controlnet,
            unimaia_config.VLMModelArchitectureType.controlnet_controllable_policy_head,
            unimaia_config.VLMModelArchitectureType.cross_encoder,
            unimaia_config.VLMModelArchitectureType.cross_encoder_controllable_policy_head,
        ):
            if umt_config.should_use_value_loss:
                self.value_winner_clone = layers.AttentionValueHead(
                    config, wdl=self.wdl, use_err=False, use_cat=False
                )
                self.zero_proj_value_winner = layers.zero_init_module(
                    nn.Linear(3 if self.wdl else 1, 3 if self.wdl else 1)
                )
            else:
                self.register_parameter("value_winner_clone", None)
                self.register_parameter("zero_proj_value_winner", None)
            if umt_config.should_use_moves_left_loss:
                self.moves_left_clone = layers.AttentionMovesLeftHead(config)
                self.zero_proj_moves_left = layers.zero_init_module(nn.Linear(1, 1))
            else:
                self.register_parameter("moves_left_clone", None)
                self.register_parameter("zero_proj_moves_left", None)

        else:
            self.register_parameter("value_winner_clone", None)
            self.register_parameter("zero_proj_value_winner", None)
            self.register_parameter("moves_left_clone", None)
            self.register_parameter("zero_proj_moves_left", None)

        if umt_config.should_use_outcome_type_loss:
            self.outcome = layers.AttentionOutcomeHead(
                config, num_categories=umt_config.num_outcome_types
            )
        else:
            self.register_parameter("outcome", None)
        if umt_config.should_use_move_delay_loss:
            self.move_delay_rms_norm = nn.RMSNorm(config.model.embedding_size)
            self.move_delay = layers.AttentionMovesLeftHead(config)
        else:
            self.register_parameter("move_delay_rms_norm", None)
            self.register_parameter("move_delay", None)

    @classmethod
    def from_config(
        cls,
        config: leela_config.LeelaNetConfig,
        umt_config: unimaia_config.UniMaiaConfig,
        text_encoder: tp.Any,
    ) -> "UniMaiaModel":
        """Initialize UniMaia architecture from already-resolved configs."""
        return cls(config=config, umt_config=umt_config, text_encoder=text_encoder)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load UniMaia weights; tokenizers remain a separate component."""
        from chessnets.loaders.load_unimaia_model import load_unimaia_model

        model, _tokenizer = load_unimaia_model(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, cls):
            raise TypeError(
                f"Checkpoint resolved to {type(model).__name__}, not {cls.__name__}"
            )
        return model

    @property
    def base_text_encoder(self) -> tp.Any:
        """Return the underlying text encoder, unwrapping a PEFT adapter."""
        text_encoder = self.text_encoder
        get_base_model = getattr(text_encoder, "get_base_model", None)
        if get_base_model is not None:
            return get_base_model()
        base_model = getattr(text_encoder, "base_model", None)
        if base_model is not None:
            return getattr(base_model, "model", base_model)
        return text_encoder

    def forward(
        self, x: torch.Tensor, text_tokens: dict[str, torch.Tensor]
    ) -> MultiValueModelOutput:
        board_embeddings = self.input_block(x)
        text_encoder_output = self.text_encoder(
            **text_tokens, output_hidden_states=True
        )
        if isinstance(
            self.text_encoder,
            (
                transformers.GPTNeoXModel,
                transformers.models.t5gemma2.modeling_t5gemma2.T5Gemma2Encoder,
            ),
        ):
            text_embeddings = (
                text_encoder_output.last_hidden_state
            )  # Shape: (batch_size, n_tokens, hidden_size)
        else:
            text_embeddings = text_encoder_output.hidden_states[
                -1
            ]  # Shape: (batch_size, n_tokens, hidden_size)
        del text_encoder_output

        if self.umt_config.use_key_padding_mask:
            key_padding_mask = ~text_tokens["attention_mask"].bool()
        else:
            key_padding_mask = None

        if self.vlm_architecture_type == unimaia_config.VLMModelArchitectureType.vlm:
            # Pre-layer norm Cross-Attention
            # We unfreeze the final layer norm in the text encoder, so we don't need an extra one
            board_embeddings_norm = self.input_block_ln(board_embeddings)
            x = (
                board_embeddings
                + self.fusion_layer(
                    board_embeddings_norm,
                    text_embeddings,
                    text_embeddings,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            del board_embeddings_norm

            # Pre-layer norm SwiGLU FFN
            x = x + self.fusion_layer_ffn(self.fusion_layer_ffn_ln(x))

            x = self.encoder_layers(x)
            policy_delta = None
        elif (
            self.vlm_architecture_type
            == unimaia_config.VLMModelArchitectureType.controlnet
            or self.vlm_architecture_type
            == unimaia_config.VLMModelArchitectureType.controlnet_controllable_policy_head
        ):
            if self.umt_config.use_intermediate_states_for_controlnet:
                for enc_layer, enc_clone_layer, zero_proj_enc_layer in zip(
                    self.encoder_layers,
                    self.encoder_clone_layers,
                    self.zero_proj_encoder_layers,
                ):
                    x = enc_clone_layer(
                        board_embeddings,
                        text_embeddings,
                        key_padding_mask=key_padding_mask,
                    )
                    board_embeddings = enc_layer(board_embeddings)
                    board_embeddings = board_embeddings + zero_proj_enc_layer(x)

                x = board_embeddings
                del board_embeddings
            else:
                x = board_embeddings
                for enc_layer, enc_clone_layer, zero_proj_enc_layer in zip(
                    self.encoder_layers,
                    self.encoder_clone_layers,
                    self.zero_proj_encoder_layers,
                ):
                    x = enc_clone_layer(
                        x, text_embeddings, key_padding_mask=key_padding_mask
                    )
                    board_embeddings = enc_layer(board_embeddings)
                    board_embeddings = board_embeddings + zero_proj_enc_layer(x)

                x = board_embeddings
                del board_embeddings

            pol_embed = self.default_activation_fn(self.policy_embed_clone(x))
            if (
                self.vlm_architecture_type
                == unimaia_config.VLMModelArchitectureType.controlnet_controllable_policy_head
            ):
                pol_embed_norm = self.policy_embed_clone_mha_ln(pol_embed)
                pol_embed = (
                    pol_embed
                    + self.policy_embed_clone_mha(
                        pol_embed_norm,
                        text_embeddings,
                        text_embeddings,
                        key_padding_mask=key_padding_mask,
                        need_weights=False,
                    )[0]
                )
                del pol_embed_norm
            policy_delta = self.zero_proj_policy_vanilla(
                self.policy_vanilla_clone(pol_embed)
            )
        elif (
            self.vlm_architecture_type
            == unimaia_config.VLMModelArchitectureType.cross_encoder
            or self.vlm_architecture_type
            == unimaia_config.VLMModelArchitectureType.cross_encoder_controllable_policy_head
        ):
            for enc_layer in self.encoder_layers:
                board_embeddings = enc_layer(
                    board_embeddings, text_embeddings, key_padding_mask=key_padding_mask
                )

            x = board_embeddings
            del board_embeddings

            pol_embed = self.default_activation_fn(self.policy_embed_clone(x))
            if (
                self.vlm_architecture_type
                == unimaia_config.VLMModelArchitectureType.cross_encoder_controllable_policy_head
            ):
                pol_embed_norm = self.policy_embed_clone_mha_ln(pol_embed)
                pol_embed = (
                    pol_embed
                    + self.policy_embed_clone_mha(
                        pol_embed_norm,
                        text_embeddings,
                        text_embeddings,
                        key_padding_mask=key_padding_mask,
                        need_weights=False,
                    )[0]
                )
                del pol_embed_norm
            policy_delta = self.zero_proj_policy_vanilla(
                self.policy_vanilla_clone(pol_embed)
            )
        else:
            raise ValueError(
                f"Unsupported VLM architecture type: {self.vlm_architecture_type}"
            )

        pol_embed = self.default_activation_fn(self.policy_embed(x))

        policy = self.policy_vanilla(pol_embed)
        if policy_delta is not None:
            policy = policy + policy_delta

        policy_optimistic_st = optional_apply(self.policy_optimistic_st, pol_embed)
        policy_soft = optional_apply(self.policy_soft, pol_embed)
        policy_opponent = optional_apply(self.policy_opponent, pol_embed)
        policy_next = optional_apply(self.policy_next, pol_embed)

        value_winner, _, _ = self.value_winner(x)
        value_q, value_q_err, value_q_cat = optional_apply_triple(self.value_q, x)
        value_st, value_st_err, value_st_cat = optional_apply_triple(self.value_st, x)

        if self.value_winner_clone is not None:
            value_winner_clone, _, _ = self.value_winner_clone(x)
            value_winner = value_winner + self.zero_proj_value_winner(
                value_winner_clone
            )

        moves_left = optional_apply(self.moves_left_head, x)

        if self.moves_left_clone is not None:
            moves_left_clone = self.moves_left_clone(x)
            moves_left = moves_left + self.zero_proj_moves_left(moves_left_clone)

        if self.outcome is not None:
            outcome = self.outcome(x)
        else:
            outcome = None

        if self.move_delay is not None:
            move_delay = self.move_delay(self.move_delay_rms_norm(x))
        else:
            move_delay = None

        return MultiValueModelOutput(
            policy,
            policy_optimistic_st,
            policy_soft,
            policy_opponent,
            policy_next,
            value_winner,
            value_q,
            value_q_err,
            value_q_cat,
            value_st,
            value_st_err,
            value_st_cat,
            moves_left,
            future=None,
            outcome=outcome,
            move_delay=move_delay,
        )


__all__ = ["MultiValueModelOutput", "UniMaiaModel"]
