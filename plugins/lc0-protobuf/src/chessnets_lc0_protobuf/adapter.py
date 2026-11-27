"""Bridge legacy Lc0 protobuf checkpoints into the ChessNets model classes."""

from __future__ import annotations

import json
import pathlib

import yaml

from chessnets.models.lc0 import modeling_lc0_chessformer
from chessnets.models.lc0 import configuration_lc0 as leela_config
from chessnets.models.lc0 import modeling_lc0_seresnet

from . import checkpoint, load_protobuf
from . import net_pb2 as pb


class ProtobufAdapter:
    """Parse, validate, and decode legacy Lc0 protobuf networks."""

    @staticmethod
    def inspect_checkpoint(model_path: str | pathlib.Path) -> dict[str, str]:
        net = checkpoint.read_checkpoint(model_path)
        if net.format.weights_encoding != pb.Format.LINEAR16:
            raise ValueError("Only LINEAR16 protobuf weights are currently supported")
        return {
            "source_container_format": "lc0-protobuf",
            "source_weight_encoding": "LINEAR16",
            "decoded_tensor_dtype": "float32",
        }

    @staticmethod
    def load_model(
        model_path: str | pathlib.Path,
        config_path: str | pathlib.Path,
        *,
        is_eval: bool = True,
    ):
        config_path = pathlib.Path(config_path)
        with config_path.open(encoding="utf-8") as stream:
            config_values = (
                json.load(stream)
                if config_path.suffix == ".json"
                else yaml.safe_load(stream) or {}
            )
        config = leela_config.Lc0InferenceConfig.from_dict(config_values)
        net = checkpoint.read_checkpoint(model_path)
        if net.format.weights_encoding != pb.Format.LINEAR16:
            raise ValueError("Only LINEAR16 protobuf weights are currently supported")
        if (
            net.format.network_format.input
            != pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
        ):
            raise ValueError("Only classical 112-plane protobuf inputs are supported")
        if net.format.network_format.output != pb.NetworkFormat.OUTPUT_WDL:
            raise ValueError("Only WDL protobuf outputs are supported")
        if net.format.network_format.value != pb.NetworkFormat.VALUE_WDL:
            raise ValueError("Only WDL protobuf value heads are supported")

        network = net.format.network_format.network
        if network == pb.NetworkFormat.NETWORK_SE_WITH_HEADFORMAT:
            if net.format.network_format.policy != pb.NetworkFormat.POLICY_CONVOLUTION:
                raise ValueError("SENet protobuf must use a convolutional policy")
            if config.model.filters != checkpoint.filter_count(net):
                raise ValueError("Config filters do not match the protobuf network")
            if config.model.residual_blocks != checkpoint.residual_block_count(net):
                raise ValueError(
                    "Config residual blocks do not match the protobuf network"
                )
            model = modeling_lc0_seresnet.Lc0SeResNetModel.from_config(config)
            load_protobuf.load_se_model(net.weights, model)
        elif network in {
            pb.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT,
            pb.NetworkFormat.NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT,
        }:
            if net.format.network_format.policy != pb.NetworkFormat.POLICY_ATTENTION:
                raise ValueError("Attention protobuf must use an attention policy")
            model = modeling_lc0_chessformer.Lc0ChessFormerModel.from_config(config)
            load_protobuf.load_bt_model(net.weights, model)
        else:
            raise ValueError(f"Unsupported network format {network}")

        if is_eval:
            model.eval()
        return model


adapter = ProtobufAdapter()

__all__ = ["ProtobufAdapter", "adapter"]
