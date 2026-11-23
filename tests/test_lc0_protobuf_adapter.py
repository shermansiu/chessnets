import gzip

import numpy as np
import pytest

pytest.importorskip("chessnets_lc0_protobuf")

from chessnets_lc0_protobuf import adapter, checkpoint  # noqa: E402
from chessnets_lc0_protobuf import net_pb2 as pb  # noqa: E402

NF = pb.NetworkFormat


def _supported_se(filters=8, blocks=2):
    net = pb.Net()
    net.format.weights_encoding = pb.Format.LINEAR16
    fmt = net.format.network_format
    fmt.input = NF.INPUT_CLASSICAL_112_PLANE
    fmt.output = NF.OUTPUT_WDL
    fmt.value = NF.VALUE_WDL
    fmt.network = NF.NETWORK_SE_WITH_HEADFORMAT
    fmt.policy = NF.POLICY_CONVOLUTION
    net.weights.input.bn_means.params = np.zeros(filters, np.uint16).tobytes()
    for _ in range(blocks):
        net.weights.residual.add()
    return net


def _write(tmp_path, net):
    path = tmp_path / "model.pb.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(net.SerializeToString())
    return path


def _config(tmp_path, filters=8, blocks=2):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"model:\n  filters: {filters}\n  residual_blocks: {blocks}\n", encoding="utf-8"
    )
    return path


def test_read_checkpoint_round_trips_and_reports_structure(tmp_path):
    path = _write(tmp_path, _supported_se(filters=8, blocks=3))

    net = checkpoint.read_checkpoint(path)

    assert net.format.network_format.network == NF.NETWORK_SE_WITH_HEADFORMAT
    assert checkpoint.filter_count(net) == 8
    assert checkpoint.residual_block_count(net) == 3


def test_inspect_checkpoint_prints_nothing(tmp_path, capsys):
    path = _write(tmp_path, _supported_se())

    assert adapter.inspect_checkpoint(path)["source_weight_encoding"] == "LINEAR16"
    # The UCI engine speaks its protocol on stdout, so loading must stay quiet.
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda n: setattr(n.format, "weights_encoding", 0), "LINEAR16"),
        (
            lambda n: setattr(
                n.format.network_format, "input", NF.INPUT_112_WITH_CANONICALIZATION
            ),
            "112-plane",
        ),
        (
            lambda n: setattr(n.format.network_format, "output", 1),
            "WDL protobuf outputs",
        ),
        (lambda n: setattr(n.format.network_format, "value", 1), "WDL protobuf value"),
        (
            lambda n: setattr(n.format.network_format, "network", 2),
            "Unsupported network",
        ),
        (
            lambda n: setattr(n.format.network_format, "policy", 1),
            "convolutional policy",
        ),
    ],
)
def test_load_model_rejects_unsupported_formats(tmp_path, change, match):
    net = _supported_se()
    change(net)

    with pytest.raises(ValueError, match=match):
        adapter.load_model(_write(tmp_path, net), _config(tmp_path))


@pytest.mark.parametrize(
    ("filters", "blocks", "match"),
    [(16, 2, "filters"), (8, 3, "residual blocks")],
)
def test_load_model_rejects_config_that_does_not_match_the_network(
    tmp_path, filters, blocks, match
):
    path = _write(tmp_path, _supported_se(filters=8, blocks=2))

    with pytest.raises(ValueError, match=match):
        adapter.load_model(path, _config(tmp_path, filters=filters, blocks=blocks))
