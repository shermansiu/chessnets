import importlib.util
import io
import pathlib
import tarfile
import zipfile

import pytest

ROOT = pathlib.Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "check_distribution_licenses.py"
SPEC = importlib.util.spec_from_file_location("check_distribution_licenses", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PLUGIN_SRC = ROOT / "plugins" / "lc0-protobuf" / "src" / "chessnets_lc0_protobuf"


def _metadata(expression):
    return f"Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\nLicense-Expression: {expression}\n"


def _wheel(path, expression, files):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pkg-1.0.dist-info/METADATA", _metadata(expression))
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def _sdist(path, expression, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, text in {"PKG-INFO": _metadata(expression), **files}.items():
            data = text.encode()
            info = tarfile.TarInfo(f"pkg-1.0/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


CLEAN = {
    "pkg/__init__.py": "# SPDX-License-Identifier: MIT\nimport importlib\n",
    "pkg/loader.py": 'MESSAGE = "needs the GPL-3.0-or-later adapter"\n'
    'importlib.import_module("chessnets_lc0_protobuf.adapter")\n',
}


@pytest.mark.parametrize("build", [_wheel, _sdist])
def test_clean_permissive_artifact_passes(tmp_path, build):
    artifact = build(tmp_path / "artifact", "Apache-2.0 AND MIT", CLEAN)
    assert MODULE.check_artifact(artifact) == []


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"pkg/x.py": "# SPDX-License-Identifier: GPL-3.0-or-later\n"}, "SPDX"),
        ({"pkg/x.py": "# SPDX-License-Identifier: BSD-3-Clause\n"}, "SPDX"),
        ({"pkg/x.py": "# GNU General Public License v3\n"}, "GNU license text"),
        ({"pkg/x.py": "from google.protobuf import descriptor\n"}, "protobuf"),
        ({"pkg/x.py": "from . import net_pb2 as pb\n"}, "protobuf"),
        ({"chessnets_lc0_protobuf/adapter.py": "x = 1\n"}, "plugin package"),
    ],
)
@pytest.mark.parametrize("build", [_wheel, _sdist])
def test_permissive_artifact_rejects_copyleft_content(tmp_path, build, files, expected):
    artifact = build(tmp_path / "artifact", "Apache-2.0 AND MIT", {**CLEAN, **files})
    problems = MODULE.check_artifact(artifact)
    assert len(problems) == 1 and expected in problems[0]


def test_renamed_plugin_files_are_caught_by_content(tmp_path):
    files = {
        "chessnets/loaders/net.py": (PLUGIN_SRC / "net_pb2.py").read_text(),
        "chessnets/loaders/schema.py": (PLUGIN_SRC / "load_protobuf.py").read_text(),
    }
    artifact = _wheel(tmp_path / "a.whl", "Apache-2.0 AND MIT", files)
    problems = MODULE.check_artifact(artifact)
    assert any("net.py" in problem for problem in problems)
    assert any("schema.py" in problem for problem in problems)


def test_copyleft_artifact_may_contain_gpl_code(tmp_path):
    files = {
        "chessnets_lc0_protobuf/net.py": "# SPDX-License-Identifier: GPL-3.0-or-later\n"
        "from . import net_pb2\n",
        "LICENSE": "GNU GENERAL PUBLIC LICENSE\n",
    }
    artifact = _wheel(tmp_path / "a.whl", "GPL-3.0-or-later", files)
    assert MODULE.check_artifact(artifact) == []


def test_missing_license_expression_fails(tmp_path):
    artifact = tmp_path / "a.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\n")
    assert "License-Expression" in MODULE.check_artifact(artifact)[0]
