import os
import subprocess
import sys

import pytest


def _run_import_check(*statements: str) -> set[str]:
    source = "\n".join((*statements, "print(','.join(sorted(sys.modules)))"))
    env = os.environ.copy()
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import sys\n" + source],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return set(result.stdout.strip().split(","))


def test_pipeline_registry_import_is_lazy():
    modules = _run_import_check(
        "from chessnets.pipelines.auto_pipeline import AutoPipeline",
    )
    assert "einops" not in modules
    assert "transformers" not in modules
    assert "openai" not in modules
    assert "outlines" not in modules
    assert "open_clip" not in modules
    assert "chessnets.models.maia2.modeling_maia2" not in modules
    assert "chessnets.pipelines.maia2.pipeline_maia2" not in modules


def test_model_id_lookup_does_not_import_torch_or_pipelines():
    # Tools like chesseval only read PIPELINE_INFOS to resolve model names.
    modules = _run_import_check(
        "from chessnets.pipelines import AutoPipeline, PIPELINE_INFOS",
        "AutoPipeline.get_pipeline_info('maia-1500')",
        "assert PIPELINE_INFOS",
    )
    assert "torch" not in modules
    assert not any(
        name.startswith("chessnets.pipelines.") and name.count(".") > 2
        for name in modules
    ), sorted(name for name in modules if name.startswith("chessnets.pipelines."))


def test_maia2_support_modules_are_lazy_without_model_dependency():
    modules = _run_import_check(
        "import chessnets.models.maia2",
        "from chessnets.models.maia2 import Maia2Config",
        "from chessnets.models.maia2 import utils",
        "import chessnets.pipelines.maia2",
        "from chessnets.utils import board_history",
    )
    assert "einops" not in modules
    assert "chessnets.models.maia2.modeling_maia2" not in modules


def test_maia2_pipeline_reports_its_optional_extra_when_model_dependency_is_missing():
    env = os.environ.copy()
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.modules['einops'] = None; "
            "from chessnets import AutoPipeline; "
            "AutoPipeline.from_pretrained('maia2_blitz', device='cpu')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "install chessnets[maia2]" in result.stderr


def test_pipeline_family_packages_are_lazy_until_symbol_access():
    modules = _run_import_check(
        "import chessnets.pipelines.lc0",
        "import chessnets.pipelines.unimaia",
        "import chessnets.pipelines.text_generation",
    )
    assert "transformers" not in modules
    assert "openai" not in modules
    assert "outlines" not in modules


def test_canonical_pipeline_imports_do_not_use_legacy_implementation():
    modules = _run_import_check(
        "from chessnets.pipelines.lc0 import Lc0Pipeline",
        "from chessnets.pipelines.unimaia import UniMaiaPipeline",
        "from chessnets.pipelines.text_generation import TextGenerationPipeline",
    )
    assert "chessnets.pipelines.lc0.pipeline_lc0" in modules
    assert "chessnets.pipelines.unimaia.pipeline_unimaia" in modules
    assert "chessnets.pipelines.text_generation.pipeline_text_generation" in modules


@pytest.mark.parametrize("module_name", ["inference", "inference_factory"])
def test_legacy_pipeline_modules_are_removed(module_name):
    with pytest.raises(ModuleNotFoundError):
        __import__(f"chessnets.pipelines.{module_name}")


def test_public_pipeline_exports_do_not_include_predictor_aliases():
    import chessnets.pipelines as pipelines
    from chessnets.pipelines import auto_pipeline

    assert not any(name.endswith("Predictor") for name in pipelines.__all__)
    assert not any("predictor" in name.lower() for name in auto_pipeline.__all__)
