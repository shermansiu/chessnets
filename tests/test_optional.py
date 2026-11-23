import pytest

from chessnets._optional import import_optional, optional_module


def test_missing_optional_package_names_the_extra():
    with pytest.raises(ImportError, match=r"install chessnets\[demo\]"):
        import_optional("chessnets_missing_optional_pkg", "demo")


def test_package_without_an_extra_asks_for_the_package_itself():
    with pytest.raises(ImportError, match=r"install chessnets_missing_optional_pkg\.$"):
        import_optional("chessnets_missing_optional_pkg.sub", None)


def test_missing_transitive_dependency_is_not_attributed_to_the_extra(
    tmp_path, monkeypatch
):
    (tmp_path / "chessnets_installed_pkg.py").write_text(
        "import chessnets_missing_transitive_dep\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as excinfo:
        import_optional("chessnets_installed_pkg", "demo")
    assert excinfo.value.name == "chessnets_missing_transitive_dep"


def test_optional_module_defers_import_until_attribute_access():
    module = optional_module("chessnets_missing_optional_pkg", "demo")

    with pytest.raises(ImportError, match=r"install chessnets\[demo\]"):
        module.anything
