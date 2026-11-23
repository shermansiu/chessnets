import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).parents[1]
EXTRAS = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
    "optional-dependencies"
]
# dev is tooling; lc0-protobuf is GPL-3.0-or-later and stays opt-in.
NOT_IN_ALL = {"all", "dev", "lc0-protobuf"}


def test_all_extra_covers_every_runtime_extra():
    (requirement,) = EXTRAS["all"]
    match = re.fullmatch(r"chessnets\[([^\]]*)\]", requirement)
    assert match, requirement
    included = {name.strip() for name in match.group(1).split(",")}
    assert included == set(EXTRAS) - NOT_IN_ALL


def test_readme_lists_every_extra():
    readme = (ROOT / "README.md").read_text()
    for name in set(EXTRAS) - {"dev"}:
        assert f"`{name}`" in readme, name
