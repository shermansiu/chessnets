#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Build and check chessnets and its official lc0-protobuf plugin, and optionally
upload them to TestPyPI.

Production releases are not made from here: push a tag v<version> and
.github/workflows/release.yml publishes to PyPI and creates the GitHub Release
(see docs/maintainers/releasing.md).

Usage:
  scripts/publish_pypi.sh [options]

Package selection:
  --package NAME       What to build: core, lc0-protobuf, or all (default: all).

Upload:
  --upload             Upload after building and validating the artifacts.
  --test-pypi          Upload to TestPyPI (requires --upload).
  --skip-existing      Skip files already on the index (resume a partial upload).
  --yes                Skip the upload confirmation prompt.
  --allow-dirty        Allow uploading from a working tree with changes.

Other:
  --outdir DIR         Keep the built artifacts in DIR (default: a temporary
                       directory that is deleted on exit).
  -h, --help           Show this help.

With no upload options, the script only builds and validates. Validation:
  - scripts/sync_versions.py: both packages share one version and pin each
    other exactly (change versions with scripts/sync_versions.py --bump);
  - twine check on every artifact;
  - scripts/check_distribution_licenses.py on every artifact: SPDX headers
    must fit its License-Expression, and the Apache-2.0/MIT core must contain
    no GPL text, protobuf code, or plugin package.

A package whose pyproject.toml has a "Private ::" classifier is built and
validated but never uploaded. When both packages are uploaded, the plugin goes
first so the core package's lc0-protobuf extra never points at a missing
release.

Authentication is handled by Twine. Set TWINE_USERNAME=__token__ and
TWINE_PASSWORD to a TestPyPI API token, or configure ~/.pypirc. Set PYTHON_BIN to use
a specific Python interpreter (3.11+).
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

package="all"
upload=false
repository=""
allow_dirty=false
assume_yes=false
skip_existing=false
outdir=""

while (($#)); do
    case "$1" in
        --package)
            (($# >= 2)) || die "--package needs a value"
            package="$2"
            shift
            ;;
        --package=*) package="${1#*=}" ;;
        --upload) upload=true ;;
        --test-pypi)
            [[ -z "$repository" || "$repository" == "testpypi" ]] || die "choose only one upload target"
            repository="testpypi"
            ;;
        --pypi)
            die "production releases go through the release workflow: push tag v<version> (see docs/maintainers/releasing.md)"
            ;;
        --skip-existing) skip_existing=true ;;
        --tag)
            die "tags are created by hand and pushed to trigger the release workflow (see docs/maintainers/releasing.md)"
            ;;
        --allow-dirty) allow_dirty=true ;;
        --yes) assume_yes=true ;;
        --outdir)
            (($# >= 2)) || die "--outdir needs a value"
            outdir="$2"
            shift
            ;;
        --outdir=*) outdir="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

case "$package" in
    core|lc0-protobuf|all) ;;
    *) die "--package must be core, lc0-protobuf, or all (got: $package)" ;;
esac
if [[ -n "$repository" && "$upload" != true ]]; then
    die "--${repository} requires --upload"
fi
if [[ "$upload" == true && -z "$repository" ]]; then
    die "--upload needs --test-pypi"
fi
if [[ "$upload" != true ]]; then
    [[ "$skip_existing" != true ]] || die "--skip-existing requires --upload"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$project_dir"

[[ -f pyproject.toml ]] || die "pyproject.toml not found in $project_dir"
python_bin="${PYTHON_BIN:-python3}"
command -v "$python_bin" >/dev/null 2>&1 || die "Python is required (set PYTHON_BIN if it is not named python3)"
"$python_bin" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || \
    die "Python 3.11+ is required (set PYTHON_BIN)"
"$python_bin" -m build --version >/dev/null 2>&1 || \
    die "the 'build' package is missing; install publishing tools with: $python_bin -m pip install build twine"
"$python_bin" -m twine --version >/dev/null 2>&1 || \
    die "the 'twine' package is missing; install publishing tools with: $python_bin -m pip install build twine"

if [[ "$upload" == true ]]; then
    command -v git >/dev/null 2>&1 || die "git is required for upload checks"
fi
if [[ "$upload" == true && "$allow_dirty" != true ]]; then
    [[ -z "$(git status --porcelain)" ]] || \
        die "working tree has uncommitted changes; commit them or pass --allow-dirty"
fi

printf 'Checking versions...\n'
"$python_bin" scripts/sync_versions.py || die "versions are out of sync"

# Reads "name version private" for a project directory.
project_info() {
    "$python_bin" - "$1/pyproject.toml" <<'PY'
import pathlib
import sys
import tomllib

project = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]
private = any(c.startswith("Private ::") for c in project.get("classifiers", []))
print(project["name"], project["version"], "private" if private else "public")
PY
}

# Build order is upload order: the plugin first.
declare -a selected_dirs=()
case "$package" in
    lc0-protobuf) selected_dirs=("plugins/lc0-protobuf") ;;
    core) selected_dirs=(".") ;;
    all) selected_dirs=("plugins/lc0-protobuf" ".") ;;
esac

read -r _ version _ < <(project_info .)

if [[ -n "$outdir" ]]; then
    mkdir -p -- "$outdir"
    artifact_root="$(cd -- "$outdir" && pwd)"
    [[ -z "$(ls -A -- "$artifact_root")" ]] || die "--outdir must be empty: $artifact_root"
else
    artifact_root="$(mktemp -d "${TMPDIR:-/tmp}/chessnets-publish.XXXXXXXX")"
    cleanup() {
        rm -rf -- "$artifact_root"
    }
    trap cleanup EXIT
fi

declare -a upload_dirs=() upload_names=() private_names=()
for dir in "${selected_dirs[@]}"; do
    read -r name pkg_version visibility < <(project_info "$dir")
    dist_dir="$artifact_root/$name"
    printf '\nBuilding %s %s...\n' "$name" "$pkg_version"
    "$python_bin" -m build --outdir "$dist_dir" "$dir"

    printf '\nValidating %s...\n' "$name"
    "$python_bin" -m twine check --strict "$dist_dir"/*
    "$python_bin" scripts/check_distribution_licenses.py "$dist_dir"/*

    if [[ "$visibility" == "private" ]]; then
        private_names+=("$name")
    else
        upload_dirs+=("$dist_dir")
        upload_names+=("$name")
    fi
done

printf '\nArtifacts in %s:\n' "$artifact_root"
for dist_dir in "$artifact_root"/*/; do
    for artifact in "$dist_dir"*; do
        printf '  %s\n' "$(basename -- "$artifact")"
    done
done
for name in "${private_names[@]}"; do
    printf 'Note: %s has a "Private ::" classifier and will not be uploaded.\n' "$name"
done

if [[ "$upload" != true ]]; then
    printf '\nBuild and validation succeeded; nothing was uploaded.\n'
    exit 0
fi

((${#upload_dirs[@]})) || die "nothing to upload: every selected package is private"
if [[ "$package" == "all" && ${#private_names[@]} -gt 0 ]]; then
    die "${private_names[*]} is private, so '--package all' cannot upload both; pass --package core to upload only chessnets"
fi

if [[ "$assume_yes" != true ]]; then
    printf '\nUpload %s %s to %s? [y/N] ' "${upload_names[*]}" "$version" "$repository"
    read -r answer
    [[ "$answer" == "y" || "$answer" == "Y" ]] || die "upload cancelled"
fi

declare -a twine_args=(--repository "$repository")
[[ "$skip_existing" != true ]] || twine_args+=(--skip-existing)
for i in "${!upload_dirs[@]}"; do
    printf '\nUploading %s...\n' "${upload_names[$i]}"
    "$python_bin" -m twine upload "${twine_args[@]}" "${upload_dirs[$i]}"/*
done
printf '\nPublished %s %s to %s.\n' "${upload_names[*]}" "$version" "$repository"
