#!/usr/bin/env bash
# Turn Lc0/Maia-1 checkpoint Git repositories into the published ChessNets
# layout: original files under original/ and training/, a converted
# model.safetensors, config.json, provenance.json, and a generated model card.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/prepare_lc0_checkpoint_repo.sh REPOSITORY [REPOSITORY ...]
  scripts/prepare_lc0_checkpoint_repo.sh --apply REPOSITORY [REPOSITORY ...]

Each REPOSITORY is a Git repository (normally a Hugging Face clone) containing
the upstream model.pb.gz and config.yaml at its root, and nothing generated
yet. It may be an absolute path, a relative path, or a directory name under
CHECKPOINT_ROOT. Preview is the default; --apply performs the in-place changes.
Nothing is committed or pushed.

Environment overrides:
  CHECKPOINT_ROOT  Parent directory containing the checkpoint Git repositories.
  PYTHON_BIN       Python interpreter from the conversion environment.
EOF
}

apply=false
case "${1:-}" in
    --apply)
        apply=true
        shift
        ;;
    --help|-h)
        usage
        exit 0
        ;;
esac

if (($# == 0)); then
    usage >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
chessnets_root=$(cd -- "$script_dir/.." && pwd)
checkpoint_root=${CHECKPOINT_ROOT:-"$chessnets_root/../board_game_nn_huggingface/checkpoints"}
python_bin=${PYTHON_BIN:-python}
python_path="$chessnets_root/src:$chessnets_root/plugins/lc0-protobuf/src"

resolve_repo() {
    local requested=$1
    if [[ -d "$requested" ]]; then
        cd -- "$requested" && pwd
    elif [[ -d "$checkpoint_root/$requested" ]]; then
        cd -- "$checkpoint_root/$requested" && pwd
    else
        printf 'Checkpoint repository not found: %s\n' "$requested" >&2
        return 1
    fi
}

require_unprepared_repo() {
    local repo=$1
    git -C "$repo" rev-parse --is-inside-work-tree >/dev/null
    [[ -f "$repo/model.pb.gz" ]] || {
        printf 'Missing %s/model.pb.gz\n' "$repo" >&2
        return 1
    }
    [[ -f "$repo/config.yaml" ]] || {
        printf 'Missing %s/config.yaml\n' "$repo" >&2
        return 1
    }
    local output
    for output in \
        config.json \
        model.safetensors \
        provenance.json \
        original/model.pb.gz \
        training/original_lczero_training_config.yaml; do
        if [[ -e "$repo/$output" ]]; then
            printf 'Refusing partially prepared repository; %s exists\n' \
                "$repo/$output" >&2
            return 1
        fi
    done
}

preflight() {
    local repo=$1
    printf '\nPreflighting %s\n' "$repo"
    PYTHONPATH="$python_path${PYTHONPATH:+:$PYTHONPATH}" \
        "$python_bin" -m chessnets.loaders.convert_lc0 \
        "$repo" "$repo" --dry-run

    local lfs_filter
    lfs_filter=$(git -C "$repo" check-attr filter -- model.safetensors)
    if [[ "$lfs_filter" != *': lfs' ]]; then
        printf 'model.safetensors is not configured for Git LFS: %s\n' \
            "$lfs_filter" >&2
        return 1
    fi

    printf '\nProposed README update:\n'
    "$python_bin" "$chessnets_root/scripts/update_model_card.py" "$repo"
}

apply_preparation() {
    local repo=$1
    local dirty
    dirty=$(git -C "$repo" status --porcelain)
    if [[ -n "$dirty" ]]; then
        printf 'Refusing dirty repository %s:\n%s\n' "$repo" "$dirty" >&2
        return 1
    fi

    mkdir -p -- "$repo/original" "$repo/training"
    git -C "$repo" mv model.pb.gz original/model.pb.gz
    git -C "$repo" mv \
        config.yaml training/original_lczero_training_config.yaml

    if ! (
        cd -- "$repo"
        PYTHONPATH="$python_path${PYTHONPATH:+:$PYTHONPATH}" \
            "$python_bin" -m chessnets.loaders.convert_lc0 \
            original/model.pb.gz . \
            --config training/original_lczero_training_config.yaml
    ); then
        printf 'Conversion failed; restoring the two Git-tracked source paths.\n' >&2
        git -C "$repo" mv original/model.pb.gz model.pb.gz
        git -C "$repo" mv \
            training/original_lczero_training_config.yaml config.yaml
        printf 'Inspect and remove any untracked partial outputs before retrying.\n' >&2
        return 1
    fi

    "$python_bin" "$chessnets_root/scripts/update_model_card.py" \
        "$repo" --write

    printf '\nPreparation completed without committing or pushing: %s\n' "$repo"
    git -C "$repo" status --short
    printf '\nAfter review, stage generated artifacts with:\n'
    printf '  git -C %q add config.json model.safetensors provenance.json README.md\n' \
        "$repo"
    printf 'The two git mv operations are already staged.\n'
}

for requested in "$@"; do
    repo=$(resolve_repo "$requested")
    require_unprepared_repo "$repo"
    preflight "$repo"
    if "$apply"; then
        apply_preparation "$repo"
    else
        printf '\nPreview only. Apply this repository with:\n'
        printf '  %q --apply %q\n' "$0" "$requested"
    fi
done
