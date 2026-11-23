#!/usr/bin/env bash
# ARCHIVED: one-off migration that produced the maia2-blitz and maia2-rapid
# repositories from the original combined Maia 2 checkpoint. Kept for reference
# and reproducibility; it is not needed for new checkpoints.
# Prepare separate, locally validated Maia 2 blitz and rapid repositories.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/archive/migrate_maia2_checkpoint_repo.sh SOURCE OUTPUT_ROOT
  scripts/archive/migrate_maia2_checkpoint_repo.sh --apply SOURCE OUTPUT_ROOT

Preview is the default. On --apply, OUTPUT_ROOT/maia2-blitz and
OUTPUT_ROOT/maia2-rapid must not already exist. No repository is created,
renamed, committed, or pushed on Hugging Face.
EOF
}

apply=false
if [[ ${1:-} == --apply ]]; then
    apply=true
    shift
fi
if (($# != 2)); then
    usage >&2
    exit 2
fi

source_repo=$(cd -- "$1" && pwd)
output_root=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
chessnets_root=$(cd -- "$script_dir/../.." && pwd)
python_bin=${PYTHON_BIN:-python}
python_path="$chessnets_root/src"

[[ -f "$source_repo/config.yaml" ]] || { echo "Missing config.yaml" >&2; exit 1; }
for variant in blitz rapid; do
    [[ -f "$source_repo/${variant}_model.pt" ]] || {
        echo "Missing ${variant}_model.pt" >&2
        exit 1
    }
done
if [[ -n $(git -C "$source_repo" status --porcelain) ]]; then
    echo "Refusing dirty source repository: $source_repo" >&2
    exit 1
fi

for variant in blitz rapid; do
    target="$output_root/maia2-$variant"
    source_url=$(
        [[ $variant == blitz ]] &&
            echo 'https://drive.google.com/uc?id=1X-Z4J3PX3MQFJoa8gRt3aL8CIH0PWoyt' ||
            echo 'https://drive.google.com/uc?id=1gbC1-c7c0EQOPPAVpGWubezeEW8grVwc'
    )
    if $apply; then
        [[ ! -e "$target" ]] || { echo "Refusing existing target: $target" >&2; exit 1; }
        mkdir -p -- "$target/original" "$target/training"
        cp -- "$source_repo/${variant}_model.pt" "$target/original/model.pt"
        cp -- "$source_repo/config.yaml" "$target/training/original_config.yaml"
        PYTHONPATH="$python_path${PYTHONPATH:+:$PYTHONPATH}" \
            "$python_bin" -m chessnets.loaders.convert_maia2 \
            "$target/original/model.pt" \
            "$target/training/original_config.yaml" \
            "$target" --time-control "$variant" --source-url "$source_url" \
            --training-config-url \
            'https://drive.google.com/uc?id=1GQTskYMVMubNwZH2Bi6AmevI15CS6gk0'
        "$python_bin" "$chessnets_root/scripts/update_model_card.py" "$target" \
            --write --family maia2 --repository-id "shermansiu/maia2-$variant"
        cp -- "$source_repo/.gitattributes" "$target/.gitattributes"
    else
        PYTHONPATH="$python_path${PYTHONPATH:+:$PYTHONPATH}" \
            "$python_bin" -m chessnets.loaders.convert_maia2 \
            "$source_repo/${variant}_model.pt" "$source_repo/config.yaml" \
            "$target" --time-control "$variant" --source-url "$source_url" \
            --training-config-url \
            'https://drive.google.com/uc?id=1GQTskYMVMubNwZH2Bi6AmevI15CS6gk0' \
            --dry-run
    fi
done
