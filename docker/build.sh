#!/usr/bin/env bash

# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the Autoware-ML container image.
#
#   ./docker/build.sh
#       ghcr.io/tier4/autoware-ml:latest from the locked `dev` pixi
#       environment. This is the image everything else documents.
#
#   ./docker/build.sh --rulebook /path/to/spconv-replacement
#       ghcr.io/tier4/autoware-ml:rulebook from the locked `rulebook`
#       environment: the same stack without spconv, plus `rulebook_torch`
#       compiled from the given checkout for H100 and Blackwell. The checkout
#       is wired in as a named build context because the repository is
#       private; only CUTLASS is fetched during the build. Needs Docker 23 or
#       newer for `--build-context`.
#
# Environment variables:
#   IMAGE            Full image tag to build, overriding the default above.
#   RULEBOOK_SRC     Same as --rulebook.
#   RULEBOOK_ARCHS   CUDA architectures of the extension (default "90;120").
#   RULEBOOK_DTYPES  Kernel operand dtypes (default "f16;bf16").

set -euo pipefail

SCRIPT_DIR=$(readlink -f "$(dirname "$0")")
WORKSPACE_ROOT=$(readlink -f "${SCRIPT_DIR}/..")

RULEBOOK_SRC="${RULEBOOK_SRC:-}"
PIXI_ENV="dev"

print_help() {
    cat <<'EOF'
Usage: build.sh [--rulebook <path to a spconv-replacement checkout>]

  (no option)   Build ghcr.io/tier4/autoware-ml:latest from the locked `dev`
                pixi environment, which includes spconv.
  --rulebook    Build ghcr.io/tier4/autoware-ml:rulebook from the locked
                `rulebook` environment: the same stack without spconv, plus
                `rulebook_torch` compiled from the given checkout for H100 and
                Blackwell. Needs Docker 23 or newer for `--build-context`.

Environment variables: IMAGE, RULEBOOK_SRC, RULEBOOK_ARCHS, RULEBOOK_DTYPES.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
    --help | -h)
        print_help
        exit 0
        ;;
    --rulebook)
        if [ -z "${2:-}" ] || [[ $2 == -* ]]; then
            echo "Error: --rulebook requires the path of a spconv-replacement checkout." >&2
            exit 1
        fi
        RULEBOOK_SRC="$2"
        shift
        ;;
    --rulebook=*)
        RULEBOOK_SRC="${1#*=}"
        ;;
    *)
        echo "Unknown option: $1" >&2
        print_help
        exit 1
        ;;
    esac
    shift
done

BUILD_ARGS=()
BUILD_CONTEXTS=()

if [ -n "$RULEBOOK_SRC" ]; then
    RULEBOOK_SRC=$(readlink -f "$RULEBOOK_SRC")
    if [ ! -d "${RULEBOOK_SRC}/python/rulebook_torch" ]; then
        echo "Error: ${RULEBOOK_SRC} is not a spconv-replacement checkout." >&2
        exit 1
    fi
    PIXI_ENV="rulebook"
    BUILD_CONTEXTS=(--build-context "rulebook=${RULEBOOK_SRC}")
    BUILD_ARGS=(
        --build-arg "RULEBOOK_ARCHS=${RULEBOOK_ARCHS:-90;120}"
        --build-arg "RULEBOOK_DTYPES=${RULEBOOK_DTYPES:-f16;bf16}"
    )
    IMAGE="${IMAGE:-ghcr.io/tier4/autoware-ml:rulebook}"
fi

IMAGE="${IMAGE:-ghcr.io/tier4/autoware-ml:latest}"

DOCKER_BUILDKIT=1 docker build -t "${IMAGE}" -f "${WORKSPACE_ROOT}/docker/Dockerfile" \
    --build-arg "PIXI_ENV=${PIXI_ENV}" \
    "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}" \
    "${BUILD_CONTEXTS[@]+"${BUILD_CONTEXTS[@]}"}" \
    "${WORKSPACE_ROOT}" --progress=plain
