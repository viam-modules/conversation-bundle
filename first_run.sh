#!/bin/bash
set -euo pipefail

OS="$(uname -s)"

if [[ "$OS" == "Linux" ]]; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends libnlopt-dev
elif [[ "$OS" == "Darwin" ]]; then
    brew tap viamrobotics/brews
    brew install nlopt-static
fi
