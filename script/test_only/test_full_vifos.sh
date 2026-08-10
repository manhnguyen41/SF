#!/usr/bin/env bash

# Convenience wrapper. The canonical implementation is test_one_experiment.sh;
# this file intentionally contains no duplicate paths or model arguments.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

exec bash "$SCRIPT_DIR/test_one_experiment.sh" \
  full_vifos "$PROJECT_ROOT/script/round1/fullvifos.sh"
