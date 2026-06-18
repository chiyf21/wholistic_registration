#!/bin/bash
# Job payload for the F260517 smoke test on a Janelia GPU node.
# Submit with bsub (see the bsub line in the chat / below). Runs the N-frame
# smoke test defined by N_FRAMES_LIMIT in test_F260517_v2.py.
set -euo pipefail

unset XDG_RUNTIME_DIR || true

source /groups/ahrens/home/ruttenv/miniforge3/etc/profile.d/conda.sh
conda activate wholistic-registration

cd /groups/ahrens/home/ruttenv/python_packages/wholistic_registration

echo "host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

python -u src/wholistic_registration/tests/test_F260517_v2.py
