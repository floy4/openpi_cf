#!/usr/bin/env bash
set -uo pipefail

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=2

TASK_SUITES=(
  libero_spatial
  libero_object
  libero_goal
  libero_10
)

BASE_DIR="data/pi05_cf_libero"

for suite in "${TASK_SUITES[@]}"; do
  out_dir="${BASE_DIR}/${suite}"

  if [ -d "${out_dir}" ]; then
    echo "[SKIP] ${suite} -> ${out_dir} already exists"
    continue
  fi

  mkdir -p "${out_dir}"
  echo "==== Running ${suite} ===="

  python -u examples/libero/main.py \
    --args.task_suite_name "${suite}" \
    --args.video_out_path "${out_dir}" \
    2>&1 | tee "${out_dir}/run.log"

  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "[OK] ${suite}"
  else
    echo "[FAIL] ${suite}, see ${out_dir}/run.log"
  fi
done
