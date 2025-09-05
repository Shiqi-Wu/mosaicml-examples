#!/usr/bin/env bash
set -euo pipefail

### ====== 必改区 ======
NSCC_USER="e0983425"
NSCC_HOST="aspire-login"     # 例如 as-shared-login.nscc.sg 或你们的登录节点主机名
# 你本地电脑上的代码与数据路径（请改成本地真实路径）
SRC_CODE="/home/shiqi_w/codes/MSRA/mosaicml-examples/examples/benchmarks/bert"
SRC_DATA="/home/shiqi_w/nfs/data/dnabert_2_pretrain_full"
### ====================

# NSCC 上的目标路径（和你之前保持一致）
DEST_CODE="/home/users/nus/${NSCC_USER}/mosaicml-examples/examples/benchmarks/bert"
DEST_DATA="/home/users/nus/${NSCC_USER}/dnabert_2_pretrain_full"

echo "[INFO] Upload code -> ${NSCC_USER}@${NSCC_HOST}:${DEST_CODE}"
rsync -avP "${SRC_CODE}/" "${NSCC_USER}@${NSCC_HOST}:${DEST_CODE}/"

echo "[INFO] Upload data -> ${NSCC_USER}@${NSCC_HOST}:${DEST_DATA}"
rsync -avP "${SRC_DATA}/" "${NSCC_USER}@${NSCC_HOST}:${DEST_DATA}/"

echo "[DONE] Upload finished."
