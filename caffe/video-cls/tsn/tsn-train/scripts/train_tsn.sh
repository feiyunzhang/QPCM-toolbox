#!/usr/bin/env bash

DATASET=$1
MODALITY=$2
GPU_NUM=$3

TOOLS=lib/caffe-action/build/install/bin
LOG_FILE=logs/${DATASET}_${MODALITY}_split1.log
MPI_BIN_DIR= #/usr/local/openmpi/bin/


echo "logging to ${LOG_FILE}"

if [ "$GPU_NUM" -gt 1 ]; then
  ${MPI_BIN_DIR}mpirun -np $GPU_NUM --allow-run-as-root\
  $TOOLS/caffe train --solver=models/${DATASET}/tsn_bn_inception_${MODALITY}_solver.prototxt  \
     --weights=models/bn_inception_${MODALITY}_init.caffemodel 2>&1 | tee ${LOG_FILE}
else
  $TOOLS/caffe train --solver=models/${DATASET}/tsn_bn_inception_${MODALITY}_solver.prototxt  \
     --weights=models/bn_inception_${MODALITY}_init.caffemodel 2>&1 | tee ${LOG_FILE}
fi
