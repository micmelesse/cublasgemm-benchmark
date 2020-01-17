#!/bin/bash

# use bash
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR_PATH=$(dirname $SCRIPT_PATH)
if [ ! "$BASH_VERSION" ]; then
    echo "Using bash to run this script $0" 1>&2
    exec bash $SCRIPT_PATH "$@"
    exit 1
fi

# add data path
DATA_PATH=$(pwd)/data/rocblas_log_bench_bert_512_hist.csv
OUTPUT_PATH="${DATA_PATH%.*}_Nvidia_GFLOPS.txt"

# set flags
export CUDA_VISIBLE_DEVICES=0 # choose gpu

# run cublas-bench commands
cd build # switch to cublas-bench directory
while IFS= read -r line; do
    FILTERED_LINE=$(echo $line | grep -oP "./rocblas-bench.*\d")
    CUBLAS_BENCH_COMMAND="${FILTERED_LINE//rocblas-bench/cublas-bench}"
    source <(echo $CUBLAS_BENCH_COMMAND)
done <"$DATA_PATH" >"$OUTPUT_PATH"
