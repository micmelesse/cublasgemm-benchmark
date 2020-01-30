#!/bin/bash

# use bash
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR_PATH=$(dirname $SCRIPT_PATH)
if [ ! "$BASH_VERSION" ]; then
    echo "Using bash to run this script $0" 1>&2
    exec bash $SCRIPT_PATH "$@"
    exit 1
fi

# set flags
export CUDA_VISIBLE_DEVICES=0 # choose gpu

# add data path
DATA_PATH=$(pwd)/data/rocblas_log_bench_bert_512_hist.csv

# run cublas-bench commands
cd build # switch to cublas-bench directory

# enable persistence mode for apps
nvidia-smi -pm 1

# execute cublas commands
for i in 1 2 3 4 5; do
    OUTPUT_PATH="${DATA_PATH%.*}_Nvidia_GFLOPS_$i.txt"
    while IFS= read -r line; do
        FILTERED_LINE=$(echo $line | grep -oP "./rocblas-bench.*\d")
        if [ ! -z "$FILTERED_LINE" ]; then
            CUBLAS_BENCH_COMMAND="${FILTERED_LINE//rocblas-bench/cublas-bench}"
            source <(echo $CUBLAS_BENCH_COMMAND --iters 100)
        fi
    done <"$DATA_PATH" >"$OUTPUT_PATH"
done

# disable persistence mode for apps
nvidia-smi -pm 0
