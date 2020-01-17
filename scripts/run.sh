sh scripts/build.sh
sh scripts/cublas_bench.sh
# cd build
# ./cublas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 512 -k 512 --alpha 1 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0 --ldc 64 --stride_c 32768 --batch 16
