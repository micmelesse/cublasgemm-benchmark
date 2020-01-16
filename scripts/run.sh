sh scripts/build.sh
cd build

# ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 2048 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024
./cublas_bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 2048 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024