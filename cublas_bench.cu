#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

#include "cxxopts.hpp"
#include "fp16_conversion.h"

using namespace std;

// #define FP16MM

const char *cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline cublasStatus_t checkCublas(cublasStatus_t result) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
  int a = 1;

  for (int i = 0; i < nr_rows_A * nr_cols_A; i++) {
    A[i] = (float)rand() / (float)(RAND_MAX / a);
  }
}

int main(int argc, char **argv) {

  cxxopts::Options options("cublas_bench", "Benchmark Cublas");
  auto opp_adder = options.add_options();
  opp_adder("f", "f", cxxopts::value<std::string>());
  opp_adder("r", "r", cxxopts::value<std::string>());
  opp_adder("transposeA", "transposeA", cxxopts::value<std::string>());
  opp_adder("transposeB", "transposeB", cxxopts::value<std::string>());
  opp_adder("m", "M", cxxopts::value<int>());
  opp_adder("n", "N", cxxopts::value<int>());
  opp_adder("k", "K", cxxopts::value<int>());
  opp_adder("alpha", "alpha", cxxopts::value<float>());
  opp_adder("lda", "lda", cxxopts::value<int>());
  opp_adder("ldb", "ldb", cxxopts::value<int>());
  opp_adder("beta", "beta", cxxopts::value<float>());
  opp_adder("ldc", "ldc", cxxopts::value<int>());

  auto result = options.parse(argc, argv);

  std::string f = result["f"].as<std::string>();
  std::string r = result["r"].as<std::string>();
  std::string transposeA_str = result["transposeA"].as<std::string>();
  int transposeA = (transposeA_str == "T" ? 1 : 0);
  std::string transposeB_str = result["transposeB"].as<std::string>();
  int transposeB = (transposeB_str == "T" ? 1 : 0);
  int m = result["m"].as<int>();
  int n = result["n"].as<int>();
  int k = result["k"].as<int>();
  float alpha = result["alpha"].as<float>();
  int lda = result["lda"].as<int>();
  int ldb = result["ldc"].as<int>();
  float beta = result["beta"].as<float>();
  int ldc = result["ldc"].as<int>();

  int repeats = 100;

  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  // Allocate 3 arrays on CPU

  float *h_A = (float *)malloc(m * k * sizeof(float));
  float *h_B = (float *)malloc(k * n * sizeof(float));
  float *h_C = (float *)malloc(m * n * sizeof(float));

  CPU_fill_rand(h_A, m, k);
  CPU_fill_rand(h_B, k, n);
  CPU_fill_rand(h_C, m, n);

  // Allocate 3 arrays on GPU
  float *d_A, *d_B, *d_C;
  checkCuda(cudaMallocManaged(&d_A, m * k * sizeof(float)));
  checkCuda(cudaMallocManaged(&d_B, k * n * sizeof(float)));
  checkCuda(cudaMallocManaged(&d_C, m * n * sizeof(float)));

  checkCuda(
      cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(
      cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(
      cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float totalTime_ms = 0.0;
  for (int rep = 0; rep < repeats; rep++) {
    cudaEventRecord(start, 0);
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A,
                       lda, d_B, ldb, &beta, d_C, ldc);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if (stat != CUBLAS_STATUS_SUCCESS) {
      cerr << "cublasSgemmBatched failed" << endl;
      exit(1);
    }
    assert(!cudaGetLastError());

    float elapsedTime_ms;
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    totalTime_ms += elapsedTime_ms;
  }

  float avgTime_ms = totalTime_ms / repeats;
  float avgTime_s = avgTime_ms / 1000.0f;
  float avgTime_us = avgTime_ms * 1000.0f;
  float totalSize =
      static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(k);
  float gflop = totalSize * 2.0f / 1e9;
  float gflopPerSec = gflop / avgTime_s;

  std::cout << transposeA_str << "," << transposeB_str << "," << m << "," << n
            << "," << k << "," << alpha << "," << lda << "," << ldb << ","
            << beta << "," << ldc << "," << gflopPerSec << "," << avgTime_us
            << std::endl;

  // Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
