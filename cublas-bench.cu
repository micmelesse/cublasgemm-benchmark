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

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A, int batch) {
  int a = 1;

  for (int i = 0; i < nr_rows_A * nr_cols_A * batch; i++) {
    A[i] = (float)rand() / (float)(RAND_MAX / a);
  }
}

int main(int argc, char **argv) {

  // print device info
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Device " << i << ": " << prop.name << std::endl;
  }

  // parse input arguments
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
  opp_adder("stride_a", "stride_a", cxxopts::value<long long int>());
  opp_adder("stride_b", "stride_c", cxxopts::value<long long int>());
  opp_adder("stride_c", "stride_c", cxxopts::value<long long int>());
  opp_adder("batch", "batch", cxxopts::value<int>());

  auto result = options.parse(argc, argv);

  std::string f = result["f"].as<std::string>();
  std::string r = result["r"].as<std::string>();
  std::string transposeA_str = result["transposeA"].as<std::string>();
  cublasOperation_t transposeA =
      (transposeA_str == "T" ? CUBLAS_OP_T : CUBLAS_OP_N);
  std::string transposeB_str = result["transposeB"].as<std::string>();
  cublasOperation_t transposeB =
      (transposeB_str == "T" ? CUBLAS_OP_T : CUBLAS_OP_N);
  int m = result["m"].as<int>();
  int n = result["n"].as<int>();
  int k = result["k"].as<int>();
  float alpha = result["alpha"].as<float>();
  int lda = result["lda"].as<int>();
  int ldb = result["ldb"].as<int>();
  float beta = result["beta"].as<float>();
  int ldc = result["ldc"].as<int>();

  long long int stride_a, stride_b, stride_c;
  int batch;
  int is_batched = result.count("batch");
  if (is_batched) {
    stride_a = result["stride_a"].as<long long int>();
    stride_b = result["stride_b"].as<long long int>();
    stride_c = result["stride_c"].as<long long int>();
    batch = result["batch"].as<int>();
  }

  // execute paramters
  int repeats = 100;

  cublasStatus_t stat;
  cublasHandle_t handle;

  // allocate buffers
  checkCublas(cublasCreate(&handle));
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  if (is_batched) {
    // Allocate 3 arrays on CPU
    h_A = (float *)malloc(batch * m * k * sizeof(float));
    h_B = (float *)malloc(batch * k * n * sizeof(float));
    h_C = (float *)malloc(batch * m * n * sizeof(float));

    CPU_fill_rand(h_A, m, k, batch);
    CPU_fill_rand(h_B, k, n, batch);
    CPU_fill_rand(h_C, m, n, batch);

    // Allocate 3 arrays on GPU
    checkCuda(cudaMallocManaged(&d_A, batch * m * k * sizeof(float)));
    checkCuda(cudaMallocManaged(&d_B, batch * k * n * sizeof(float)));
    checkCuda(cudaMallocManaged(&d_C, batch * m * n * sizeof(float)));

    // copy data to GPU
    checkCuda(cudaMemcpy(d_A, h_A, batch * m * k * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, batch * k * n * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_C, h_C, batch * m * n * sizeof(float),
                         cudaMemcpyHostToDevice));
  } else {
    // Allocate 3 arrays on CPU
    h_A = (float *)malloc(m * k * sizeof(float));
    h_B = (float *)malloc(k * n * sizeof(float));
    h_C = (float *)malloc(m * n * sizeof(float));

    CPU_fill_rand(h_A, m, k);
    CPU_fill_rand(h_B, k, n);
    CPU_fill_rand(h_C, m, n);

    // Allocate 3 arrays on GPU
    checkCuda(cudaMallocManaged(&d_A, m * k * sizeof(float)));
    checkCuda(cudaMallocManaged(&d_B, k * n * sizeof(float)));
    checkCuda(cudaMallocManaged(&d_C, m * n * sizeof(float)));

    // copy data to GPU
    checkCuda(
        cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(
        cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(
        cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice));
  }

  // call Sgemm repeatdly
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float totalTime_ms = 0.0;
  for (int rep = 0; rep < repeats; rep++) {

    if (is_batched) {
      cudaEventRecord(start, 0);
      stat = cublasSgemmStridedBatched(
          handle, transposeA, transposeB, m, n, k, &alpha, d_A, lda, stride_a,
          d_B, ldb, stride_b, &beta, d_C, ldc, stride_c, batch);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
    } else {
      cudaEventRecord(start, 0);
      stat = cublasSgemm(handle, transposeA, transposeB, m, n, k, &alpha, d_A,
                         lda, d_B, ldb, &beta, d_C, ldc);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
    }

    if (stat != CUBLAS_STATUS_SUCCESS) {
      if (is_batched) {
        std::cerr << "cublasSgemmStridedBatched failed: " << transposeA_str
                  << "," << transposeB_str << "," << m << "," << n << "," << k
                  << "," << alpha << "," << lda << "," << stride_a << "," << ldb
                  << "," << stride_b << ","
                  << "," << beta << "," << ldc << stride_c << "," << batch
                  << std::endl;
      } else {
        std::cerr << "cublasSgemm failed: " << transposeA_str << ","
                  << transposeB_str << "," << m << "," << n << "," << k << ","
                  << alpha << "," << lda << "," << ldb << "," << beta << ","
                  << ldc << std::endl;
      }

      exit(1);
    }

    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
      std::cerr << cudaGetErrorString(lastError) << std::endl;
    }

    float elapsedTime_ms;
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    totalTime_ms += elapsedTime_ms;
  }

  // calculate avg time and gflops
  float avgTime_ms = totalTime_ms / repeats;
  float avgTime_s = avgTime_ms / 1000.0f;
  float avgTime_us = avgTime_ms * 1000.0f;
  float totalSize;

  if (is_batched) {
    totalSize = batch * static_cast<float>(m) * static_cast<float>(n) *
                static_cast<float>(k);
  } else {
    totalSize =
        static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(k);
  }

  float gflop = totalSize * 2.0f / 1e9;
  float gflopPerSec = gflop / avgTime_s;

  if (is_batched) {
    std::cout << transposeA_str << "," << transposeB_str << "," << m << "," << n
              << "," << k << "," << alpha << "," << lda << "," << stride_a
              << "," << ldb << "," << stride_b << "," << beta << "," << ldc
              << "," << stride_c << "," << batch << "," << gflopPerSec << ","
              << avgTime_us << std::endl;
  } else {
    std::cout << transposeA_str << "," << transposeB_str << "," << m << "," << n
              << "," << k << "," << alpha << "," << lda << "," << ldb << ","
              << beta << "," << ldc << "," << gflopPerSec << "," << avgTime_us
              << std::endl;
  }

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
