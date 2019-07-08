#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "fp16_conversion.h"
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <random>
#include <string>
using namespace std;

// #define FP16MM

vector<vector<int>> sizes;
//vector<vector<double>> data;
std::vector<double>matrix_initA;
std::vector<double>matrix_initB;
std::vector<double>matrix_initC;

int get_size(string filein){
  std::ifstream infile(filein);
  int m, n, k, lda, ldb, ldc;
 while (infile >> m >> n >> k >> lda >> ldb >> ldc){
  //while (infile >> m ){
    printf("m: %d, n: %d, k: %d, lda: %d, ldb: %d, ldc: %d\n", m, n, k, lda, ldb, ldc);
    printf("m: %d\n", m);
    vector<int> tmp;
    tmp.push_back(m);
    tmp.push_back(n);
    tmp.push_back(k);
    tmp.push_back(lda);
    tmp.push_back(ldb);
    tmp.push_back(ldc);
    sizes.push_back(tmp);
  }
  return 0;
}

double get_hpl_data(double *A, double *B, double *C, int nr_rows_A, int nr_cols_A) {
	std::ifstream matrixA("HPL_0050_A_384x25416.txt");
	std::ifstream matrixB("HPL_0050_B_25416x384.txt");
	std::ifstream matrixC("HPL_0050_C_25416x25416.txt");
	double matrix_init=0;
	//double test1=0.086312158648422788;
	//long long test2=0.086312158648422788;
	//float test3=0.086312158648422788;
	//printf("double %.17g, long long %.17g, float %.17g, original %.17g\n", test1, test2, test3, 0.086312158648422788);
	int count=0;
	if(matrixA.is_open())
		printf("opened: HPL_0000_C_5760x5760.txt, m=%d, n=%d\n", nr_rows_A, nr_cols_A);
	if(matrixB.is_open())
		printf("opened: HPL_0005_C_5760x5760.txt, m=%d, n=%d\n", nr_rows_A, nr_cols_A);
	if(matrixC.is_open())
		printf("opened: HPL_0010_C_5760x5760.txt, m=%d, n=%d\n", nr_rows_A, nr_cols_A);
	while(matrixA >> matrix_init){
	//cout << "line\n" << number << endl;
		if(count<= nr_rows_A * nr_cols_A){
			//matrix_initA.push_back(matrix_init);
			A[count] = matrix_init;
			//output << (reinterpret_cast<double> (matrix_init)) << " " ;
			if(count <=10)
			printf("Matrix A value: %.17g\n", A[count]);
			count++;
		}
		else
			break;
	}
	count = 0;
	while(matrixB >> matrix_init){
	//cout << "line\n" << number << endl;
		if(count<= nr_rows_A * nr_cols_A){
			//matrix_initB.push_back(matrix_init);
			B[count] = matrix_init;
			if(count <=10)
			printf("Matrix B value: %.17g\n", B[count]);
			count++;
		}
		else
			break;
	}
	count = 0;
	while(matrixC >> matrix_init){
	//cout << "line\n" << number << endl;
		if(count<= nr_rows_A * nr_rows_A){
			//matrix_initB.push_back(matrix_init);
			C[count] = matrix_init;
			if(count <=10)
			printf("Matrix C value: %.17g\n", C[count]);
			count++;
		}
		else
			break;
	}
	return 0;
}
	

//hash table for different init
#define random_init 13891919151264982137
#define trig_float 8246975194281651152
#define board_init 8246135241782432000
#define all_zero 7572162586047357
#define narrow_init 13885799058007952913
#define const_init 8246182209224226911
#define hpl_init 7572466080418652
#define rocblas 229481534085259
int count=0;
unsigned long hash_func(const char *str) {
    unsigned long hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
        printf("HASH value '%lu' is valid command.\n", hash);
    return hash;
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}


// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
// random_int range(full range random)
//-1.7976931348623157e308 to + 1.7976931348623157e308
void CPU_fill_rand_init(double *A, int nr_rows_A, int nr_cols_A) {
	int count=1;
	union Float64Rand
	{
		struct {
			unsigned long long Frac0 : 32;
			unsigned long long Frac1 : 20;
			unsigned long long Exp : 11;
			unsigned long long Signed : 1;
		} BitArea;
		double Value;
		//algorithm: (-1)^S*(1.M)*2^(E-1023)     
	} r;

	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		r.Value = 0;

		r.BitArea.Frac0 = rand();
		r.BitArea.Frac1 = rand() % 0x100000;
		r.BitArea.Exp = rand() % 0x7ff;
		r.BitArea.Signed = rand() & 0x1;
		//If mant is out of scope, keep looping

		A[i] = r.Value; 
		if(count<=5){
printf("Matrix init:Frac0 %lld, frac1: %lld, Exp %lld, Value:%e final value %e, random value:%d\n", r.BitArea.Frac0,r.BitArea.Frac1, r.BitArea.Exp, r.Value, A[i], rand());
			count++;
		}
	}

}

/* generate random number :*/
/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
// Fill the array A(nr_rows_A, nr_cols_A) with [1 to 10 ] interger range on CPU
void CPU_fill_rocblas_rand_narrow_init(double *A, int nr_rows_A, int nr_cols_A) {
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(1,10);
	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		A[i] = distribution(generator);
		if(i<=14){
			printf("rocblas init data %e\n", A[i]);
		}
        }
}
// Fill the array A(nr_rows_A, nr_cols_A) with trig random numbers on CPU
void CPU_fill_sin(double *A, int nr_rows_A, int nr_cols_A) {
	int count=1;
	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
               A[i] = (double)sin(i);
		if(count<=4){
			printf("Matrix init %e\n", A[i]);
			count++;
		}
        }
}

// Fill the array A(nr_rows_A, nr_cols_A) with trig random numbers on CPU
void CPU_fill_cos(double *A, int nr_rows_A, int nr_cols_A) {
	int count=1;
	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
               A[i] = (double)cos(i);
		if(count<=4){
			printf("Matrix init %e\n", A[i]);
			count++;
		}
        }
}

// Fill the array A(nr_rows_A, nr_cols_A) with all zero numbers on CPU
void CPU_fill_all_zero(double *A, int nr_rows_A, int nr_cols_A) {
	int count=1;
	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
               A[i] = 0;
		if(count<=4){
			printf("Matrix init %e\n", A[i]);
			count++;
		}
        }
}

// Fill the array A(nr_rows_A, nr_cols_A) with narrow [-2, 2] on CPU
void CPU_fill_narrow_rand(double *A, int nr_rows_A, int nr_cols_A) {

	int count=1;
	std::ofstream output("HPL_0000_narrow_5760x5760.txt");
	union Float64Rand
	{
		struct {
			unsigned long long Frac0 : 32;
			unsigned long long Frac1 : 20;
			unsigned long long Exp : 11;
			unsigned long long Signed : 1;
		} BitArea;
		double Value;
		//algorithm: (-1)^S*(1.M)*2^(E-1023)     
	} r;

	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		r.Value = 0;

		r.BitArea.Frac0 = rand() % 0x100000000;
		r.BitArea.Frac1 = rand() % 0x100000;
		r.BitArea.Exp = (rand() % 0x400) & 0x3FF;
		r.BitArea.Signed = rand() & 0x1;
		//If mant is out of scope, keep looping

		A[i] = r.Value;
		output << A[i] << " ";
		if(count<=5){
			printf("Matrix init:Frac %lld, %lld, Exp %lld, Value:%e final value %e, random value:%d\n", r.BitArea.Frac0, r.BitArea.Frac1, r.BitArea.Exp, r.Value, A[i], rand());
			count++;
		}
	}
}
  
// Fill the array A(nr_rows_A, nr_cols_A) with constant on CPU
void CPU_fill_const_init(double *A, int nr_rows_A, int nr_cols_A) {
	double x = 2.97652966479759093428968129014E1;
	int count=1;
	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		A[i] = x;
		if(count<=4){
			printf("Matrix init %e\n", A[i]);
			count++;
		}
	}
}
  
// Fill the array A(nr_rows_A, nr_cols_A) with constant on CPU
/*void CPU_fill_hpl_init(double *A, int nr_rows_A, int nr_cols_A,) {
	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		A[i] = X[i];
	}
}
*/
  
int main(int argc, char ** argv){

  string filein = argv[1];
  get_size(filein);

  int min_m_k_n ; // min matrix size
  int max_m_k_n ; // max matrix size
  int m1, n1, k1;
  char *init_command= argv[2];
  int repeats = atoi(argv[3]);
  //string datain = argv[4];
  //get_hpl_data(datain, 5760, 5760);
  int verbose = 1;
  unsigned long long command;
  for(vector<vector<int>>::iterator idx = sizes.begin(); idx !=sizes.end(); idx++){
    //printf("m,n:%d, %d\n",(*idx)[0],(*idx)[1]);
    min_m_k_n = (*idx)[0];
    max_m_k_n = (*idx)[1];
    m1 = (*idx)[0];
    n1 = (*idx)[1];
    k1 = (*idx)[2];
  }

#ifndef FP16MM
  cout << "\ncublasDgemm test result:\n" << endl;
#else
  cout << "\ncublasHgemm test result:\n" << endl;
#endif
 command=hash_func(init_command); 
  if(verbose) 
    cout << "running with" 
	 << " min_m_k_n: " << min_m_k_n
	 << " max_m_k_n: " << max_m_k_n
	 << " m: " << m1
	 << " n: " << n1
	 << " k: " << k1
	 << " repeats: " << repeats
	 << endl;

  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  if(verbose) cout << "allocating device variables" << endl;
  
  // Allocate 3 arrays on CPU
	double *h_A = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	double *h_B = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	double *h_C = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	printf("matrix for hpl init...%d, %d\n", max_m_k_n, max_m_k_n);
  switch(command) {
	case random_init:
		printf("Running random_init initialization...\n");
		CPU_fill_rand_init(h_A, max_m_k_n, max_m_k_n);
		CPU_fill_rand_init(h_B, max_m_k_n, max_m_k_n);
		CPU_fill_rand_init(h_C, max_m_k_n, max_m_k_n);
		break;
	case rocblas:
		printf("Running rocBLAS initialization 1 to 10 ...\n");
		CPU_fill_rocblas_rand_narrow_init(h_A, max_m_k_n, max_m_k_n);
		CPU_fill_rocblas_rand_narrow_init(h_B, max_m_k_n, max_m_k_n);
		CPU_fill_rocblas_rand_narrow_init(h_C, max_m_k_n, max_m_k_n);
		break;
	case trig_float:
		printf("Running trig_float initialization...\n");
		CPU_fill_sin(h_A, max_m_k_n, max_m_k_n);
		CPU_fill_cos(h_B, max_m_k_n, max_m_k_n);
		CPU_fill_sin(h_C, max_m_k_n, max_m_k_n);
		break;
	case all_zero:
		printf("Running all_zero initialization...\n");
		CPU_fill_all_zero(h_A, max_m_k_n, max_m_k_n);
		CPU_fill_all_zero(h_B, max_m_k_n, max_m_k_n);
		CPU_fill_all_zero(h_C, max_m_k_n, max_m_k_n);
		break;
	case narrow_init:
		printf("Running Narrow range on A, B, C, m = rand(), e = 1023...\n");
		CPU_fill_narrow_rand(h_A, max_m_k_n, max_m_k_n);
		CPU_fill_narrow_rand(h_B, max_m_k_n, max_m_k_n);
		CPU_fill_narrow_rand(h_C, max_m_k_n, max_m_k_n);
		break;
	case const_init:
		printf("Running const_init initialization...\n");
		CPU_fill_const_init(h_A, max_m_k_n, max_m_k_n);
		CPU_fill_const_init(h_B, max_m_k_n, max_m_k_n);
		CPU_fill_const_init(h_C, max_m_k_n, max_m_k_n);
		break;
	case hpl_init:
		printf("Running hpl initialization...\n");
		get_hpl_data(h_A, h_B, h_C, m1, k1);
		break;
	 default:
		printf("[ERROR] '%s' is not a valid command.\n", init_command);
  }
  
#ifndef FP16MM

	// Allocate 3 arrays on GPU
	double *d_A, *d_B, *d_C;
	checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(double)));
	checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(double)));
	checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(double)));

	checkCuda(cudaMemcpy(d_A,h_A,max_m_k_n * max_m_k_n * sizeof(double),cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B,h_B,max_m_k_n * max_m_k_n * sizeof(double),cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_C,h_C,max_m_k_n * max_m_k_n * sizeof(double),cudaMemcpyHostToDevice));

	double lda, ldb, ldc, m, n, k;
	const double alf = -1.0f;
	const double bet = 1.0f;
	const double *alpha = &alf;
	const double *beta = &bet;
#else
    
  	__half *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(__half)));
    
    for (int i = 0; i < max_m_k_n * max_m_k_n; i++) {
      d_A[i] = approx_double_to_half(h_A[i]);
  	  d_B[i] = approx_double_to_half(h_B[i]);
  	  d_C[i] = approx_double_to_half(h_C[i]);
    }
    
    int lda, ldb, ldc, m, n, k;
    const __half alf = approx_double_to_half(1.0);
    const __half bet = approx_double_to_half(0.0);
    const __half *alpha = &alf;
    const __half *beta = &bet;
	
#endif
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(vector<vector<int>>::iterator idx = sizes.begin(); idx !=sizes.end(); idx++){
    //std::this_thread::sleep_for(std::chrono::milliseconds(50));
    printf("m,n:%d, %d\n",(*idx)[0],(*idx)[1]);
    m = (*idx)[0];
    n = (*idx)[1];
    k = (*idx)[2];
    lda = (*idx)[3];
    ldb = (*idx)[4];
    ldc = (*idx)[5];


printf("m: %e, n: %e, k: %e, lda: %e, ldb: %e, ldc: %e\n", m, n, k, lda, ldb, ldc);
    float sum = 0.0;
    for(int rep = 0; rep < repeats; rep++){
      cudaEventRecord(start, 0);
//	  m=n=k=size;
//	  lda = m;
//	  ldb = k;
//          ldc = m;
#ifndef FP16MM
	//printf("started Dgemm %ll, %ll, %ll, %ll, %ll, %ll\n", m, n, k, lda, ldb, ldc);
	stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc); 
#else
	stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc); 
#endif
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      if(stat != CUBLAS_STATUS_SUCCESS){
	cerr << "cublasDgemmBatched failed" << endl;
	exit(1);
      }
      assert(!cudaGetLastError());
      
      float elapsed;
      cudaEventElapsedTime(&elapsed, start, stop);
      elapsed /= 1000.0f;
      sum += elapsed;
    }
    double time = sum/repeats;


  //printf("end of dgemm %ll, %ll, %ll, %ll, %ll, %ll\n", m, n, k, lda, ldb, ldc);
  double tmp = m*n*k*2;
#ifndef FP16MM	
  cout << " matrix (32): " 
#else
  cout << " matrix (16): " 
#endif
<< " m: " << m << " n: " << n << " k: " << k << " -lda: " << lda << " -ldb: " << ldb << " -ldc: " << ldc << ", ops: " << " average time: " << time << " s " << " GFLOPS: " << (tmp/time)/1e9 << endl;
    cout << "GFLOPS:" << (m*n*k*2/time)/1e9 << endl;
    cout << m*n*k*2/time << endl;
    cout << m*n*k*2/time/1e9 << endl;

  }

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);
  return 0;
}
