#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <hip/hip_runtime.h>
#include "hip/hip_fp16.h"
#include <hipblas.h>
#include "fp16_conversion.h"
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <random>
using namespace std;

// #define FP16MM

vector<vector<int>> sizes;

int get_size(string filein){
	std::ifstream infile(filein);
	int m, n, k, lda, ldb, ldc;
	while (infile >> m >> n >> k >> lda >> ldb >> ldc){
		printf("m: %d, n: %d, k: %d, lda: %d, ldb: %d, ldc: %d\n", m, n, k, lda, ldb, ldc);
		vector<int> tmp;
		tmp.push_back(m);
		tmp.push_back(n);
		tmp.push_back(k);
		tmp.push_back(lda);
		tmp.push_back(ldb);
		tmp.push_back(ldc);
		sizes.push_back(tmp);
	}
}


double get_hpl_data(double *A, double *B, double *C, int nr_rows_A, int nr_cols_A) {
        std::ifstream matrixA("HPL_0000_C_44616x44616.txt");
        std::ifstream matrixB("HPL_0000_C_44616x44616.txt");
        std::ifstream matrixC("HPL_0000_C_44616x44616.txt");
        double matrix_init=0;
        //double test1=0.086312158648422788;
        //long long test2=0.086312158648422788;
        //float test3=0.086312158648422788;
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
	printf("Matrix A value: %d\n", count);
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
	printf("Matrix B value: %d\n", count);
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
	printf("Matrix C value: %d\n", count);
        return 0;
}

//hash table for different init
#define rand1 210726353755
#define trig1 210729331180
#define broad1 6953363702686
#define zeros1 6954287658697
#define narrow1 229475838685615
#define const1 6953399264509
#define hpl1 6385303930
#define rocblas 229481534085259
unsigned long hash_func(const char *str) {
	unsigned long hash = 5381;
	int c;

	while ((c = *str++))
		hash = ((hash << 5) + hash) + c;
	printf("HASH value '%lu' is valid command.\n", hash);
	return hash;
}

const char* cublasGetErrorString(hipblasStatus_t status)
{
	switch(status)
	{
		case HIPBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case HIPBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case HIPBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case HIPBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
		case HIPBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
		case HIPBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case HIPBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
		case HIPBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
	}
	return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
hipError_t checkCuda(hipError_t result)
{
	if (result != hipSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", hipGetErrorString(result));
		assert(result == hipSuccess);
	}
	return result;
}

inline
hipblasStatus_t checkCublas(hipblasStatus_t result)
{
	if (result != HIPBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
		assert(result == HIPBLAS_STATUS_SUCCESS);
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
                        int Frac0 : 32;
                        int Frac1 : 20;
                        int Exp : 11;
                        int Signed : 1;
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
                if(count<=100){
			cout << "Matrix init values:" << A[i] << endl;
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

	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
               A[i] = (double)sin(i);
        }
}

// Fill the array A(nr_rows_A, nr_cols_A) with trig random numbers on CPU
void CPU_fill_cos(double *A, int nr_rows_A, int nr_cols_A) {

	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
               A[i] = (double)cos(i);
        }
}

// Fill the array A(nr_rows_A, nr_cols_A) with all zero numbers on CPU
void CPU_fill_all_zero(double *A, int nr_rows_A, int nr_cols_A) {

	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
               A[i] = 0;
        }
}

// Fill the array A(nr_rows_A, nr_cols_A) with narrow [-3, 3] on CPU
void CPU_fill_narrow_rand(double *A, int nr_rows_A, int nr_cols_A) {


        int count=1;
        union Float64Rand
        {
                struct {
                        unsigned int Frac0 : 32;
                        unsigned int Frac1 : 20;
                        unsigned int Exp : 11;
                        unsigned int Signed : 1;
                } BitArea;
                double Value;
        } r;

	cout << "Matrix and prints to check numbers bw -3 to 3" << endl;
        for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
                r.Value = 0;

                r.BitArea.Frac0 = rand();
		r.BitArea.Frac1 = rand() % 0x100000;
                r.BitArea.Exp = rand() % 0x401;
		if (r.BitArea.Exp == 0x400){
			r.BitArea.Frac1 &= 0x7ffff;
			r.BitArea.Frac0 &= 0xffffffff;
		} 
                r.BitArea.Signed = rand() & 0x1;
                //If mant is out of scope, keep looping

                A[i] = r.Value;
                if(count<= 200){
		if (r.BitArea.Exp == 0x400){
			cout << "Matrix init values:" << r.Value << endl;
			count++;
                }
		}
        }
}

// Fill the array A(nr_rows_A, nr_cols_A) with constant on CPU
void CPU_fill_const_init(double *A, int nr_rows_A, int nr_cols_A) {
	double x = 2.97652966479759093428968129014E1;
	for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		A[i] = x;
	}
}
  
int main(int argc, char ** argv){

	string filein = argv[1];
	get_size(filein);

	int min_m_k_n; // min matrix size
	int max_m_k_n; // max matrix size
	int m1, n1, k1;
	char *init_command= argv[2];
	int repeats = atoi(argv[3]);
	int verbose = 1;

	for(vector<vector<int>>::iterator idx = sizes.begin(); idx !=sizes.end(); idx++){
		//printf("m,n:%d, %d\n",(*idx)[0],(*idx)[1]);
		min_m_k_n = (*idx)[0];
		max_m_k_n = (*idx)[1];
		m1 = (*idx)[0];
		n1 = (*idx)[1];
		k1 = (*idx)[2];
	}

	cout << "\nhipblasDgemm test result:\n" << endl;
  
	if(verbose) 
	cout << "running with" 
		 << " min_m_k_n: " << min_m_k_n
		 << " max_m_k_n: " << max_m_k_n
		 << " m: " << m1
		 << " n: " << n1
		 << " k: " << k1
		 << " repeats: " << repeats
		 << endl;

	hipblasStatus_t stat;
	hipblasHandle_t handle;

	checkCublas(hipblasCreate(&handle));

	if(verbose) cout << "allocating device variables" << endl;
  
	// Allocate 3 arrays on CPU

	double *h_A = (double *)malloc(max_m_k_n * k1 * sizeof(double));
	double *h_B = (double *)malloc(max_m_k_n * k1 * sizeof(double));
	double *h_C = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	printf("matrix for hpl init...%d, %d\n", max_m_k_n, max_m_k_n);
	switch(hash_func(init_command)) {

		case rand1:
			printf("Running random_init initialization...\n");
			CPU_fill_rand_init(h_A, max_m_k_n, k1);
			CPU_fill_rand_init(h_B, max_m_k_n, k1);
			CPU_fill_rand_init(h_C, max_m_k_n, max_m_k_n);
			break;
		case rocblas:
			printf("Running rocBLAS initialization 1 to 10 ...\n");
			CPU_fill_rocblas_rand_narrow_init(h_A, max_m_k_n, k1);
			CPU_fill_rocblas_rand_narrow_init(h_B, max_m_k_n, k1);
			CPU_fill_rocblas_rand_narrow_init(h_C, max_m_k_n, max_m_k_n);
			break;
		case trig1:
			printf("Running trig_float initialization...\n");
			CPU_fill_sin(h_A, max_m_k_n, max_m_k_n);
			CPU_fill_cos(h_B, max_m_k_n, max_m_k_n);
			CPU_fill_sin(h_C, max_m_k_n, max_m_k_n);
			break;
		case zeros1:
			printf("Running all_zero initialization...\n");
			CPU_fill_all_zero(h_A, max_m_k_n, max_m_k_n);
			CPU_fill_all_zero(h_B, max_m_k_n, max_m_k_n);
			CPU_fill_all_zero(h_C, max_m_k_n, max_m_k_n);
			break;
		case narrow1:
			printf("Running Narrow range on A, B, C, m = rand(), e = 1023...\n");
			CPU_fill_narrow_rand(h_A, max_m_k_n, max_m_k_n);
			CPU_fill_narrow_rand(h_B, max_m_k_n, max_m_k_n);
			CPU_fill_narrow_rand(h_C, max_m_k_n, max_m_k_n);
			break;
		case const1:
			printf("Running const_init initialization...\n");
			CPU_fill_const_init(h_A, max_m_k_n, max_m_k_n);
			CPU_fill_const_init(h_B, max_m_k_n, max_m_k_n);
			CPU_fill_const_init(h_C, max_m_k_n, max_m_k_n);
			break;
		case hpl1:
			printf("Running hpl initialization...\n");
			get_hpl_data(h_A, h_B, h_C, m1, k1);
			break;
		default:
			printf("[ERROR] '%s' is not a valid command.\n", init_command);
	}
	// Allocate 3 arrays on GPU
	printf("Running venkat0...\n");
	double *d_A, *d_B, *d_C;
	checkCuda(hipMalloc(&d_A, max_m_k_n * k1 * sizeof(double)));
	checkCuda(hipMalloc(&d_B, max_m_k_n * k1 * sizeof(double)));
	checkCuda(hipMalloc(&d_C, max_m_k_n * max_m_k_n * sizeof(double)));

	checkCuda(hipMemcpy(d_A,h_A,max_m_k_n * k1 * sizeof(double),hipMemcpyHostToDevice));
	checkCuda(hipMemcpy(d_B,h_B,max_m_k_n * k1 * sizeof(double),hipMemcpyHostToDevice));
	checkCuda(hipMemcpy(d_C,h_C,max_m_k_n * max_m_k_n * sizeof(double),hipMemcpyHostToDevice));

	double lda, ldb, ldc, m, n, k;
	const double alf = -1.0f;
	const double bet = 1.0f;
	const double *alpha = &alf;
	const double *beta = &bet;

	printf("Running venkat1...\n");
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	//  for(int size = min_m_k_n; size <= max_m_k_n; size=size+64){ // step size

	printf("Running venkat2...\n");
	for(vector<vector<int>>::iterator idx = sizes.begin(); idx !=sizes.end(); idx++){
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		//printf("m,n:%d, %d\n",(*idx)[0],(*idx)[1]);
		m = (*idx)[0];
		n = (*idx)[1];
		k = (*idx)[2];
		lda = (*idx)[3];
		ldb = (*idx)[4];
		ldc = (*idx)[5];

		float sum = 0.0;
		for(int rep = 0; rep < repeats; rep++){
			hipEventRecord(start, 0);
			stat = hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, m, n, k, alpha, \
							d_A, lda, d_B, ldb, beta, d_C, ldc); 
		hipEventRecord(stop,0);
		hipEventSynchronize(stop);
		if(stat != HIPBLAS_STATUS_SUCCESS){
			cerr << "hipblasDgemmBatched failed" << endl;
			exit(1);
		}
	assert(!hipGetLastError());

	float elapsed;
	hipEventElapsedTime(&elapsed, start, stop);
	elapsed /= 1000.0f;
	sum += elapsed;
	}
	double time = sum/repeats;

	printf("Running venkat4...\n");

	//printf("end of dgemm %ll, %ll, %ll, %ll, %ll, %ll\n", m, n, k, lda, ldb, ldc);
	double tmp = m*n*k*2;
	cout << " matrix (64 bit): " 
		<< " m: " << m << " n: " << n << " k: " << k \
		<< " -lda: " << lda << " -ldb: " << ldb << " -ldc: " << ldc \
		<< ", ops: " << " average time: " << time << " s " \
		<< " GFLOPS: " << (tmp/time)/1e9 << endl;
	}

	//Free GPU memory
	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);
      
	return 0;
}
