/********************************
* Author: Julian Soh
* Date: Fall quarter 2026
* EE P 524 - GPU Programming
* Homework 1 - CUDA Kernels
* *******************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Homework-1.h"

#include <stdio.h>
#include <cstdlib>
#include <cmath>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t saxpyWithCuda(float* x, float* y, float a, unsigned int N);
cudaError_t Matrix2DWithCuda(float* A, float* B, float* C, int M, int N);
bool runSaxpyCase(unsigned int N, float saxpyA);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void saxpyKernel(float* x, float* y, float a, unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void Matrix2DWithCudaKernel(float* A, float* B, float* C, int M, int N)
{
	// Implement 2D matrix addition kernel here
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned int>(M * N))
	{
		C[i] = A[i] + B[i];
	}
}

int main()
{
	//Variables for vector addition
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}
	checkCuda(cudaStatus, "addWithCuda");

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    //////////////////////////////
	//Homework 1 - CUDA Kernels //
    /////////////////////////////

    //SAXPY Kernel Use Cases
    float saxpyA = 2.0f;
    unsigned int testSizes[] = { 10, 1000, 10000000 };
    std::srand(42);

    for (unsigned int i = 0; i < sizeof(testSizes) / sizeof(testSizes[0]); i++) {
        runSaxpyCase(testSizes[i], saxpyA);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    checkCuda(cudaStatus, "cudaDeviceReset");

    // 2D matrix addition kernel use cases
	int M = 4;
    int N = 4;
	float* A = new float[M * N];
    float* B = new float[M * N];
    float* C = new float[M * N];
	float* C_h = new float[M * N];  // Host result for verification because C will be overwritten by device result
	std::srand(42);
    for (int i = 0; i < M * N; i++) {
        A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        B[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
	for (int i = 0; i < M * N; i++) {
		C_h[i] = A[i] + B[i];  // Host computation for verification
	}

	cudaStatus = Matrix2DWithCuda(A, B, C, M, N);
	checkCuda(cudaStatus, "Matrix2DWithCuda");

    int matrixMismatchCount = 0;
    const float matrixEpsilon = 1e-5f;
    for (int i = 0; i < M * N; i++) {
        if (std::fabs(C[i] - C_h[i]) > matrixEpsilon) {
            matrixMismatchCount++;
        }
    }

    if (matrixMismatchCount == 0) {
        printf("Matrix2D check PASSED for %dx%d\n", M, N);
    }
    else {
        printf("Matrix2D check FAILED for %dx%d: %d mismatches out of %d\n", M, N, matrixMismatchCount, M * N);
        for (int i = 0; i < M * N; i++) {
            if (std::fabs(C[i] - C_h[i]) > matrixEpsilon) {
                int row = i / N;
                int col = i % N;
                printf("Mismatch at (%d,%d): C_host=%f, C_device=%f\n", row, col, C_h[i], C[i]);
            }
        }
    }


    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_h;

    cudaStatus = cudaDeviceReset();
    checkCuda(cudaStatus, "cudaDeviceReset");

	// 3D matrix addition kernel use cases
  
    return 0;
}

bool runSaxpyCase(unsigned int N, float saxpyA)
{
    float* x = new float[N];
    float* y = new float[N];
    float* y_h = new float[N];

    for (unsigned int i = 0; i < N; i++) {
        x[i] = static_cast<float>(std::rand()) / RAND_MAX;
        y[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    for (unsigned int i = 0; i < N; i++) {
        y_h[i] = saxpyA * x[i] + y[i];
    }

    cudaError_t cudaStatus = saxpyWithCuda(x, y, saxpyA, N);
    checkCuda(cudaStatus, "saxpyWithCuda");
    if (cudaStatus != cudaSuccess) {
        delete[] x;
        delete[] y;
        delete[] y_h;
        return false;
    }

    int mismatchCount = 0;
    const float epsilon = 1e-5f;
    for (unsigned int i = 0; i < N; i++) {
        if (std::fabs(y[i] - y_h[i]) > epsilon) {
            mismatchCount++;
        }
    }

    if (mismatchCount == 0) {
        printf("SAXPY check PASSED for N=%u\n", N);
    }
    else {
        printf("SAXPY check FAILED for N=%u: %d mismatches out of %u\n", N, mismatchCount, N);
    }

    delete[] x;
    delete[] y;
    delete[] y_h;
    return mismatchCount == 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


//Homework 1 - CUDA Kernels
cudaError_t saxpyWithCuda(float* x, float* y, float a, unsigned int N)
{
    float* x_d = nullptr;
    float* y_d = nullptr;

	// Part 1: Allocate device memory and copy from host memory to device memory
    cudaMalloc((void**)&x_d, N * sizeof(float));
    cudaMalloc((void**)&y_d, N * sizeof(float));
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

	// Part 2: Call kernel - to launch a grid of threads to perform SAXPY operation
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    saxpyKernel<<<blocksPerGrid, threadsPerBlock>>>(x_d, y_d, a, N);

	// Part 3: Copy from device memory to host memory
    cudaMemcpy(y, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

	// Part 4: Free device memory
    cudaFree(x_d);
    cudaFree(y_d);

    return cudaSuccess;
}

cudaError_t Matrix2DWithCuda(float* A, float* B, float* C, int M, int N)
{
	float* A_d = nullptr;
	float* B_d = nullptr;
	float* C_d = nullptr;

    // Part 1: Allocate device memory and copy from host memory to device memory
    cudaMalloc((void**)&A_d, M * N * sizeof(float));
    cudaMalloc((void**)&B_d, M * N * sizeof(float));
    cudaMalloc((void**)&C_d, M * N * sizeof(float));
    cudaMemcpy(A_d, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Part 2: Call kernel - to launch a grid of threads to perform 2D operation
    int threadsPerBlock = 256;
    int totalElements = M * N;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
	Matrix2DWithCudaKernel << <blocksPerGrid, threadsPerBlock >> > (A_d, B_d, C_d, M, N);

    // Part 3: Copy from device memory to host memory
    cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Part 4: Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

	return cudaSuccess;
}