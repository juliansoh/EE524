
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Homework-1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
cudaError_t addWithSaxpy(float* y, const float* x, float a, int N);
cudaError_t add2withcuda(float* c, const float* a, const float* b, int M, int N);
cudaError_t add3dwithcuda(float* c, const float* a, const float* b, int M, int N, int P);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int threadIdxX = threadIdx.x;
    int blockIdxX = blockIdx.x;
    int globalIdx = blockIdxX * blockDim.x + threadIdxX;
    printf("Thread ID x: %d, Block ID x: %d, GTID: %d\n", threadIdxX, blockIdxX, globalIdx);

    c[globalIdx] = a[globalIdx] + b[globalIdx];
}

void addCpu(int* cpuResult, const int* a, const int* b, unsigned int size)
{
    for (unsigned int i = 0; i < size; i++) {
        cpuResult[i] = a[i] + b[i];
    }
}

// Compare CPU and GPU results using combined relative and absolute tolerances.
int withinThreshold(float cpuResult, float gpuResult)
{
    float rTol = 1e-12f;
    float aTol = 1e-15f;
    float cpuMagnitude = fabsf(cpuResult);
    float gpuMagnitude = fabsf(gpuResult);
    float maxMagnitude;
    if (cpuMagnitude > gpuMagnitude) {
        maxMagnitude = cpuMagnitude;
    }
    else {
        maxMagnitude = gpuMagnitude;
    }

    float relativeThreshold = rTol * maxMagnitude;
    float acceptableThreshold;
    if (aTol > relativeThreshold) {
        acceptableThreshold = aTol;
    }
    else {
        acceptableThreshold = relativeThreshold;
    }

    return fabsf(cpuResult - gpuResult) <= acceptableThreshold;
}


__global__ void saxpyKernel(float* y, float* x, float a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void matrixAddKernel(float* c, float* a, float* b, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int index = row * N + col;
        c[index] = a[index] + b[index];
    }
}

__global__ void volumeAddKernel(float* c, float* a, float* b, int M, int N, int P)
{
    // Map CUDA x/y/z coordinates to the volume's P/N/M dimensions.
    int dimM = blockIdx.z * blockDim.z + threadIdx.z;
    int dimN = blockIdx.y * blockDim.y + threadIdx.y;
    int dimP = blockIdx.x * blockDim.x + threadIdx.x;
    if (dimM < M && dimN < N && dimP < P) {
        // Store the [M][N][P] volume as a flat array with P changing fastest.
        int index = (dimM * N + dimN) * P + dimP;
        c[index] = a[index] + b[index];
    }
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    int allTolerancesPassed = 1;

    // SAXPY kernel
    // Change saxpyN to 1000 or 10000000 to test the other vector sizes.
    int saxpyN = 10;
    float alpha = 0.7f;
    float* x = (float*)malloc((size_t)saxpyN * sizeof(float));
    float* y = (float*)malloc((size_t)saxpyN * sizeof(float));
    float* cpuY = (float*)malloc((size_t)saxpyN * sizeof(float));
    if (x == NULL || y == NULL || cpuY == NULL) {
        fprintf(stderr, "Malloc failed\n");
        free(x);
        free(y);
        free(cpuY);
        return 1;
    }

    // Use fractional inputs to expose possible CPU/GPU rounding differences.
    for (int i = 0; i < saxpyN; i++) {
        x[i] = 0.1f * (i + 1);
        y[i] = 0.3f * (i + 1);
        cpuY[i] = y[i];
    }

    cudaStatus = addWithSaxpy(y, x, alpha, saxpyN);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "SAXPY failed for N=%d: %s\n", saxpyN, cudaGetErrorString(cudaStatus));
        free(x);
        free(y);
        free(cpuY);
        return 1;
    }

    // Verify every element and record the maximum absolute CPU/GPU difference.
    float saxpyMaxAbsError = 0.0f;
    int saxpyMatches = 1;
    for (int i = 0; i < saxpyN; i++) {
        float cpuResult = alpha * x[i] + cpuY[i];
        float absError = fabsf(cpuResult - y[i]);
        if (absError > saxpyMaxAbsError) {
            saxpyMaxAbsError = absError;
        }
        if (!withinThreshold(cpuResult, y[i])) {
            saxpyMatches = 0;
        }
    }
    printf("SAXPY max absolute error: %.9g\n", saxpyMaxAbsError);
    if (!saxpyMatches) {
        fprintf(stderr, "SAXPY tolerance check failed\n");
        allTolerancesPassed = 0;
    }
    else {
        printf("SAXPY N=%d: SUCCESS\n", saxpyN);
    }

    free(x);
    free(y);
    free(cpuY);

    // 2D matrix addition
    // Change M and N to test 500x500 or 3024x4032.
    int M = 4;
    int N = 4;
    size_t matrixSize = (size_t)M * N;
    float* matrixA = (float*)malloc(matrixSize * sizeof(float));
    float* matrixB = (float*)malloc(matrixSize * sizeof(float));
    float* matrixC = (float*)malloc(matrixSize * sizeof(float));
    if (matrixA == NULL || matrixB == NULL || matrixC == NULL) {
        fprintf(stderr, "Malloc failed\n");
        free(matrixA);
        free(matrixB);
        free(matrixC);
        return 1;
    }

    // Fractional inputs exercise FP32 rounding in the matrix calculation.
    for (size_t i = 0; i < matrixSize; i++) {
        matrixA[i] = 0.1f * (float)(i + 1);
        matrixB[i] = 0.3f * (float)(i + 2);
    }

    cudaStatus = add2withcuda(matrixC, matrixA, matrixB, M, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Matrix addition failed: %s\n", cudaGetErrorString(cudaStatus));
        free(matrixA);
        free(matrixB);
        free(matrixC);
        return 1;
    }

    // Check every element and retain the largest CPU/GPU difference.
    float matrixMaxAbsError = 0.0f;
    int matrixMatches = 1;
    for (size_t i = 0; i < matrixSize; i++) {
        float cpuResult = matrixA[i] + matrixB[i];
        float absError = fabsf(cpuResult - matrixC[i]);
        if (absError > matrixMaxAbsError) {
            matrixMaxAbsError = absError;
        }
        if (!withinThreshold(cpuResult, matrixC[i])) {
            matrixMatches = 0;
        }
    }
    printf("2D matrix addition max absolute error: %.9g\n", matrixMaxAbsError);
    if (!matrixMatches) {
        fprintf(stderr, "2D matrix addition tolerance check failed\n");
        allTolerancesPassed = 0;
    }
    else {
        printf("2D matrix addition %dx%d: SUCCESS\n", M, N);
    }

    free(matrixA);
    free(matrixB);
    free(matrixC);

    // Set all dimensions to 100, then 1000, for the larger cases; use Release for 1000^3.
    int volumeM = 3;
    int volumeN = 3;
    int volumeP = 3;
    size_t volumeSize = (size_t)volumeM * volumeN * volumeP;
    float* volumeA = (float*)malloc(volumeSize * sizeof(float));
    float* volumeB = (float*)malloc(volumeSize * sizeof(float));
    float* volumeC = (float*)malloc(volumeSize * sizeof(float));
    if (volumeA == NULL || volumeB == NULL || volumeC == NULL) {
        fprintf(stderr, "Malloc failed\n");
        free(volumeA);
        free(volumeB);
        free(volumeC);
        return 1;
    }

    for (size_t i = 0; i < volumeSize; i++) {
        volumeA[i] = 0.1f * (float)(i + 1);
        volumeB[i] = 0.3f * (float)(i + 2);
    }

    cudaStatus = add3dwithcuda(volumeC, volumeA, volumeB, volumeM, volumeN, volumeP);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "3D volume addition failed: %s\n", cudaGetErrorString(cudaStatus));
        free(volumeA);
        free(volumeB);
        free(volumeC);
        return 1;
    }

    // Check every volume element and retain the largest CPU/GPU difference.
    float volumeMaxAbsError = 0.0f;
    int volumeMatches = 1;
    for (size_t i = 0; i < volumeSize; i++) {
        float cpuResult = volumeA[i] + volumeB[i];
        float absError = fabsf(cpuResult - volumeC[i]);
        if (absError > volumeMaxAbsError) {
            volumeMaxAbsError = absError;
        }
        if (!withinThreshold(cpuResult, volumeC[i])) {
            volumeMatches = 0;
        }
    }
    printf("3D volume addition max absolute error: %.9g\n", volumeMaxAbsError);
    if (!volumeMatches) {
        fprintf(stderr, "3D volume addition tolerance check failed\n");
        allTolerancesPassed = 0;
    }
    else {
        printf("3D volume addition %dx%dx%d: SUCCESS\n", volumeM, volumeN, volumeP);
    }
    printf("Max absolute error comparison: SAXPY=%.9g, 2D=%.9g, 3D=%.9g\n",
        saxpyMaxAbsError, matrixMaxAbsError, volumeMaxAbsError);

    free(volumeA);
    free(volumeB);
    free(volumeC);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    if (!allTolerancesPassed) {
        return 1;
    }
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
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

    // printf kernel, uncomment to test
    const unsigned int repetitions = 10;
    //for (unsigned int run = 0; run < repetitions; run++) {
    //    addKernel<<<1, 32>>>(dev_c, dev_a, dev_b);
    //}

    //for (unsigned int run = 0; run < repetitions; run++) {
    //    addKernel << <4, 8 >> > (dev_c, dev_a, dev_b);
    //}

    // for (unsigned int run = 0; run < repetitions; run++) {
    //     addKernel << <32, 1 >> > (dev_c, dev_a, dev_b);
    // }

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

cudaError_t addWithSaxpy(float* y, const float* x, float a, int N)
{
    float* dev_x = 0;
    float* dev_y = 0;
    cudaError_t cudaStatus;
    size_t bytes = (size_t)N * sizeof(float);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_x, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_y, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_x, x, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_y, y, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    saxpyKernel << <blocksPerGrid, threadsPerBlock >> > (dev_y, dev_x, a, N);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "saxpyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching saxpyKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(y, dev_y, bytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_y);
    cudaFree(dev_x);
    return cudaStatus;
}

cudaError_t add2withcuda(float* c, const float* a, const float* b, int M, int N)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;
    size_t bytes = (size_t)M * N * sizeof(float);
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    matrixAddKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, M, N);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixAddKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matrixAddKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
    return cudaStatus;
}

cudaError_t add3dwithcuda(float* c, const float* a, const float* b, int M, int N, int P)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;
    size_t bytes = (size_t)M * N * P * sizeof(float);
    // x covers P, y covers N, and z covers M; round up to cover edge cells.
    dim3 threadsPerBlock(8, 8, 4);
    dim3 blocksPerGrid(
        (P + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (M + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    volumeAddKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, M, N, P);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "volumeAddKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching volumeAddKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
    return cudaStatus;
}