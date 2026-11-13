/********************************
* Author: Julian Soh
* Date: Fall quarter 2026
* EE P 524 - GPU Programming
* Homework 3 - CUDA Kernels
* *******************************/

#include "device_launch_parameters.h"

#include <stdio.h>
#include "Error-Checking.h"
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cstdint>

// This file demonstrates branch divergence behavior in CUDA warps.

__global__ void single_branch_kernel(const float* in, float* out, int n) {
    // Compute the global thread index for the current thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Exit early if this thread maps outside the valid data range.
    if (idx >= n) return;

    // Compute lane id (thread position within a warp: 0..31).
    int lane = threadIdx.x & 31;  // 0..31 within each warp

    // Single branch divergence: half warp takes one branch, half takes the other.
    if (lane < 16) {
        out[idx] = in[idx] + 1.0f;   // GMEM write in branch A
    }
    else {
        out[idx] = in[idx] - 1.0f;   // GMEM write in branch B
    }
}

__global__ void nested_branch_kernel(const float* in, float* out, int n) {
    // Compute the global thread index for the current thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Exit early if this thread maps outside the valid data range.
    if (idx >= n) return;

    // Compute lane id (thread position within a warp: 0..31).
    int lane = threadIdx.x & 31;

    // Nested divergence: warp is split into multiple control-flow paths.
    if (lane < 8) {
        out[idx] = in[idx] + 10.0f;  // path 1
    }
    else {
        if (lane < 16) {
            out[idx] = in[idx] + 20.0f;  // path 2
        }
        else if (lane < 24) {
            out[idx] = in[idx] + 30.0f;  // path 3
        }
        else {
            out[idx] = in[idx] + 40.0f;  // path 4
        }
    }
}


int main()
{
    // Step 1: define problem size and total byte count.
    const int n = 256;
    const size_t bytes = n * sizeof(float);

    // Step 2: allocate host buffers for input and output.
    float* in_h = (float*)malloc(bytes);
    float* out_h = (float*)malloc(bytes);
    if (in_h == nullptr || out_h == nullptr) {
        fprintf(stderr, "Host malloc failed!\n");
        free(in_h);
        free(out_h);
        return 1;
    }

    // Step 3: initialize host input data with a simple ramp.
    for (int i = 0; i < n; i++) {
        in_h[i] = (float)i;
    }

    // Step 4: declare device pointers (GPU memory).
    float* in_d = nullptr;
    float* out_d = nullptr;

    // Step 5: select GPU device 0.
    if (checkCuda(cudaSetDevice(0), "cudaSetDevice(0)") != cudaSuccess) {
        free(in_h);
        free(out_h);
        return 1;
    }

    // Step 6: allocate device input buffer.
    if (checkCuda(cudaMalloc((void**)&in_d, bytes), "cudaMalloc((void**)&in_d, bytes)") != cudaSuccess) {
        free(in_h);
        free(out_h);
        return 1;
    }

    // Step 7: allocate device output buffer.
    if (checkCuda(cudaMalloc((void**)&out_d, bytes), "cudaMalloc((void**)&out_d, bytes)") != cudaSuccess) {
        cudaFree(in_d);
        free(in_h);
        free(out_h);
        return 1;
    }

    // Step 8: copy input data from host to device.
    if (checkCuda(cudaMemcpy(in_d, in_h, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(in_d, in_h, bytes, cudaMemcpyHostToDevice)") != cudaSuccess) {
        cudaFree(in_d);
        cudaFree(out_d);
        free(in_h);
        free(out_h);
        return 1;
    }

    // Step 9: configure launch dimensions.
    const int blockSize = 128;
    const int gridSize = (n + blockSize - 1) / blockSize;

    // Step 10: launch kernel 1 (single_branch_kernel).
    single_branch_kernel<<<gridSize, blockSize>>>(in_d, out_d, n);

    // Step 11: check asynchronous launch errors.
    if (checkCuda(cudaGetLastError(), "cudaGetLastError()") != cudaSuccess) {
        cudaFree(in_d);
        cudaFree(out_d);
        free(in_h);
        free(out_h);
        return 1;
    }

    // Step 12: wait for kernel completion and check runtime errors.
    if (checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize()") != cudaSuccess) {
        cudaFree(in_d);
        cudaFree(out_d);
        free(in_h);
        free(out_h);
        return 1;
    }

    // Step 13: copy results back from device to host.
    if (checkCuda(cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost)") != cudaSuccess) {
        cudaFree(in_d);
        cudaFree(out_d);
        free(in_h);
        free(out_h);
        return 1;
    }

    // Step 14: print the first few output values for quick verification.
    for (int i = 0; i < 8; i++) {
        printf("out_h[%d] = %.1f\n", i, out_h[i]);
    }

    // Step 15: release device and host resources.
    cudaFree(in_d);
    cudaFree(out_d);
    free(in_h);
    free(out_h);

    // Step 16: reset the device before exiting.
    cudaDeviceReset();

    return 0;
}

