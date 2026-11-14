/********************************
* Author: Julian Soh
* Date: Fall quarter 2026
* EE P 524 - GPU Programming
* Homework 4 - GMEM and SMEM
* *******************************/
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include "Error-Checking.h"
#include <cstdlib>
#include <cmath>
#include <new>
#include <string.h>
#include <fstream>
#include <vector>
#include <iomanip>
#include <limits>
#include <locale>

// ==============================
// Function Declarations
// ==============================

bool load_mmult_inputs(int width, std::vector<float>& M_h, std::vector<float>& N_h);
cudaError_t run_mmult(const float* M_h, const float* N_h, float* P_h, int width, int tileWidth, int blockWidth);


// ==============================
// Kernel Declarations
// ==============================

// Naive square matrix multiplication: one thread computes one element of P = M * N.
// M, N, and P are device pointers to row-major matrices, indexed as row * width + col.
// Each multiply reads its operands from global memory (possibly served by caches).
__global__ void mmult_naive_Kernel(const float* M, const float* N, float* P, int width)
{
    // Map the two-dimensional block and thread coordinates to an output element.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < width && col < width) {
		float sum = 0.0f;
		// Take the dot product of row 'row' of M and column 'col' of N.
		for (int k = 0; k < width; ++k)
		{
			sum += M[row * width + k] * N[k * width + col];
		}
		P[row * width + col] = sum;
	}
}

// Each template instantiation uses a matching TILE_WIDTH-by-TILE_WIDTH block.
template<int TILE_WIDTH>
__global__ void mmult_tiled_Kernel(const float* M, const float* N, float* P, int width)
{
	__shared__ float M_tile[TILE_WIDTH][TILE_WIDTH];
	__shared__ float N_tile[TILE_WIDTH][TILE_WIDTH];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	// Each phase covers TILE_WIDTH elements of the shared inner dimension.
	for (int tile = 0; tile < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
		// Cooperatively load one element per thread; pad partial edge tiles with zero.
		if (row < width && tile * TILE_WIDTH + threadIdx.x < width) {
			M_tile[threadIdx.y][threadIdx.x] = M[row * width + tile * TILE_WIDTH + threadIdx.x];
		}
		else {
			M_tile[threadIdx.y][threadIdx.x] = 0.0f;
		}
		if (col < width && tile * TILE_WIDTH + threadIdx.y < width) {
			N_tile[threadIdx.y][threadIdx.x] = N[(tile * TILE_WIDTH + threadIdx.y) * width + col];
		}
		else {
			N_tile[threadIdx.y][threadIdx.x] = 0.0f;
		}
		// All tile elements must be loaded before any thread consumes them.
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k) {
			sum += M_tile[threadIdx.y][k] * N_tile[k][threadIdx.x];
		}
		// Finish reading the tiles before any thread overwrites them in the next phase.
		__syncthreads();
	}
	if (row < width && col < width) {
		P[row * width + col] = sum;
	}
}


int main(int argc, char** argv)
{
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "-h") == 0) {
			printf("Usage: %s [-input4|-input8|-input256] [-naive|-tile8|-tile16|-tile32] [-block8|-block16|-block32]\n", argv[0]);
			printf("\nOptions (choose at most one from each group; order does not matter):\n");
			printf("  -input4      Load 4x4 M and N matrices.\n");
			printf("  -input8      Load 8x8 M and N matrices.\n");
			printf("  -input256    Load 256x256 M and N matrices (default).\n");
			printf("  -naive       Run global-memory matrix multiplication (default).\n");
			printf("  -tile8       Run shared-memory multiplication with 8x8 tiles.\n");
			printf("  -tile16      Run shared-memory multiplication with 16x16 tiles.\n");
			printf("  -tile32      Run shared-memory multiplication with 32x32 tiles.\n");
			printf("  -block8      Use an 8x8 thread block.\n");
			printf("  -block16     Use a 16x16 thread block (default for naive).\n");
			printf("  -block32     Use a 32x32 thread block.\n");
			printf("  -h           Show this help and exit without running a kernel.\n");
			printf("\nTiled kernels default to a block matching the tile size; if a block is\n");
			printf("specified explicitly, it must match the selected tile size.\n");
			printf("Input files are read from inputs\\ relative to the working directory.\n");
			printf("For -input4 and -input8, the program prints M, N, and P for verification.\n");
			printf("\nExamples:\n");
			printf("  %s -input256 -naive -block8\n", argv[0]);
			printf("  %s -input8 -tile16 -block16\n", argv[0]);
			return 0;
		}
	}

	int width = 256;
	int tileWidth = 0; // Zero selects the naive global-memory kernel.
	int blockWidth = 16;
	bool sizeSelected = false;
	bool algorithmSelected = false;
	bool blockSelected = false;
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "-input4") == 0 || strcmp(argv[i], "-input8") == 0 ||
			strcmp(argv[i], "-input256") == 0) {
			if (sizeSelected) {
				fprintf(stderr, "[ERROR] Specify only one matrix size.\n");
				return 1;
			}
			sizeSelected = true;
			width = strcmp(argv[i], "-input4") == 0 ? 4 :
				(strcmp(argv[i], "-input8") == 0 ? 8 : 256);
		}
		else if (strcmp(argv[i], "-naive") == 0 || strcmp(argv[i], "-tile8") == 0 ||
			strcmp(argv[i], "-tile16") == 0 || strcmp(argv[i], "-tile32") == 0) {
			if (algorithmSelected) {
				fprintf(stderr, "[ERROR] Specify only one algorithm.\n");
				return 1;
			}
			algorithmSelected = true;
			if (strcmp(argv[i], "-tile8") == 0) tileWidth = 8;
			else if (strcmp(argv[i], "-tile16") == 0) tileWidth = 16;
			else if (strcmp(argv[i], "-tile32") == 0) tileWidth = 32;
		}
		else if (strcmp(argv[i], "-block8") == 0 || strcmp(argv[i], "-block16") == 0 ||
			strcmp(argv[i], "-block32") == 0) {
			if (blockSelected) {
				fprintf(stderr, "[ERROR] Specify only one block size.\n");
				return 1;
			}
			blockSelected = true;
			blockWidth = strcmp(argv[i], "-block8") == 0 ? 8 :
				(strcmp(argv[i], "-block16") == 0 ? 16 : 32);
		}
		else {
			fprintf(stderr, "[ERROR] Unknown option: %s\n", argv[i]);
			fprintf(stderr, "Usage: %s [-input4|-input8|-input256] [-naive|-tile8|-tile16|-tile32] [-block8|-block16|-block32]\n", argv[0]);
			return 1;
		}
	}

	// Tiled kernels require one thread per tile element. Preserve existing tile
	// defaults, and reject an explicitly requested block that does not match.
	if (tileWidth != 0) {
		if (blockSelected && blockWidth != tileWidth) {
			fprintf(stderr, "[ERROR] Tiled block size must match tile size.\n");
			return 1;
		}
		blockWidth = tileWidth;
	}

	// The vectors own the input and result data in host RAM.
	std::vector<float> M_h;
	std::vector<float> N_h;
	if (!load_mmult_inputs(width, M_h, N_h)) {
		return 1;
	}
	std::vector<float> P_h(static_cast<size_t>(width) * width);
	printf("Matrix size: %dx%d; ", width, width);
	if (tileWidth == 0) printf("algorithm: naive (%dx%d block)\n", blockWidth, blockWidth);
	else printf("algorithm: tiled (%dx%d block)\n", blockWidth, blockWidth);
	const cudaError_t cudaStatus = run_mmult(M_h.data(), N_h.data(), P_h.data(), width, tileWidth, blockWidth);
	if (cudaStatus != cudaSuccess) {
		return 1;
	}
	printf("P[0,0] = %.9g\n", P_h[0]);

	// For verification only: 4x4 and 8x8 matrices are small enough to print;
	// printing the complete 256x256 matrices is not practical.
	if (width == 4 || width == 8) {
		const char* names[] = { "M", "N", "P" };
		const std::vector<float>* matrices[] = { &M_h, &N_h, &P_h };
		for (int matrix = 0; matrix < 3; ++matrix) {
			printf("%s:\n", names[matrix]);
			for (int row = 0; row < width; ++row) {
				for (int col = 0; col < width; ++col) {
					printf("%s%.9g", col == 0 ? "" : " ", (*matrices[matrix])[static_cast<size_t>(row) * width + col]);
				}
				printf("\n");
			}
		}
	}

	// Reset after all allocations are freed so profiling tools see complete traces.
	if (checkCuda(cudaDeviceReset(), "cudaDeviceReset") != cudaSuccess) {
		return 1;
	}
	return 0;
}

// ==============================
// non-CUDA helper functions
// ==============================

// Load headerless, row-major little-endian float32 matrices from the working directory.
// Return false and report the input file if opening, sizing, or reading fails.
bool load_mmult_inputs(int width, std::vector<float>& M_h, std::vector<float>& N_h)
{
    if (width != 4 && width != 8 && width != 256) {
        fprintf(stderr, "[ERROR] Unsupported matrix width: %d\n", width);
        return false;
    }
    const size_t elementCount = static_cast<size_t>(width) * width;
    static_assert(sizeof(float) == 4, "Input matrices require 32-bit floats");
    const std::streamsize expectedBytes = static_cast<std::streamsize>(elementCount * sizeof(float));
    char mFileName[64];
    char nFileName[64];
    snprintf(mFileName, sizeof(mFileName), "inputs\\M_%dx%d_f32le.bin", width, width);
    snprintf(nFileName, sizeof(nFileName), "inputs\\N_%dx%d_f32le.bin", width, width);
    const char* fileNames[] = { mFileName, nFileName };
    std::vector<float>* matrices[] = { &M_h, &N_h };

    for (int i = 0; i < 2; ++i) {
        // Open at the end so tellg can verify the exact file size before reading.
        std::ifstream file(fileNames[i], std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            fprintf(stderr, "[ERROR] Failed to open matrix input: %s\n", fileNames[i]);
            return false;
        }
        if (file.tellg() != expectedBytes) {
            fprintf(stderr, "[ERROR] Matrix input has an unexpected size: %s\n", fileNames[i]);
            return false;
        }

        matrices[i]->resize(elementCount);
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char*>(matrices[i]->data()), expectedBytes)) {
            fprintf(stderr, "[ERROR] Failed to read matrix input: %s\n", fileNames[i]);
            return false;
        }
    }
    return true;
}

// ==============================
// CUDA helper functions
// ==============================

// Run the selected multiplication on row-major square matrices and return its result.
cudaError_t run_mmult(const float* M_h, const float* N_h, float* P_h, int width, int tileWidth, int blockWidth)
{
    if (M_h == nullptr || N_h == nullptr || P_h == nullptr || width <= 0 ||
        (tileWidth != 0 && tileWidth != 8 && tileWidth != 16 && tileWidth != 32) ||
        (blockWidth != 8 && blockWidth != 16 && blockWidth != 32) ||
        (tileWidth != 0 && tileWidth != blockWidth) ||
        static_cast<size_t>(width) > (std::numeric_limits<size_t>::max)() / sizeof(float) / static_cast<size_t>(width)) {
        return checkCuda(cudaErrorInvalidValue, "mmult inputs, dimensions, tile or block width");
    }

    float* M_d = nullptr;
    float* N_d = nullptr;
    float* P_d = nullptr;
    const size_t mBytes = static_cast<size_t>(width) * width * sizeof(float);
    const size_t nBytes = static_cast<size_t>(width) * width * sizeof(float);
    const size_t pBytes = static_cast<size_t>(width) * width * sizeof(float);
    cudaError_t cudaStatus = cudaSuccess;
    cudaError_t cleanupStatus = cudaSuccess;
    const dim3 block(blockWidth, blockWidth);
    const dim3 grid((width + blockWidth - 1) / blockWidth, (width + blockWidth - 1) / blockWidth);

    // Part 1: Allocate space for both inputs and the output in GPU global memory.
    cudaStatus = checkCuda(cudaMalloc((void**)&M_d, mBytes), "cudaMalloc M_d");
    if (cudaStatus != cudaSuccess) goto Cleanup;
    cudaStatus = checkCuda(cudaMalloc((void**)&N_d, nBytes), "cudaMalloc N_d");
    if (cudaStatus != cudaSuccess) goto Cleanup;
    cudaStatus = checkCuda(cudaMalloc((void**)&P_d, pBytes), "cudaMalloc P_d");
    if (cudaStatus != cudaSuccess) goto Cleanup;

    // Copy the two row-major host inputs to their device buffers. P_d is output
    // storage, so its contents do not need to be copied from the host.
	cudaStatus = checkCuda(cudaMemcpy(M_d, M_h, mBytes, cudaMemcpyHostToDevice), "copy M to device");
	if (cudaStatus != cudaSuccess) goto Cleanup;
	cudaStatus = checkCuda(cudaMemcpy(N_d, N_h, nBytes, cudaMemcpyHostToDevice), "copy N to device");
	if (cudaStatus != cudaSuccess) goto Cleanup;

	// Part 2: Launch the requested kernel with one thread per output element.
	switch (tileWidth) {
	case 0: mmult_naive_Kernel<<<grid, block>>>(M_d, N_d, P_d, width); break;
	case 8: mmult_tiled_Kernel<8><<<grid, block>>>(M_d, N_d, P_d, width); break;
	case 16: mmult_tiled_Kernel<16><<<grid, block>>>(M_d, N_d, P_d, width); break;
	case 32: mmult_tiled_Kernel<32><<<grid, block>>>(M_d, N_d, P_d, width); break;
	}
	cudaStatus = checkCuda(cudaGetLastError(), "launch matrix multiplication kernel");
	if (cudaStatus != cudaSuccess) goto Cleanup;
	cudaStatus = checkCuda(cudaDeviceSynchronize(), "execute matrix multiplication kernel");
	if (cudaStatus != cudaSuccess) goto Cleanup;

	// Part 3: Copy the completed result back into host memory.
	cudaStatus = checkCuda(cudaMemcpy(P_h, P_d, pBytes, cudaMemcpyDeviceToHost), "copy P to host");
	if (cudaStatus != cudaSuccess) goto Cleanup;

	// Part 4: Release allocations even if an earlier operation failed. Preserve
    // the first error while still attempting to free every allocated buffer.
Cleanup:
    if (P_d != nullptr) {
        cleanupStatus = checkCuda(cudaFree(P_d), "cudaFree P_d");
        if (cudaStatus == cudaSuccess) cudaStatus = cleanupStatus;
    }
    if (N_d != nullptr) {
        cleanupStatus = checkCuda(cudaFree(N_d), "cudaFree N_d");
        if (cudaStatus == cudaSuccess) cudaStatus = cleanupStatus;
    }
    if (M_d != nullptr) {
        cleanupStatus = checkCuda(cudaFree(M_d), "cudaFree M_d");
        if (cudaStatus == cudaSuccess) cudaStatus = cleanupStatus;
    }
    return cudaStatus;
}
