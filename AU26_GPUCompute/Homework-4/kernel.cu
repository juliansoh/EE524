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
#include <Windows.h>
#include <profileapi.h>

// ==============================
// Function Declarations
// ==============================

bool load_mmult_inputs(int width, std::vector<float>& M_h, std::vector<float>& N_h);
void mmult_cpu(const float* M_h, const float* N_h, float* P_h, int width);
bool verify_mmult(const float* P_h, const float* P_gpu_h, int width);
cudaError_t run_mmult(const float* M_h, const float* N_h, float* P_h, int width, int tileWidth, int blockWidth, bool transposeM);


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

// Transpose M on the device into a separate buffer, preserving the original input.
// Row-major M's transpose is the column-major storage of the original mathematical M:
// M_transposed_d[col * width + row] = M_d[row * width + col].
// The padded shared tile avoids bank conflicts when its indices are exchanged.
__global__ void transpose_Kernel(const float* M_d, float* M_transposed_d, int width)
{
	__shared__ float tile[16][17];
	int x = blockIdx.x * 16 + threadIdx.x;
	int y = blockIdx.y * 16 + threadIdx.y;
	if (x < width && y < width) {
		tile[threadIdx.y][threadIdx.x] = M_d[y * width + x];
	}
	__syncthreads();
	x = blockIdx.y * 16 + threadIdx.x;
	y = blockIdx.x * 16 + threadIdx.y;
	if (x < width && y < width) {
		M_transposed_d[y * width + x] = tile[threadIdx.x][threadIdx.y];
	}
}

// Changed the original code: M is now transposed on the device before this launch.
// No host transpose or intermediate device-to-host transfer is needed.
// M_transposed_d stores the original M column-major; N_d and P_d stay row-major.
template<int TILE_WIDTH>
__global__ void mmult_v1b_Kernel(const float* M_transposed_d, const float* N_d, float* P_d, int width)
{
	__shared__ float M_tile[TILE_WIDTH][TILE_WIDTH + 1];
	__shared__ float N_tile[TILE_WIDTH][TILE_WIDTH];
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int row = blockIdx.y * TILE_WIDTH + ty;
	const int col = blockIdx.x * TILE_WIDTH + tx;
	float sum = 0.0f;

	for (int phase = 0; phase < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
		// Adjacent x threads load adjacent rows of column-major M, while ty
		// selects its inner-dimension column. Store as [inner][output row].
		const int mRow = blockIdx.y * TILE_WIDTH + tx;
		const int inner = phase * TILE_WIDTH + ty;
		M_tile[ty][tx] = mRow < width && inner < width
			? M_transposed_d[inner * width + mRow] : 0.0f;
		// Adjacent x threads also load adjacent columns of row-major N.
		N_tile[ty][tx] = inner < width && col < width
			? N_d[inner * width + col] : 0.0f;
		__syncthreads();

		// Corner turn: read M with [k][ty], not the ordinary [ty][k].
		// This still computes original M[row,k] * N[k,col], not M^T * N.
		for (int k = 0; k < TILE_WIDTH; ++k) {
			sum += M_tile[k][ty] * N_tile[k][tx];
		}
		// No thread may overwrite the shared tiles until all readers finish.
		__syncthreads();
	}
	if (row < width && col < width) {
		P_d[row * width + col] = sum;
	}
}

// Original row-major, all-global v1b placeholder retained for comparison.
// Replaced above by phased shared-memory loads using device-transposed M.
/*
__global__ void mmult_v1b_Kernel(const float* M, const float* N, float* P, int width)
{
	// This thread is responsible for one output element P[row, col].
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Ignore threads outside the matrix when width is not divisible by the block dimensions.
	if (row < width && col < width) {
		float sum = 0.0f;

		// Compute the dot product of M's row and N's column.
		// These indices assume both input matrices are row-major in global memory:
		// M[row, k] is M[row * width + k], and N[k, col] is N[k * width + col].
		for (int k = 0; k < width; ++k)
		{
			sum += M[row * width + k] * N[k * width + col];
		}
		// Store this thread's completed output element in row-major P.
		P[row * width + col] = sum;
	}
}
*/


int main(int argc, char** argv)
{
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "-h") == 0) {
			printf("Usage: %s [-input4|-input8|-input256] [-naive|-tile8|-tile16|-tile32] [-block8|-block16|-block32] [-T]\n", argv[0]);
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
			printf("  -T           Transpose M on the GPU, then run corner-turning MMULTv1b.\n");
			printf("               Requires -tile8, -tile16, or -tile32; N is not transposed.\n");
			printf("  -h           Show this help and exit without running a kernel.\n");
			printf("\nTiled kernels default to a block matching the tile size; if a block is\n");
			printf("specified explicitly, it must match the selected tile size.\n");
			printf("Input files are read from inputs\\ relative to the working directory.\n");
			printf("All algorithms print Windows high-resolution full round-trip time in ms.\n");
			printf("Each GPU result is compared with a sequential CPU reference.\n");
			printf("For -input4 and -input8, the program prints M, N, and P for verification.\n");
			printf("\nExamples:\n");
			printf("  %s -input256 -naive -block8\n", argv[0]);
			printf("  %s -input8 -tile16 -block16\n", argv[0]);
			printf("  %s -input256 -tile16 -T\n", argv[0]);
			return 0;
		}
	}

	int width = 256;
	int tileWidth = 0; // Zero selects the naive global-memory kernel.
	int blockWidth = 16;
	bool sizeSelected = false;
	bool algorithmSelected = false;
	bool blockSelected = false;
	bool transposeM = false;
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
		else if (strcmp(argv[i], "-T") == 0) {
			if (transposeM) {
				fprintf(stderr, "[ERROR] Specify -T only once.\n");
				return 1;
			}
			transposeM = true;
		}
		else {
			fprintf(stderr, "[ERROR] Unknown option: %s\n", argv[i]);
			fprintf(stderr, "Usage: %s [-input4|-input8|-input256] [-naive|-tile8|-tile16|-tile32] [-block8|-block16|-block32] [-T]\n", argv[0]);
			return 1;
		}
	}

	if (transposeM && tileWidth == 0) {
		fprintf(stderr, "[ERROR] -T requires -tile8, -tile16, or -tile32.\n");
		return 1;
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
	// Both results reside in host RAM: P_h holds the sequential CPU reference,
	// and P_gpu_h receives the selected CUDA kernel's result.
	std::vector<float> P_h(static_cast<size_t>(width) * width);
	std::vector<float> P_gpu_h(static_cast<size_t>(width) * width);
	// Run the CPU-only reference first on the same row-major inputs as the GPU.
	// This calculation is outside the GPU round-trip timing interval.
	mmult_cpu(M_h.data(), N_h.data(), P_h.data(), width);
	printf("Matrix size: %dx%d; ", width, width);
	if (tileWidth == 0) printf("algorithm: naive (%dx%d block)\n", blockWidth, blockWidth);
	else printf("algorithm: %s (%dx%d block)\n", transposeM ? "corner-turning (GPU transpose)" : "tiled", blockWidth, blockWidth);
	LARGE_INTEGER perfFrequency = {};
	LARGE_INTEGER roundTripStart = {};
	LARGE_INTEGER roundTripStop = {};
	// Use the same full round-trip interval for all algorithms: allocations,
	// input copies, optional device transpose, multiplication, output and cleanup.
	// File loading, printing and cudaDeviceReset are outside this interval.
	// The first run may include CUDA context initialization inside run_mmult.
	if (!QueryPerformanceFrequency(&perfFrequency) || perfFrequency.QuadPart <= 0) {
		fprintf(stderr, "[ERROR] QueryPerformanceFrequency failed.\n");
		return 1;
	}
	if (!QueryPerformanceCounter(&roundTripStart)) {
		fprintf(stderr, "[ERROR] QueryPerformanceCounter failed at timer start.\n");
		return 1;
	}
	const cudaError_t cudaStatus = run_mmult(M_h.data(), N_h.data(), P_gpu_h.data(), width, tileWidth, blockWidth, transposeM);
	if (!QueryPerformanceCounter(&roundTripStop)) {
		fprintf(stderr, "[ERROR] QueryPerformanceCounter failed at timer stop.\n");
		return 1;
	}
	if (cudaStatus != cudaSuccess) {
		return 1;
	}
	const double roundTripMs = static_cast<double>(roundTripStop.QuadPart - roundTripStart.QuadPart)
		* 1000.0 / static_cast<double>(perfFrequency.QuadPart);
	const char* algorithm = transposeM ? "MMULTv2b GPU transpose + corner turning" :
		(tileWidth == 0 ? "MMULTv0 naive" : "MMULTv1 tiled");
	printf("%s full round-trip time: %.6f ms\n", algorithm, roundTripMs);
	// Compare every copied-back GPU output against the independent CPU result.
	// Report a failure and return a nonzero exit status if any element differs.
	const bool verified = verify_mmult(P_h.data(), P_gpu_h.data(), width);
	printf("P[0,0] = %.9g\n", P_gpu_h[0]);

	// For verification only: 4x4 and 8x8 matrices are small enough to print;
	// printing the complete 256x256 matrices is not practical.
	if (width == 4 || width == 8) {
		const char* names[] = { "M", "N", "P" };
		const std::vector<float>* matrices[] = { &M_h, &N_h, &P_gpu_h };
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
	return verified ? 0 : 1;
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

// CPU-only reference for square row-major matrices, independent of CUDA kernels.
// For each output P[row,col], sum M[row,k] * N[k,col] over the inner dimension.
// The caller supplies distinct host buffers sized for width * width floats.
void mmult_cpu(const float* M_h, const float* N_h, float* P_h, int width)
{
	for (int row = 0; row < width; ++row) {
		for (int col = 0; col < width; ++col) {
			float sum = 0.0f;
			for (int k = 0; k < width; ++k) {
				sum += M_h[static_cast<size_t>(row) * width + k] *
					N_h[static_cast<size_t>(k) * width + col];
			}
			P_h[static_cast<size_t>(row) * width + col] = sum;
		}
	}
}

// Compare the CPU reference with the GPU result copied back to host memory.
// Allow FP32 rounding differences using abs(cpu-gpu) <= max(aTol, rTol *
// max(abs(cpu), abs(gpu))). Non-finite results are always treated as failures.
bool verify_mmult(const float* P_h, const float* P_gpu_h, int width)
{
	constexpr double aTol = 1e-6;
	constexpr double rTol = 1e-5;
	size_t mismatches = 0;
	const size_t count = static_cast<size_t>(width) * width;
	for (size_t i = 0; i < count; ++i) {
		const double cpu = P_h[i];
		const double gpu = P_gpu_h[i];
		const double absCpu = std::fabs(cpu);
		const double absGpu = std::fabs(gpu);
		const double scaledTol = rTol * (absCpu > absGpu ? absCpu : absGpu);
		const double tolerance = aTol > scaledTol ? aTol : scaledTol;
		if (!std::isfinite(cpu) || !std::isfinite(gpu) || std::fabs(cpu - gpu) > tolerance) {
			// Print the first mismatch to identify the problem without flooding output.
			if (mismatches == 0) {
				fprintf(stderr, "[ERROR] First verification mismatch at P[%zu,%zu]: CPU=%.9g GPU=%.9g\n",
					i / static_cast<size_t>(width), i % static_cast<size_t>(width), cpu, gpu);
			}
			++mismatches;
		}
	}
	if (mismatches != 0) {
		fprintf(stderr, "Verification FAILED: %zu of %zu elements differ or are non-finite.\n", mismatches, count);
		return false;
	}
	printf("Verification PASSED: all %zu GPU elements match the CPU reference (aTol=1e-6, rTol=1e-5).\n", count);
	return true;
}

// ==============================
// CUDA helper functions
// ==============================

// Run the selected multiplication on row-major square matrices and return its result.
cudaError_t run_mmult(const float* M_h, const float* N_h, float* P_h, int width, int tileWidth, int blockWidth, bool transposeM)
{
    if (M_h == nullptr || N_h == nullptr || P_h == nullptr || width <= 0 ||
        (tileWidth != 0 && tileWidth != 8 && tileWidth != 16 && tileWidth != 32) ||
        (blockWidth != 8 && blockWidth != 16 && blockWidth != 32) ||
        (tileWidth != 0 && tileWidth != blockWidth) ||
        (transposeM && tileWidth == 0) ||
        static_cast<size_t>(width) > (std::numeric_limits<size_t>::max)() / sizeof(float) / static_cast<size_t>(width)) {
        return checkCuda(cudaErrorInvalidValue, "mmult inputs, dimensions, tile or block width");
    }

    float* M_d = nullptr;
    float* N_d = nullptr;
    float* P_d = nullptr;
    float* M_transposed_d = nullptr;
    const size_t mBytes = static_cast<size_t>(width) * width * sizeof(float);
    const size_t nBytes = static_cast<size_t>(width) * width * sizeof(float);
    const size_t pBytes = static_cast<size_t>(width) * width * sizeof(float);
    cudaError_t cudaStatus = cudaSuccess;
    cudaError_t cleanupStatus = cudaSuccess;
    const dim3 block(blockWidth, blockWidth);
    const dim3 grid((width + blockWidth - 1) / blockWidth, (width + blockWidth - 1) / blockWidth);
    const dim3 transposeBlock(16, 16);
    const dim3 transposeGrid((width + 15) / 16, (width + 15) / 16);

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

	// With -T, transpose once into persistent device storage. Both launches
	// use the default stream, so multiplication waits for the transpose.
	if (transposeM) {
		cudaStatus = checkCuda(cudaMalloc((void**)&M_transposed_d, mBytes), "cudaMalloc M_transposed_d");
		if (cudaStatus != cudaSuccess) goto Cleanup;
		transpose_Kernel<<<transposeGrid, transposeBlock>>>(M_d, M_transposed_d, width);
		cudaStatus = checkCuda(cudaGetLastError(), "launch transpose kernel");
		if (cudaStatus != cudaSuccess) goto Cleanup;
	}

	// Part 2: Launch the requested kernel with one thread per output element.
	if (transposeM) {
		switch (tileWidth) {
		case 8: mmult_v1b_Kernel<8><<<grid, block>>>(M_transposed_d, N_d, P_d, width); break;
		case 16: mmult_v1b_Kernel<16><<<grid, block>>>(M_transposed_d, N_d, P_d, width); break;
		case 32: mmult_v1b_Kernel<32><<<grid, block>>>(M_transposed_d, N_d, P_d, width); break;
		}
	}
	else {
	switch (tileWidth) {
	case 0: mmult_naive_Kernel<<<grid, block>>>(M_d, N_d, P_d, width); break;
	case 8: mmult_tiled_Kernel<8><<<grid, block>>>(M_d, N_d, P_d, width); break;
	case 16: mmult_tiled_Kernel<16><<<grid, block>>>(M_d, N_d, P_d, width); break;
	case 32: mmult_tiled_Kernel<32><<<grid, block>>>(M_d, N_d, P_d, width); break;
	}
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
    if (M_transposed_d != nullptr) {
        cleanupStatus = checkCuda(cudaFree(M_transposed_d), "cudaFree M_transposed_d");
        if (cudaStatus == cudaSuccess) cudaStatus = cleanupStatus;
    }
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
