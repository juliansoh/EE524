/********************************
* Author: Julian Soh
* Date: Fall quarter 2026
* EE P 524 - GPU Programming
* Homework 3 - CUDA Kernels
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

int load_mmult_inputs(const char* sizeOption, int j, int k, int l,
    std::vector<float>& M_h, std::vector<float>& N_h);
void serial_mmult(const float* M_h, const float* N_h, float* P_h, int j, int k, int l);
int write_mmult_csv(const char* outputFileName, const float* P_h, int j, int l);
cudaError_t naive_mmult(const float* M, const float* N, float* P, int j, int k, int l);
cudaError_t tiled_mmult(const float* M, const float* N, float* P, int j, int k, int l);

// ==============================
// Kernel Declarations
// ==============================

__global__ void naive_mmult_kernel(const float* M_d, const float* N_d, float* P_d,
    int j, int k, int l)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= j || col >= l) {
        return;
    }

    float sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        sum += M_d[static_cast<size_t>(row) * k + inner] *
            N_d[static_cast<size_t>(inner) * l + col];
    }
    P_d[static_cast<size_t>(row) * l + col] = sum;
}

// Launch with a square TILE_WIDTH-by-TILE_WIDTH thread block.
template<int TILE_WIDTH = 16>
__global__ void tiled_mmult_kernel(const float* M_d, const float* N_d, float* P_d,
    int j, int k, int l)
{
    static_assert(TILE_WIDTH > 0 && TILE_WIDTH <= 32, "Tile width must be between 1 and 32");
    __shared__ float M_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_tile[TILE_WIDTH][TILE_WIDTH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_WIDTH + ty;
    const int col = blockIdx.x * TILE_WIDTH + tx;
    float sum = 0.0f;

    const int phaseCount = k / TILE_WIDTH + (k % TILE_WIDTH != 0);
    for (int phase = 0; phase < phaseCount; ++phase) {
        const int mCol = phase * TILE_WIDTH + tx;
        const int nRow = phase * TILE_WIDTH + ty;
        M_tile[ty][tx] = row < j && mCol < k
            ? M_d[static_cast<size_t>(row) * k + mCol] : 0.0f;
        N_tile[ty][tx] = nRow < k && col < l
            ? N_d[static_cast<size_t>(nRow) * l + col] : 0.0f;
        __syncthreads();

        for (int inner = 0; inner < TILE_WIDTH; ++inner) {
            sum += M_tile[ty][inner] * N_tile[inner][tx];
        }
        // All threads must finish reading before the next phase overwrites the tiles.
        __syncthreads();
    }

    if (row < j && col < l) {
        P_d[static_cast<size_t>(row) * l + col] = sum;
    }
}

// ==============================
// Host Entry Point
// ==============================
int main(int argc, char** argv)
{
    const bool validAlgorithm = argc >= 2 &&
        (strcmp(argv[1], "-naive") == 0 || strcmp(argv[1], "-tiled") == 0);
    const bool validSize = argc >= 3 &&
        (strcmp(argv[2], "-k100") == 0 || strcmp(argv[2], "-k1000") == 0 || strcmp(argv[2], "-k2500") == 0);
    if (argc != 3 || !validAlgorithm || !validSize) {
        printf("Usage: %s <algorithm> <matrix-size>\n", argc > 0 ? argv[0] : "Homework-3-MMULT.exe");
        printf("  The CPU serial reference always runs and writes P_out_serial.csv.\n");
        printf("  <algorithm>    Required GPU variant selection:\n");
        printf("    -naive       GPU naive global-memory implementation; writes P_out_naive.csv\n");
        printf("    -tiled       GPU shared-memory tiled implementation; writes P_out_tiled.csv\n");
        printf("  <matrix-size>  Required inner dimension k and output shape:\n");
        printf("    -k100        M 100x100, N 100x100, P 100x100\n");
        printf("    -k1000       M 1000x1000, N 1000x1000, P 1000x1000\n");
        printf("    -k2500       M 2000x2500, N 2500x3000, P 2000x3000\n");
        printf("Examples:\n");
        printf("  %s -naive -k100\n", argc > 0 ? argv[0] : "Homework-3-MMULT.exe");
        printf("  %s -naive -k1000\n", argc > 0 ? argv[0] : "Homework-3-MMULT.exe");
        printf("  %s -tiled -k2500\n", argc > 0 ? argv[0] : "Homework-3-MMULT.exe");
        return 1;
    }

    const char* sizeOption = argv[2];
    int j = 0;
    int k = 0;
    int l = 0;
    if (strcmp(sizeOption, "-k100") == 0) {
        j = 100;
        k = 100;
        l = 100;
    }
    else if (strcmp(sizeOption, "-k1000") == 0) {
        j = 1000;
        k = 1000;
        l = 1000;
    }
    else if (strcmp(sizeOption, "-k2500") == 0) {
        j = 2000;
        k = 2500;
        l = 3000;
    }

    std::vector<float> M_h;
    std::vector<float> N_h;
    std::vector<float> P_h(static_cast<size_t>(j) * l);
    if (load_mmult_inputs(sizeOption, j, k, l, M_h, N_h) != 0) {
        return 1;
    }

    serial_mmult(M_h.data(), N_h.data(), P_h.data(), j, k, l);
    if (write_mmult_csv("P_out_serial.csv", P_h.data(), j, l) != 0) {
        return 1;
    }

    if (strcmp(argv[1], "-naive") == 0) {
        if (naive_mmult(M_h.data(), N_h.data(), P_h.data(), j, k, l) != cudaSuccess) {
            return 1;
        }
        return write_mmult_csv("P_out_naive.csv", P_h.data(), j, l);
    }
    if (tiled_mmult(M_h.data(), N_h.data(), P_h.data(), j, k, l) != cudaSuccess) {
        return 1;
    }
    return write_mmult_csv("P_out_tiled.csv", P_h.data(), j, l);
}

int load_mmult_inputs(const char* sizeOption, int j, int k, int l,
    std::vector<float>& M_h, std::vector<float>& N_h)
{
    // MMULT input files live under inputs\ and contain headerless, row-major
    // IEEE-754 float32 values in little-endian byte order. M is j-by-k and N
    // is k-by-l; use each same file pair for the CPU reference and every GPU
    // variant so comparisons are reproducible. File names include the shape:
    //   100x100, k=100:       M_100x100_f32le.bin (100x100), N_100x100_f32le.bin (100x100)
    //   1000x1000, k=1000:    M_1000x1000_f32le.bin (1000x1000), N_1000x1000_f32le.bin (1000x1000)
    //   2000x3000, k=2500:    M_2000x2500_f32le.bin (2000x2500), N_2500x3000_f32le.bin (2500x3000)
    // Each case is in its correspondingly named inputs\case_* folder. The
    // files have no dimensions or metadata embedded; allocate/read the exact
    // element counts implied by the shapes above.

    const char* mFileName = nullptr;
    const char* nFileName = nullptr;
    if (strcmp(sizeOption, "-k100") == 0) {
        mFileName = "inputs\\case_100x100_k100\\M_100x100_f32le.bin";
        nFileName = "inputs\\case_100x100_k100\\N_100x100_f32le.bin";
    }
    else if (strcmp(sizeOption, "-k1000") == 0) {
        mFileName = "inputs\\case_1000x1000_k1000\\M_1000x1000_f32le.bin";
        nFileName = "inputs\\case_1000x1000_k1000\\N_1000x1000_f32le.bin";
    }
    else {
        mFileName = "inputs\\case_2000x3000_k2500\\M_2000x2500_f32le.bin";
        nFileName = "inputs\\case_2000x3000_k2500\\N_2500x3000_f32le.bin";
    }

    M_h.resize(static_cast<size_t>(j) * k);
    N_h.resize(static_cast<size_t>(k) * l);
    const std::streamsize expectedMBytes = static_cast<std::streamsize>(M_h.size() * sizeof(float));
    const std::streamsize expectedNBytes = static_cast<std::streamsize>(N_h.size() * sizeof(float));
    std::ifstream mFile(mFileName, std::ios::binary);
    std::ifstream nFile(nFileName, std::ios::binary);
    if (!mFile.is_open() || !nFile.is_open()) {
        fprintf(stderr, "[ERROR] Failed to open MMULT inputs: %s and %s\n", mFileName, nFileName);
        return 1;
    }

    mFile.seekg(0, std::ios::end);
    nFile.seekg(0, std::ios::end);
    if (mFile.tellg() != expectedMBytes || nFile.tellg() != expectedNBytes) {
        fprintf(stderr, "[ERROR] Input file size does not match the selected matrix dimensions.\n");
        return 1;
    }
    mFile.seekg(0, std::ios::beg);
    nFile.seekg(0, std::ios::beg);
    mFile.read(reinterpret_cast<char*>(M_h.data()), expectedMBytes);
    nFile.read(reinterpret_cast<char*>(N_h.data()), expectedNBytes);
    if (!mFile || !nFile) {
        fprintf(stderr, "[ERROR] Failed to read MMULT input matrices.\n");
        return 1;
    }

    return 0;
}

void serial_mmult(const float* M_h, const float* N_h, float* P_h, int j, int k, int l)
{
    // Row-major serial reference: P_h[row,col] = sum(M_h[row,inner] * N_h[inner,col]).
    for (int row = 0; row < j; ++row) {
        for (int col = 0; col < l; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum += M_h[static_cast<size_t>(row) * k + inner] * N_h[static_cast<size_t>(inner) * l + col];
            }
            P_h[static_cast<size_t>(row) * l + col] = sum;
        }
    }
}

int write_mmult_csv(const char* outputFileName, const float* P_h, int j, int l)
{
    std::ofstream outputFile(outputFileName, std::ios::trunc);
    if (!outputFile.is_open()) {
        fprintf(stderr, "[ERROR] Failed to open output file %s\n", outputFileName);
        return 1;
    }
    outputFile.imbue(std::locale::classic());
    outputFile << std::setprecision(std::numeric_limits<float>::max_digits10);
    // Headerless CSV: one matrix row per line, with l comma-separated values.
    for (int row = 0; row < j; ++row) {
        for (int col = 0; col < l; ++col) {
            if (col != 0) outputFile << ',';
            outputFile << P_h[static_cast<size_t>(row) * l + col];
        }
        outputFile << '\n';
    }
    outputFile.close();
    if (!outputFile) {
        fprintf(stderr, "[ERROR] Failed to write output file %s\n", outputFileName);
        return 1;
    }
    return 0;
}

cudaError_t naive_mmult(const float* M, const float* N, float* P, int j, int k, int l)
{
    if (M == nullptr || N == nullptr || P == nullptr || j <= 0 || k <= 0 || l <= 0) {
        return checkCuda(cudaErrorInvalidValue, "naive_mmult inputs and dimensions");
    }

    float* M_d = nullptr;
    float* N_d = nullptr;
    float* P_d = nullptr;
    const size_t mBytes = static_cast<size_t>(j) * k * sizeof(float);
    const size_t nBytes = static_cast<size_t>(k) * l * sizeof(float);
    const size_t pBytes = static_cast<size_t>(j) * l * sizeof(float);

    // Part 1: Allocate device memory and copy from host memory to device memory
    cudaError_t status = checkCuda(cudaMalloc(&M_d, mBytes), "cudaMalloc M_d");
    if (status == cudaSuccess) {
        status = checkCuda(cudaMalloc(&N_d, nBytes), "cudaMalloc N_d");
    }
    if (status == cudaSuccess) {
        status = checkCuda(cudaMalloc(&P_d, pBytes), "cudaMalloc P_d");
    }
    if (status == cudaSuccess) {
        status = checkCuda(cudaMemcpy(M_d, M, mBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D M");
    }
    if (status == cudaSuccess) {
        status = checkCuda(cudaMemcpy(N_d, N, nBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D N");
    }

    // Part 2: Call kernel - to launch a grid of threads to perform 2D operation
    if (status == cudaSuccess) {
        const dim3 threadsPerBlock(16, 16);
        const dim3 blocksPerGrid((l - 1) / threadsPerBlock.x + 1,
            (j - 1) / threadsPerBlock.y + 1);
        naive_mmult_kernel<<<blocksPerGrid, threadsPerBlock>>>(M_d, N_d, P_d, j, k, l);
        status = checkCuda(cudaGetLastError(), "naive_mmult_kernel launch");
    }
    if (status == cudaSuccess) {
        status = checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize naive_mmult_kernel");
    }

    // Part 3: Copy from device memory to host memory
    if (status == cudaSuccess) {
        status = checkCuda(cudaMemcpy(P, P_d, pBytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H P");
    }

    // Part 4: Free device memory, preserving the first error if cleanup also fails.
    if (P_d != nullptr) {
        const cudaError_t freeStatus = checkCuda(cudaFree(P_d), "cudaFree P_d");
        if (status == cudaSuccess) status = freeStatus;
    }
    if (N_d != nullptr) {
        const cudaError_t freeStatus = checkCuda(cudaFree(N_d), "cudaFree N_d");
        if (status == cudaSuccess) status = freeStatus;
    }
    if (M_d != nullptr) {
        const cudaError_t freeStatus = checkCuda(cudaFree(M_d), "cudaFree M_d");
        if (status == cudaSuccess) status = freeStatus;
    }
    return status;
}

cudaError_t tiled_mmult(const float* M, const float* N, float* P, int j, int k, int l)
{
    if (M == nullptr || N == nullptr || P == nullptr || j <= 0 || k <= 0 || l <= 0) {
        return checkCuda(cudaErrorInvalidValue, "tiled_mmult inputs and dimensions");
    }

    float* M_d = nullptr;
    float* N_d = nullptr;
    float* P_d = nullptr;
    const size_t mBytes = static_cast<size_t>(j) * k * sizeof(float);
    const size_t nBytes = static_cast<size_t>(k) * l * sizeof(float);
    const size_t pBytes = static_cast<size_t>(j) * l * sizeof(float);

    // Part 1: Allocate device memory and copy from host memory to device memory
    cudaError_t status = checkCuda(cudaMalloc(&M_d, mBytes), "cudaMalloc M_d");
    if (status == cudaSuccess) {
        status = checkCuda(cudaMalloc(&N_d, nBytes), "cudaMalloc N_d");
    }
    if (status == cudaSuccess) {
        status = checkCuda(cudaMalloc(&P_d, pBytes), "cudaMalloc P_d");
    }
    if (status == cudaSuccess) {
        status = checkCuda(cudaMemcpy(M_d, M, mBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D M");
    }
    if (status == cudaSuccess) {
        status = checkCuda(cudaMemcpy(N_d, N, nBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D N");
    }

    // Part 2: Call kernel - to launch a grid of threads to perform 2D operation
    if (status == cudaSuccess) {
        constexpr int TILE_WIDTH = 16;
        const dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
        const dim3 blocksPerGrid((l - 1) / threadsPerBlock.x + 1,
            (j - 1) / threadsPerBlock.y + 1);
        tiled_mmult_kernel<TILE_WIDTH><<<blocksPerGrid, threadsPerBlock>>>(M_d, N_d, P_d, j, k, l);
        status = checkCuda(cudaGetLastError(), "tiled_mmult_kernel launch");
    }
    if (status == cudaSuccess) {
        status = checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize tiled_mmult_kernel");
    }

    // Part 3: Copy from device memory to host memory
    if (status == cudaSuccess) {
        status = checkCuda(cudaMemcpy(P, P_d, pBytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H P");
    }

    // Part 4: Free device memory, preserving the first error if cleanup also fails.
    if (P_d != nullptr) {
        const cudaError_t freeStatus = checkCuda(cudaFree(P_d), "cudaFree P_d");
        if (status == cudaSuccess) status = freeStatus;
    }
    if (N_d != nullptr) {
        const cudaError_t freeStatus = checkCuda(cudaFree(N_d), "cudaFree N_d");
        if (status == cudaSuccess) status = freeStatus;
    }
    if (M_d != nullptr) {
        const cudaError_t freeStatus = checkCuda(cudaFree(M_d), "cudaFree M_d");
        if (status == cudaSuccess) status = freeStatus;
    }
    return status;
}
