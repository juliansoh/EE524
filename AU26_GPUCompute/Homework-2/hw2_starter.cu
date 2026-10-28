
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <Windows.h>
#include <profileapi.h>
#include "cuda-error-checking.h"
#include "blur_kernels.cuh"

typedef unsigned char uchar;

#define STB_IMAGE_IMPLEMENTATION // this is needed
#include "util/stb_image.h"  // download from class website files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/stb_image_write.h"  // download from class website files

// #include your error-check macro header file here
// This is in the included cuda-error-checking.h file

// global gaussian blur filter coefficients array here
const float BLUR_FILT[81] = { 0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.3292,0.5353,0.7575,0.9329,1.0000,0.9329,0.7575,0.5353,0.3292,0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084};

// Define CUDA kernel prototypes here (see blur_kernels.cuh for reference)
// Kernel implementations are defined in:
// - blur_kernel1.cu
// - blur_kernel2.cu
// - blur_kernel3.cu
// ** Shared kernel definitions are in blur_kernels.cuh


// EXTRA CREDIT
// define host sequential blur-kernel routine
struct TimingRow
{
    int imageId;
    int width;
    int height;
    double timer1TotalMs;
    double kernel1Ms;
    double kernel2Ms;
    double kernel3Ms;
    double cpuMs;
};

void blurHostSequential(const unsigned char* in, unsigned char* out, int w, int h)
{
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int pixVal = 0;
            float weightSum = 0.0f;

            for (int blurRow = -BLUR_FILTER_WIDTH / 2; blurRow <= BLUR_FILTER_WIDTH / 2; ++blurRow) {
                for (int blurCol = -BLUR_FILTER_WIDTH / 2; blurCol <= BLUR_FILTER_WIDTH / 2; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                        int filtRow = blurRow + BLUR_FILTER_WIDTH / 2;
                        int filtCol = blurCol + BLUR_FILTER_WIDTH / 2;
                        float weight = BLUR_FILT[filtRow * BLUR_FILTER_WIDTH + filtCol];

                        pixVal += (int)roundf(weight * (float)in[curRow * w + curCol]);
                        weightSum += weight;
                    }
                }
            }

            int outPix = (weightSum > 0.0f) ? (int)roundf((float)pixVal / weightSum) : 0;
            if (outPix < 0) outPix = 0;
            if (outPix > 255) outPix = 255;
            out[row * w + col] = (unsigned char)outPix;
        }
    }
}

bool runBlurCase(const char* filename, int imageId, const LARGE_INTEGER& perfFreq, TimingRow* row, FILE* resultsFile)
{
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* imgData = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
    size_t imgSize = (size_t)x_cols * (size_t)y_rows * sizeof(uchar);
    unsigned char* h_imgOut = (unsigned char*)malloc(imgSize);
    unsigned char* h_imgCpuOut = (unsigned char*)malloc(imgSize);
    unsigned char* imgData_d = nullptr;
    unsigned char* imgOut_d = nullptr;
    float* blurFilt_d = nullptr;
    cudaError_t cudaStatus = cudaSuccess;

    LARGE_INTEGER timer1Start = {};
    LARGE_INTEGER timer1Stop = {};
    LARGE_INTEGER timer2Start = {};
    LARGE_INTEGER timer2Stop = {};
    LARGE_INTEGER timer3Start = {};
    LARGE_INTEGER timer3Stop = {};

    double timer1Ms = 0.0;
    double kernel1Ms = 0.0;
    double kernel2Ms = 0.0;
    double kernel3Ms = 0.0;
    double cpuMs = 0.0;

    if (imgData == nullptr || h_imgOut == nullptr || h_imgCpuOut == nullptr) {
        fprintf(stderr, "Failed to allocate/load image buffers for image %d.\n", imageId);
        goto Error;
    }

    QueryPerformanceCounter(&timer1Start);

    cudaStatus = cudaMalloc((void**)&imgData_d, imgSize);
    checkCuda(cudaStatus, "cudaMalloc imgData_d");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&imgOut_d, imgSize);
    checkCuda(cudaStatus, "cudaMalloc imgOut_d");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&blurFilt_d, BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH * sizeof(float));
    checkCuda(cudaStatus, "cudaMalloc blurFilt_d");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(imgData_d, imgData, imgSize, cudaMemcpyHostToDevice);
    checkCuda(cudaStatus, "cudaMemcpy H2D imgData");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    cudaStatus = cudaMemcpy(blurFilt_d, BLUR_FILT, BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    checkCuda(cudaStatus, "cudaMemcpy H2D blurFilt");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((x_cols + blockSize.x - 1) / blockSize.x, (y_rows + blockSize.y - 1) / blockSize.y);

    QueryPerformanceCounter(&timer2Start);
    blurKernelGlobal<<<gridSize, blockSize>>>(imgData_d, imgOut_d, x_cols, y_rows);
    cudaStatus = cudaGetLastError();
    checkCuda(cudaStatus, "blurKernelGlobal launch");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    checkCuda(cudaStatus, "cudaDeviceSynchronize kernel1");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    QueryPerformanceCounter(&timer2Stop);
    kernel1Ms = (double)(timer2Stop.QuadPart - timer2Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;

    cudaStatus = cudaMemcpy(h_imgOut, imgOut_d, imgSize, cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus, "cudaMemcpy D2H imgOut kernel1");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    char imgFileOut1[260] = {};
    snprintf(imgFileOut1, sizeof(imgFileOut1), "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage%d_out_kernel1.png", imageId);
    stbi_write_png(imgFileOut1, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);

    QueryPerformanceCounter(&timer2Start);
    blurKernelSharedStatic<<<gridSize, blockSize>>>(imgData_d, imgOut_d, x_cols, y_rows, blurFilt_d);
    cudaStatus = cudaGetLastError();
    checkCuda(cudaStatus, "blurKernelSharedStatic launch");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    checkCuda(cudaStatus, "cudaDeviceSynchronize kernel2");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    QueryPerformanceCounter(&timer2Stop);
    kernel2Ms = (double)(timer2Stop.QuadPart - timer2Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;

    cudaStatus = cudaMemcpy(h_imgOut, imgOut_d, imgSize, cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus, "cudaMemcpy D2H imgOut kernel2");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    char imgFileOut2[260] = {};
    snprintf(imgFileOut2, sizeof(imgFileOut2), "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage%d_out_kernel2.png", imageId);
    stbi_write_png(imgFileOut2, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);

    QueryPerformanceCounter(&timer2Start);
    size_t blurFiltBytes = BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH * sizeof(float);
    blurKernelSharedDynamic<<<gridSize, blockSize, blurFiltBytes>>>(imgData_d, imgOut_d, x_cols, y_rows, blurFilt_d);
    cudaStatus = cudaGetLastError();
    checkCuda(cudaStatus, "blurKernelSharedDynamic launch");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    checkCuda(cudaStatus, "cudaDeviceSynchronize kernel3");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    QueryPerformanceCounter(&timer2Stop);
    kernel3Ms = (double)(timer2Stop.QuadPart - timer2Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;

    cudaStatus = cudaMemcpy(h_imgOut, imgOut_d, imgSize, cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus, "cudaMemcpy D2H imgOut kernel3");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    char imgFileOut3[260] = {};
    snprintf(imgFileOut3, sizeof(imgFileOut3), "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage%d_out_kernel3.png", imageId);
    stbi_write_png(imgFileOut3, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);

    QueryPerformanceCounter(&timer3Start);
    blurHostSequential(imgData, h_imgCpuOut, x_cols, y_rows);
    QueryPerformanceCounter(&timer3Stop);
    cpuMs = (double)(timer3Stop.QuadPart - timer3Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;
    char imgFileOutCpu[260] = {};
    snprintf(imgFileOutCpu, sizeof(imgFileOutCpu), "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage%d_out_cpu.png", imageId);
    stbi_write_png(imgFileOutCpu, x_cols, y_rows, 1, h_imgCpuOut, x_cols * n_pixdepth);

    QueryPerformanceCounter(&timer1Stop);
    timer1Ms = (double)(timer1Stop.QuadPart - timer1Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;

    row->imageId = imageId;
    row->width = x_cols;
    row->height = y_rows;
    row->timer1TotalMs = timer1Ms;
    row->kernel1Ms = kernel1Ms;
    row->kernel2Ms = kernel2Ms;
    row->kernel3Ms = kernel3Ms;
    row->cpuMs = cpuMs;

    fprintf(resultsFile, "Image %d (%dx%d): CPU=%.3f ms, K1=%.3f ms, K2=%.3f ms, K3=%.3f ms, Total=%.3f ms\n",
        row->imageId, row->width, row->height, row->cpuMs, row->kernel1Ms, row->kernel2Ms, row->kernel3Ms, row->timer1TotalMs);
    fflush(resultsFile);

    if (imgData_d) cudaFree(imgData_d);
    if (imgOut_d) cudaFree(imgOut_d);
    if (blurFilt_d) cudaFree(blurFilt_d);
    if (imgData) stbi_image_free(imgData);
    if (h_imgOut) free(h_imgOut);
    if (h_imgCpuOut) free(h_imgCpuOut);
    return true;

Error:
    if (imgData_d) cudaFree(imgData_d);
    if (imgOut_d) cudaFree(imgOut_d);
    if (blurFilt_d) cudaFree(blurFilt_d);
    if (imgData) stbi_image_free(imgData);
    if (h_imgOut) free(h_imgOut);
    if (h_imgCpuOut) free(h_imgCpuOut);
    return false;
}


int main()
{
    // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    // const char activeInputFile[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage1.png";
    // const char activeInputFile[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage2.png";
    // const char activeInputFile[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage3.png";
    const char activeInputFile[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage4.png";

    const char* allInputFiles[4] = {
        "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage1.png",
        "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage2.png",
        "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage3.png",
        "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage4.png"
    };
    int imageIds[4] = { 1, 2, 3, 4 };

    LARGE_INTEGER perfFreq = {};
    if (!QueryPerformanceFrequency(&perfFreq)) {
        fprintf(stderr, "QueryPerformanceFrequency failed.\n");
        return 1;
    }

    const char resultsPath[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\extra_credit.txt";
    FILE* resultsFile = fopen(resultsPath, "w");
    if (resultsFile == nullptr) {
        fprintf(stderr, "Failed to open results file: %s\n", resultsPath);
        return 1;
    }

    fprintf(resultsFile, "HW2 Extra Credit Timing Results\n");
    fprintf(resultsFile, "=============================================\n");
    fprintf(resultsFile, "Active input runs first: %s\n\n", activeInputFile);

    TimingRow rows[4] = {};
    int rowCount = 0;

    int runOrder[4] = { -1, -1, -1, -1 };
    int runCount = 0;
    for (int i = 0; i < 4; i++) {
        if (strcmp(allInputFiles[i], activeInputFile) == 0) {
            runOrder[runCount++] = i;
            break;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (strcmp(allInputFiles[i], activeInputFile) != 0) {
            runOrder[runCount++] = i;
        }
    }

    for (int runIdx = 0; runIdx < runCount; runIdx++) {
        int caseIdx = runOrder[runIdx];
        bool ok = runBlurCase(allInputFiles[caseIdx], imageIds[caseIdx], perfFreq, &rows[rowCount], resultsFile);
        if (ok) {
            printf("Completed image %d\n", imageIds[caseIdx]);
            rowCount++;
        }
        else {
            fprintf(resultsFile, "Image %d failed to run.\n", imageIds[caseIdx]);
            fflush(resultsFile);
            printf("Failed image %d\n", imageIds[caseIdx]);
        }
    }

    fprintf(resultsFile, "\nComparison Table\n");
    fprintf(resultsFile, "---------------------------------------------------------------------------------------------------------------\n");
    fprintf(resultsFile, "| Image | Resolution | CPU (ms) | Kernel1 (ms) | Kernel2 (ms) | Kernel3 (ms) | K1 Speedup | K2 Speedup | K3 Speedup |\n");
    fprintf(resultsFile, "---------------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < rowCount; i++) {
        double k1Speedup = (rows[i].kernel1Ms > 0.0) ? (rows[i].cpuMs / rows[i].kernel1Ms) : 0.0;
        double k2Speedup = (rows[i].kernel2Ms > 0.0) ? (rows[i].cpuMs / rows[i].kernel2Ms) : 0.0;
        double k3Speedup = (rows[i].kernel3Ms > 0.0) ? (rows[i].cpuMs / rows[i].kernel3Ms) : 0.0;
        fprintf(resultsFile, "| %5d | %4dx%-4d | %8.3f | %12.3f | %12.3f | %12.3f | %10.2f | %10.2f | %10.2f |\n",
            rows[i].imageId, rows[i].width, rows[i].height, rows[i].cpuMs, rows[i].kernel1Ms, rows[i].kernel2Ms, rows[i].kernel3Ms,
            k1Speedup, k2Speedup, k3Speedup);
    }
    fprintf(resultsFile, "---------------------------------------------------------------------------------------------------------------\n");

    fclose(resultsFile);
    return 0;
}


