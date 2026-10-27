
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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


int main()
{
     // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    // const char filename[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage1.png";
    // const char filename[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage2.png";
    // const char filename[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage3.png";
    const char filename[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage4.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char* imgData = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
    int imgSize = x_cols * y_rows * sizeof(uchar);
    unsigned char* h_imgOut = (unsigned char*)malloc(imgSize);
    unsigned char* imgData_d = nullptr;
    unsigned char* imgOut_d = nullptr;
    float* blurFilt_d = nullptr;
    cudaError_t cudaStatus = cudaSuccess;
    LARGE_INTEGER perfFreq = {};
    LARGE_INTEGER timer1Start = {};
    LARGE_INTEGER timer1Stop = {};
    LARGE_INTEGER timer2Start = {};
    LARGE_INTEGER timer2Stop = {};
    double timer1Ms = 0.0;
    double kernel1Ms = 0.0;
    double kernel2Ms = 0.0;
    double kernel3Ms = 0.0;

    if (imgData == nullptr || h_imgOut == nullptr) {
        fprintf(stderr, "Failed to allocate/load image buffers.\n");
        goto Error;
    }

    if (!QueryPerformanceFrequency(&perfFreq)) {
        fprintf(stderr, "QueryPerformanceFrequency failed.\n");
        goto Error;
    }

    // setup additional host variables, allocate host memory as needed

    // START timer #1
    QueryPerformanceCounter(&timer1Start);

    // allocate device memory
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

    // copy host data to device
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

    // START timer #2 (kernel #1 launch + synchronize)
    QueryPerformanceCounter(&timer2Start);
    // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range
    dim3 blockSize(16, 16);
    dim3 gridSize((x_cols + blockSize.x - 1) / blockSize.x, (y_rows + blockSize.y - 1) / blockSize.y);

    // blur kernel #1 call (implemented in blur_kernel1.cu)
    // blurKernelGlobal<<<gridSize, blockSize>>>(imgData_d, imgOut_d, x_cols, y_rows);
    blurKernelGlobal<<<1, 1>>>(imgData_d, imgOut_d, x_cols, y_rows);

    // Check for any errors launching kernel #1
    cudaStatus = cudaGetLastError();
    checkCuda(cudaStatus, "blurKernelGlobal launch");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // STOP timer #2 (kernel #1 launch + synchronize)
    QueryPerformanceCounter(&timer2Stop);
    kernel1Ms = (double)(timer2Stop.QuadPart - timer2Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;

    // Wait for kernel #1 to finish
    cudaStatus = cudaDeviceSynchronize();
    checkCuda(cudaStatus, "cudaDeviceSynchronize kernel1");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // Retrieve and save kernel #1 output
    cudaStatus = cudaMemcpy(h_imgOut, imgOut_d, imgSize, cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus, "cudaMemcpy D2H imgOut kernel1");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }
    // const char imgFileOut1[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_outimage1.png";
    // const char imgFileOut1[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage2_out_kernel1.png";
    // const char imgFileOut1[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage3_out_kernel1.png";
    const char imgFileOut1[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage4_out_kernel1.png";
    stbi_write_png(imgFileOut1, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);

    // START timer #2 (kernel #2 launch + synchronize)
    QueryPerformanceCounter(&timer2Start);

    // blur kernel #2 call (implemented in blur_kernel2.cu)
    blurKernelSharedStatic<<<gridSize, blockSize>>>(imgData_d, imgOut_d, x_cols, y_rows, blurFilt_d);

    // Check for any errors launching kernel #2
    cudaStatus = cudaGetLastError();
    checkCuda(cudaStatus, "blurKernelSharedStatic launch");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // Wait for kernel #2 to finish
    cudaStatus = cudaDeviceSynchronize();
    checkCuda(cudaStatus, "cudaDeviceSynchronize kernel2");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // STOP timer #2 (kernel #2 launch + synchronize)
    QueryPerformanceCounter(&timer2Stop);
    kernel2Ms = (double)(timer2Stop.QuadPart - timer2Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;

    // 
    // retrieve kernel #2 result data from device back to host
    cudaStatus = cudaMemcpy(h_imgOut, imgOut_d, imgSize, cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus, "cudaMemcpy D2H imgOut kernel2");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

       // save kernel #2 result output image data to file
    // const char imgFileOut2[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_outimage2.png";
    // const char imgFileOut2[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage2_out_kernel2.png";
    // const char imgFileOut2[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage3_out_kernel2.png";
    const char imgFileOut2[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage4_out_kernel2.png";
    stbi_write_png(imgFileOut2, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);

    // START timer #2 (kernel #3 launch + synchronize)
    QueryPerformanceCounter(&timer2Start);

    // blur kernel #3 call (implemented in blur_kernel3.cu)
    size_t blurFiltBytes = BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH * sizeof(float);
    blurKernelSharedDynamic<<<gridSize, blockSize, blurFiltBytes>>>(imgData_d, imgOut_d, x_cols, y_rows, blurFilt_d);

    // Check for any errors launching kernel #3
    cudaStatus = cudaGetLastError();
    checkCuda(cudaStatus, "blurKernelSharedDynamic launch");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // STOP timer #2 (kernel #3 launch + synchronize)
    QueryPerformanceCounter(&timer2Stop);
    kernel3Ms = (double)(timer2Stop.QuadPart - timer2Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;

    // Wait for kernel #3 to finish
    cudaStatus = cudaDeviceSynchronize();
    checkCuda(cudaStatus, "cudaDeviceSynchronize kernel3");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // retrieve kernel #3 result data from device back to host
    cudaStatus = cudaMemcpy(h_imgOut, imgOut_d, imgSize, cudaMemcpyDeviceToHost);
    checkCuda(cudaStatus, "cudaMemcpy D2H imgOut kernel3");
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

    // save kernel #3 result output image data to file
    // const char imgFileOut3[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_outimage3.png";
    // const char imgFileOut3[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage2_out_kernel3.png";
    // const char imgFileOut3[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage3_out_kernel3.png";
    const char imgFileOut3[] = "E:\\Dev\\EE524\\AU26_GPUCompute\\Homework-2\\hw2_testimage4_out_kernel3.png";
    stbi_write_png(imgFileOut3, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);

    // STOP timer #1
    QueryPerformanceCounter(&timer1Stop);
    timer1Ms = (double)(timer1Stop.QuadPart - timer1Start.QuadPart) * 1000.0 / (double)perfFreq.QuadPart;

    // retrieve and save timer results (write to console or file)
    printf("Timer #1 total host-side path: %.3f ms\n", timer1Ms);
    printf("Timer #2 kernel #1 (launch+sync): %.3f ms\n", kernel1Ms);
    printf("Timer #2 kernel #2 (launch+sync): %.3f ms\n", kernel2Ms);
    printf("Timer #2 kernel #3 (launch+sync): %.3f ms\n", kernel3Ms);


    // EXTRA CREDIT:
    // start timer #3
    // run host sequential blur routine
    // stop timer #3

    // retrieve and save timer results (write to console or file)
 
Error:  // assumes error macro has a goto Error statement

    // free host and device memory
    if (imgData_d) cudaFree(imgData_d);
    if (imgOut_d) cudaFree(imgOut_d);
    if (blurFilt_d) cudaFree(blurFilt_d);
    if (imgData) stbi_image_free(imgData);
    if (h_imgOut) free(h_imgOut);

    return 0;
}


