#pragma once

#include "cuda_runtime.h"
#include <stdio.h>

inline cudaError_t checkCuda(cudaError_t result, const char* srcStr)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s: %s\n", srcStr, cudaGetErrorString(result));
	}
	//print to console that CUDA runtime was successful
	fprintf(stderr, "CUDA Runtime Success: %s\n", srcStr);
	return result;
}
