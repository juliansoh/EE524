#include "blur_kernels.cuh"
#include <math.h>

__global__ void blurKernelSharedDynamic(const unsigned char* in, unsigned char* out, int w, int h, const float* blurFilt_d)
{
	extern __shared__ float blurFilt_s[];

	int localTid = threadIdx.y * blockDim.x + threadIdx.x;
	if (localTid < BLUR_FILTER_WIDTH * BLUR_FILTER_WIDTH) {
		blurFilt_s[localTid] = blurFilt_d[localTid];
	}
	__syncthreads();

	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	if (Col < w && Row < h)
	{
		int pixVal = 0;
		float weightSum = 0.0f;

		for (int blurRow = -BLUR_FILTER_WIDTH / 2; blurRow <= BLUR_FILTER_WIDTH / 2; ++blurRow) {
			for (int blurCol = -BLUR_FILTER_WIDTH / 2; blurCol <= BLUR_FILTER_WIDTH / 2; ++blurCol) {
				int curRow = Row + blurRow;
				int curCol = Col + blurCol;

				if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
					int filtRow = blurRow + BLUR_FILTER_WIDTH / 2;
					int filtCol = blurCol + BLUR_FILTER_WIDTH / 2;
					float weight = blurFilt_s[filtRow * BLUR_FILTER_WIDTH + filtCol];

					pixVal += (int)roundf(weight * (float)in[curRow * w + curCol]);
					weightSum += weight;
				}
			}
		}

		int outPix = (weightSum > 0.0f) ? (int)roundf((float)pixVal / weightSum) : 0;
		if (outPix < 0) outPix = 0;
		if (outPix > 255) outPix = 255;
		out[Row * w + Col] = (unsigned char)outPix;
	}
}
