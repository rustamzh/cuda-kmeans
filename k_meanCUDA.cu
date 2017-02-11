#include "k_meanCUDA.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"
#include "k_mean.h"

#define tile 32

__device__ double calculatedistanceGPU(unit* point1, unit* point2) {
	return (double)sqrt((double)pow(point1->dim1 - point2->dim1, 2) + (double)pow(point1->dim2 - point2->dim2, 2) + (double)pow(point1->dim3 - point2->dim3, 2) + (double)pow(point1->dim4 - point2->dim4, 2));
}

__global__ void closestcentroidGPU(unit* points, unit* centroids, int numofcentr, int numofpoints) {

	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	int threadPosInBlock = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	int blockPosInGrid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
	int tid = blockPosInGrid * threadsPerBlock + threadPosInBlock;

	if (tid < numofpoints) {
		double dist = 0;
		double firstdistance = calculatedistanceGPU(&points[tid], &centroids[0]);
		points[tid].cluster = 0;
		for (int i = 1; i < numofcentr; i++) {
			dist = calculatedistanceGPU(&points[tid], &centroids[i]);
			if (dist <= firstdistance) {
				points[tid].cluster = i;
				firstdistance = dist;
			}
		}
	}
}

__global__ void closestcentroidSharedGPU(unit* points, unit* centroids, int numofcentr, int numofpoints) {

	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	int threadPosInBlock = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	int blockPosInGrid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
	int tid = blockPosInGrid * threadsPerBlock + threadPosInBlock;

	__shared__ unit sh_points[tile*tile];
	__shared__ unit sh_centrs[4];
	
	if (tid < numofpoints) {
		sh_points[threadPosInBlock] = points[tid];
		if (tid%threadsPerBlock ==0) {
			for (int i = 0; i < numofcentr; i++) {
				sh_centrs[i] = centroids[i];
			}
			
		}
		__syncthreads();

		double dist = 0;
		double firstdistance = calculatedistanceGPU(&sh_points[threadPosInBlock], &sh_centrs[0]);
		sh_points[threadPosInBlock].cluster = 0;
		
		for (int i = 1; i < numofcentr; i++) {
			dist = calculatedistanceGPU(&sh_points[threadPosInBlock], &sh_centrs[i]);
			if (dist <= firstdistance) {
				sh_points[threadPosInBlock].cluster = i;
				firstdistance = dist;
			}
		}
		__syncthreads();

		points[tid] = sh_points[threadPosInBlock];
		
		__syncthreads();
	}
}