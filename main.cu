#define _CRT_SECURE_NO_DEPRECATE

#include "math.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k_mean.h"
#include <ctime>
#include <chrono>
#include <iostream>

#include "k_meanCUDA.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#define tile_size 32
#define numofcentr  4
#define numofelements 100000
#define numberofiter  800


void readfile(unit* read) {


	FILE* fid;



	if ((fid = fopen("dataset.txt", "r")) == NULL)
	{
		
		printf("Cannot open file");
		return;

	}
	
	char first[] = "firstMiddle";
	char sec[] = "secondMiddle";
	char thir[] = "thirdMiddle";
	char forth[] = "forthMiddle";

	char temp[20];

	for (int i = 0; i < numofelements;i++)
	{
		fscanf(fid, "%lf %lf %lf %lf %s\n", &read[i].dim1, &read[i].dim2, &read[i].dim3, &read[i].dim4, temp);
		//printf("%d\n", i);
		read[i].cluster = -1;
		if (strcmp(temp, first) == 0) {
			read[i].initcluster = 0;
		}
		if (strcmp(temp, sec) == 0) {
			read[i].initcluster = 1;
		}
		if (strcmp(temp, thir) == 0) {
			read[i].initcluster = 2;
		}
		if (strcmp(temp, forth) == 0) {
			read[i].initcluster = 3;
		}


	}

	fclose(fid);


}
void printdata(unit* data, unit* centroids ) {
	//for (int i = 0; i < numofelements; i++) {
	//	printf("%f %f %f %f ", data[i].dim1, data[i].dim2, data[i].dim3, data[i].dim4);
	//	printf("%d %d\n", data[i].cluster, data[i].initcluster);

	//}
	for (int i = 0; i < numofcentr;i++) {
		printf("\nCentroid #%d : %f %f %f %f ", i, centroids[i].dim1, centroids[i].dim2, centroids[i].dim3, centroids[i].dim4);
	}
	
}

void CPU_k_mean() {



	unit* data = (unit*)malloc(numofelements * sizeof(unit));
	unit* centroids = (unit*)calloc(numofcentr, sizeof(unit));


	readfile(data);

	centroids[0].dim1 = 0.5848632019399136;
	centroids[0].dim2 = 0.611966857319303;
	centroids[0].dim3 = 0.5976614560049508;
	centroids[0].dim4 = 0.6043396335008441;

	centroids[1].dim1 = 0.4057890614036615;
	centroids[1].dim2 = 0.39943930016009804 ;
	centroids[1].dim3 = 0.36002085029896843 ;
	centroids[1].dim4 = 0.3767179191381278;

	centroids[2].dim1 = 0.8436451075785734 ;
	centroids[2].dim2 = 0.8411071987518389 ;
	centroids[2].dim3 = 0.8192596107916138 ;
	centroids[2].dim4 = 0.8403777939339493;

	centroids[3].dim1 = 0.1571485475682983 ;
	centroids[3].dim2 = 0.16268196743699023 ;
	centroids[3].dim3 = 0.16264258613702254 ;
	centroids[3].dim4 = 0.16980746682214473;

	int i = 0;
	////
	//	Code for measuring was taken from http://www.informit.com/articles/article.aspx?p=1881386&seqNum=2
	//  Beginning
	//
	////
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;
	std::time_t end_time;

	start = std::chrono::system_clock::now();
	for (; i < numberofiter;i++) {

		for (int k = 0; k < numofelements; k++) {

			closestcentroid(&data[k], centroids, numofcentr);

			
		}
		calculateMean(data, centroids, numofcentr, numofelements);

	}
	//
		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		end_time = std::chrono::system_clock::to_time_t(end);
		std::cout << " finished computation at " << std::ctime(&end_time)
			<< "elapsed time: " << elapsed_seconds.count() << "s\n";
		////
		//
		//End of time measuring
		////


	printdata(data, centroids);
	printf("\n %d", i);
	
	//printdata(data);
	//calculateMean(data, centroids, numofcentr, numofelements);
	free(data);
	free(centroids);
}

void GPU_k_mean() {


	unit* data = (unit*)malloc(numofelements * sizeof(unit));

	unit* centroids = (unit*)calloc(numofcentr, sizeof(unit));

	unit* d_data;
	unit* d_centroids;
	cudaMalloc((void**)&d_data, numofelements * sizeof(unit));
	cudaMalloc((void**)&d_centroids, numofcentr* sizeof(unit));

	readfile(data);
	///initializeCentroids(centroids, numofcentr);
	centroids[0].dim1 = 0.5848632019399136;
	centroids[0].dim2 = 0.611966857319303;
	centroids[0].dim3 = 0.5976614560049508;
	centroids[0].dim4 = 0.6043396335008441;

	centroids[1].dim1 = 0.4057890614036615;
	centroids[1].dim2 = 0.39943930016009804;
	centroids[1].dim3 = 0.36002085029896843;
	centroids[1].dim4 = 0.3767179191381278;

	centroids[2].dim1 = 0.8436451075785734;
	centroids[2].dim2 = 0.8411071987518389;
	centroids[2].dim3 = 0.8192596107916138;
	centroids[2].dim4 = 0.8403777939339493;

	centroids[3].dim1 = 0.1571485475682983;
	centroids[3].dim2 = 0.16268196743699023;
	centroids[3].dim3 = 0.16264258613702254;
	centroids[3].dim4 = 0.16980746682214473;

	int i = 0;
	////
	//	Code for measuring was taken from http://www.informit.com/articles/article.aspx?p=1881386&seqNum=2
	//  Beginning
	//
	////
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;
	std::time_t end_time;

	start = std::chrono::system_clock::now();


	dim3 dimBlock(tile_size, tile_size, 1);
	dim3 dimGrid((sqrt(numofelements) + tile_size) / tile_size, (sqrt(numofelements) + tile_size) / tile_size,1);

	for (; i < numberofiter; i++) {

		cudaMemcpy(d_data, data, numofelements * sizeof(unit), cudaMemcpyHostToDevice);
		cudaMemcpy(d_centroids, centroids, numofcentr * sizeof(unit), cudaMemcpyHostToDevice);

		closestcentroidGPU <<< dimGrid, dimBlock >>>(d_data, d_centroids, numofcentr, numofelements);
		cudaDeviceSynchronize();

		cudaMemcpy(data, d_data, numofelements * sizeof(unit), cudaMemcpyDeviceToHost);
		

		calculateMean(data, centroids, numofcentr, numofelements);
		cudaFree(d_centroids);

	}
	
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << " finished computation at " << std::ctime(&end_time)
		<< "elapsed time: " << elapsed_seconds.count() << "s\n";
	////
	//
	//End of time measuring
	////

	printdata(data, centroids);
	printf("\n %d", i);

	free(data);
	free(centroids);
	cudaFree(d_data);
	cudaFree(d_centroids);


}

void GPU_k_meanShared() {

	


	unit* sh_data = (unit*)malloc(numofelements * sizeof(unit));

	unit* sh_centroids = (unit*)calloc(numofcentr, sizeof(unit));

	unit* sh_d_data;
	unit* sh_d_centroids;
	cudaMalloc((void**)&sh_d_data, numofelements * sizeof(unit));
	cudaMalloc((void**)&sh_d_centroids, numofcentr* sizeof(unit));

	readfile(sh_data);
	///initializeCentroids(centroids, numofcentr);
	sh_centroids[0].dim1 = 0.5848632019399136;
	sh_centroids[0].dim2 = 0.611966857319303;
	sh_centroids[0].dim3 = 0.5976614560049508;
	sh_centroids[0].dim4 = 0.6043396335008441;

	sh_centroids[1].dim1 = 0.4057890614036615;
	sh_centroids[1].dim2 = 0.39943930016009804;
	sh_centroids[1].dim3 = 0.36002085029896843;
	sh_centroids[1].dim4 = 0.3767179191381278;

	sh_centroids[2].dim1 = 0.8436451075785734;
	sh_centroids[2].dim2 = 0.8411071987518389;
	sh_centroids[2].dim3 = 0.8192596107916138;
	sh_centroids[2].dim4 = 0.8403777939339493;

	sh_centroids[3].dim1 = 0.1571485475682983;
	sh_centroids[3].dim2 = 0.16268196743699023;
	sh_centroids[3].dim3 = 0.16264258613702254;
	sh_centroids[3].dim4 = 0.16980746682214473;

	int i = 0;
	////
	//	Code for measuring was taken from http://www.informit.com/articles/article.aspx?p=1881386&seqNum=2
	//  Beginning
	//
	////
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;
	std::time_t end_time;

	start = std::chrono::system_clock::now();


	dim3 dimBlock1(tile_size, tile_size, 1);
	dim3 dimGrid1((sqrt(numofelements) + tile_size) / tile_size, (sqrt(numofelements) + tile_size) / tile_size, 1);

	for (; i < numberofiter; i++) {

		cudaMemcpy(sh_d_data, sh_data, numofelements * sizeof(unit), cudaMemcpyHostToDevice);
		cudaMemcpy(sh_d_centroids, sh_centroids, numofcentr * sizeof(unit), cudaMemcpyHostToDevice);

		closestcentroidSharedGPU << < dimGrid1, dimBlock1 >> >(sh_d_data, sh_d_centroids, numofcentr, numofelements);
		cudaDeviceSynchronize();

		cudaMemcpy(sh_data, sh_d_data, numofelements * sizeof(unit), cudaMemcpyDeviceToHost);


		calculateMean(sh_data, sh_centroids, numofcentr, numofelements);
		cudaFree(sh_d_centroids);

	}
	//
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << " finished computation at " << std::ctime(&end_time)
		<< "elapsed time: " << elapsed_seconds.count() << "s\n";
	////
	//
	//End of time measuring
	////


	printdata(sh_data, sh_centroids);
	printf("\n %d", i);

	free(sh_data);
	free(sh_centroids);
	cudaFree(sh_d_data);
	cudaFree(sh_d_centroids);
}

int main() {
	CPU_k_mean();
	//system("pause");
	printf("\n\n\n");
	//GPU_k_mean();
	printf("\n\n\n");
	//system("pause");
	GPU_k_meanShared();
	return 0;
}
