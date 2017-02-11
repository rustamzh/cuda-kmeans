#pragma once


typedef struct {
	double dim1;
	double dim2;
	double dim3;
	double dim4;
	int cluster;
	int initcluster;
} unit;


///CPU functions

double calculateDistance(unit* point1, unit* point2);

void initializeCentroids(unit* centroids, int numofcentr); //host

void closestcentroid(unit* point, unit* centroids, int numofcentr);

void calculateMean(unit* points, unit* centroids, int numofcentr, int numofpoints);

void copydata(unit* centroidscentroids, unit* latter, int numberofelem);

void checkfinish(unit* centroids, unit* latter, int num, int* boolen);
