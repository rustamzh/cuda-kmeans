#include "k_mean.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <cstdlib>



double calculateDistance(unit *point1, unit *point2)
{
	return sqrt(pow(point1->dim1 - point2->dim1, 2) + pow(point1->dim2 - point2->dim2, 2) + pow(point1->dim3 - point2->dim3, 2) + pow(point1->dim4 - point2->dim4, 2));
}

void initializeCentroids(unit* centroids, int numofcentr) {
	srand(time(NULL));
	for (int i = 0; i < numofcentr; i++) {
		centroids[i].dim1 = static_cast <double> (rand()) / (static_cast <float> (RAND_MAX / 1));
		centroids[i].dim2 = static_cast <double> (rand()) / (static_cast <float> (RAND_MAX / 1));
		centroids[i].dim3 = static_cast <double> (rand()) / (static_cast <float> (RAND_MAX / 1));
		centroids[i].dim4 = static_cast <double> (rand()) / (static_cast <float> (RAND_MAX / 1));
		//printf("%f %f %f %f \n", centroids[i].dim1, centroids[i].dim2, centroids[i].dim3, centroids[i].dim4);
	}

}


void closestcentroid(unit* point, unit* centroids, int numofcentr) {
	double dist = 0;
	double firstdistance = calculateDistance(point, &centroids[0]);
	point->cluster = 0;
	for (int i = 1; i < numofcentr; i++) {
		dist = calculateDistance(point, &centroids[i]);
		if (dist <= firstdistance) {
			point->cluster = i;
			firstdistance = dist;
		}
	}
}

void calculateMean(unit* points, unit* centroids, int numofcentr, int numofpoints) {

	unit* newCentr = (unit *)calloc(numofcentr, sizeof(unit));
	int pointsInCentroid = 0;

	for (int i = 0; i < numofcentr; i++) {

		for (int j = 0; j < numofpoints; j++) {

			if (points[j].cluster == i) {

				newCentr[i].dim1 += points[j].dim1;
				newCentr[i].dim2 += points[j].dim2;
				newCentr[i].dim3 += points[j].dim3;
				newCentr[i].dim4 += points[j].dim4;
				pointsInCentroid++;
			}

		}
		if (pointsInCentroid != 0) {
			newCentr[i].dim1 /= pointsInCentroid;
			newCentr[i].dim2 /= pointsInCentroid;
			newCentr[i].dim3 /= pointsInCentroid;
			newCentr[i].dim4 /= pointsInCentroid;
		}
		pointsInCentroid = 0;
		centroids[i].dim1 = newCentr[i].dim1;
		centroids[i].dim2 = newCentr[i].dim2;
		centroids[i].dim3 = newCentr[i].dim3;
		centroids[i].dim4 = newCentr[i].dim4;
	}


}
void copydata(unit* centroids, unit* latter, int numberofelem) {
	for (int i = 0; i < numberofelem;i++) {
		latter[i].cluster = centroids[i].cluster;

	}
}

void checkfinish(unit* centroids,unit* latter, int num,int* boolen) {
	for (int i = 0; i < num; i++) {
		if (latter[i].dim1 != centroids[i].dim1 || latter[i].dim2 != centroids[i].dim2 || latter[i].dim3 != centroids[i].dim3 || latter[i].dim4 != centroids[i].dim4)
			return;
	}
	*boolen = 1;
}