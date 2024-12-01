//
//  textureDetection.h
//  CS136PROJ
//
//  Created by An Ho on 11/14/24.
//

#ifndef textureDetection_h
#define textureDetection_h

#include "netpbm.h"
#include <stdio.h>

int ValidPos(int y, int x, int height, int width);

Matrix convolve(Matrix m1, Matrix m2);

Matrix vectorMult(Matrix m1, Matrix m2);

typedef struct point {
    double feature[14];
} Point;

void avgNeighborhood(Matrix convolvedSquared[], int amount);

double maxMatrix(Matrix m1);

Point addPoint(Point p1, Point p2);

Point multPointVal(Point p1, double mult);

double distanceSquared(Point a, Point b);

Image imageSegmentation(Image img, int clusters);

Matrix averageMatrices(Matrix m1, Matrix m2);

#endif /* textureDetection_h */

