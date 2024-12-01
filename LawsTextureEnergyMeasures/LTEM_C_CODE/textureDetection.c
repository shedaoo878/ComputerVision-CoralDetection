//
//  textureDetection.c
//  CS136PROJ
//
//  Created by An Ho on 11/14/24.
//

#include "textureDetection.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>

double maxMatrix(Matrix m1){
    double max = DBL_MIN;
    for (int x = 0; x < m1.width; x++){
        for (int y = 0; y < m1.height; y++){
            if (max < m1.map[y][x]) {
                max = m1.map[y][x];
            }
        }
    }
    return max;
}

void avgNeighborhood(Matrix convolvedSquared[], int amount)
{
    // 5 is the size of the height and width of the filters
    int border = (int)amount / 2;
    for (int i = 0; i < 14; i++)
    {
        Matrix temp = createMatrix(convolvedSquared[i].height, convolvedSquared[i].width);
        for (int x = 0; x < convolvedSquared[i].height; x++)
        {
            for (int y = 0; y < convolvedSquared[i].width; y++)
            {
                if (!(x < border || x > (convolvedSquared[i].height - border - 1) || y < border || y > (convolvedSquared[i].width - border - 1)))
                {
                    for (int k = (border * -1); k < (border + 1); k++)
                    {
                        for (int j = (border * -1); j < (border + 1); j++)
                        {
                            temp.map[x][y] += convolvedSquared[i].map[x + k][y + j];
                        }
                    }
                    // computing the average
                    temp.map[x][y] /= pow(amount, 2);
                }
            }
        }
        convolvedSquared[i] = temp;
    }
}

Matrix averageMatrices(Matrix m1, Matrix m2){
    for (int x = 0; x < m1.width; x++){
        for (int y = 0; y < m1.height; y++){
            m1.map[y][x] = ((m1.map[y][x] + m2.map[y][x]) / 2.0);
        }
    }
    return m1;
}

int validPos(int y, int x, int height, int width)
{
    if(x < 0 || y < 0 || x > width - 1 || y > height -1)
        return 0;
    return 1;
}

Matrix convolve(Matrix m1, Matrix m2){
    Matrix convolvedMatrix = createMatrix(m1.height, m1.width);
    int leftBound = floor(m2.width/2);
    int upperBound = floor(m2.height/2);
    
    for (int x = 0; x < m1.width; x++){
        for (int y = 0; y < m1.height; y++){
            float accumulator = 0;
            if(m2.width % 2 == 0){
                for(int i = 1; i <= m2.width; i++){
                    for(int j = 1; j <= m2.height; j++){
                        if (validPos(y + j, x + i, m1.height, m1.width)){
                                accumulator += (m1.map[y+j][x+i] * m2.map[j+leftBound][i+upperBound]);
                        }
                    }
                }
            }
            else{
                for(int i = -leftBound; i < m2.width - leftBound; i++){
                    for(int j = -upperBound; j < m2.width - upperBound; j++){
                        if(validPos(y + j, x + i, m1.height, m1.width)){
                                accumulator += (m1.map[y+j][x+i] * m2.map[j+leftBound][i+upperBound]);
                        }
                    }
                }
            }
            convolvedMatrix.map[y][x] = accumulator;
        }
    }
    return convolvedMatrix;
}

//height = rows
//width = collumns
Matrix matrixMult(Matrix m1, Matrix m2){
    if (m1.width != m2.height) {
        printf("error incompatible matrix size for multiplication");
        return m1;
    }
    
    Matrix product = createMatrix(m1.height, m2.width);
    
    for (int i = 0; i < m1.height; i++) {
        for(int j = 0; j < m2.width; j++){
            float sum = 0;
   
            for(int k = 0; k < m1.width; k++){
                sum = sum + (m1.map[i][k] * m2.map[k][j]);
            }
            product.map[i][j] = sum;
        }
    }
    return product;
}

Point addPoint(Point p1, Point p2){
    Point sum = { {0} };

    for (int i = 0; i < 14; i++) {
        sum.feature[i] = p1.feature[i] + p2.feature[i];
    }
    return sum;
}

Point multPointVal(Point p1, double mult){
    Point result = { {0} };

    for (int i = 0; i < 14; i++) {
        result.feature[i] = p1.feature[i] * mult;
    }
    return result;
}

//height = rows
//width = collumns
//matrix notation row x collumn
Matrix rotateMatrix(Matrix m1){
    Matrix rotMat = createMatrix(m1.width, m1.height);

    for (int i = 0; i < rotMat.width; i++)
    {
        for(int j = 0; j < rotMat.height; j++){
            rotMat.map[j][i] = m1.map[i][j];
        }
    }
    return rotMat;
}


double distanceSquared(Point a, Point b){
    double distance = 0;
    for (int i = 0; i < 14; i++) {
        distance += (a.feature[i] - b.feature[i]) * (a.feature[i] - b.feature[i]);
    }

    return distance;
}


//height = rows
//width = collumns
//matrix notation row x collumn
Image imageSegmentation(Image img, int clusters){

    if(clusters <= 0)
        clusters = 2;
    clusters = clusters + 1;
    
    Matrix l = createMatrix(5,1);
    l.map[0][0] = 1;
    l.map[1][0] = 4;
    l.map[2][0] = 6;
    l.map[3][0] = 4;
    l.map[4][0] = 1;
    /*
     {
     1
     2
     3
     4
     5}
     */
    Matrix lt = rotateMatrix(l);
    /* {1,2,3,4,5}*/
    
    
    Matrix s = createMatrix(5, 1);
    s.map[0][0] = -1;
    s.map[1][0] = 0;
    s.map[2][0] = 2;
    s.map[3][0] = 0;
    s.map[4][0] = -1;
    Matrix st = rotateMatrix(s);
    
    Matrix e = createMatrix(5,1);
    e.map[0][0] = -1;
    e.map[1][0] = -2;
    e.map[2][0] = 0;
    e.map[3][0] = 2;
    e.map[4][0] = 1;
    Matrix et = rotateMatrix(e);

    Matrix r = createMatrix(5,1);
    r.map[0][0] = 1;
    r.map[1][0] = -4;
    r.map[2][0] = 6;
    r.map[3][0] = -4;
    r.map[4][0] = 1;
    Matrix rt = rotateMatrix(r);

    Matrix w = createMatrix(5,1);
    w.map[0][0] = -1;
    w.map[1][0] = 2;
    w.map[2][0] = 0;
    w.map[3][0] = -2;
    w.map[4][0] = 1;
    Matrix wt = rotateMatrix(w);
  
    Matrix lawsMatrices[25];
    
    

    Matrix ll = matrixMult(l, lt);
    Matrix le = matrixMult(l, et);
    Matrix ls = matrixMult(l, st);
    Matrix lw = matrixMult(l, wt);
    Matrix lr = matrixMult(l, rt);

    Matrix el = matrixMult(e, lt);
    Matrix ee = matrixMult(e, et);
    Matrix es = matrixMult(e, st);
    Matrix ew = matrixMult(e, wt);
    Matrix er = matrixMult(e, rt);

    Matrix sl = matrixMult(s, lt);
    Matrix se = matrixMult(s, et);
    Matrix ss = matrixMult(s, st);
    Matrix sw = matrixMult(s, wt);
    Matrix sr = matrixMult(s, rt);
    
    Matrix wl = matrixMult(w, lt);
    Matrix we = matrixMult(w, et);
    Matrix ws = matrixMult(w, st);
    Matrix ww = matrixMult(w, wt);
    Matrix wr = matrixMult(w, rt);
    
    Matrix rl = matrixMult(r, lt);
    Matrix re = matrixMult(r, et);
    Matrix rs = matrixMult(r, st);
    Matrix rw = matrixMult(r, wt);
    Matrix rr = matrixMult(r, rt);

    
    lawsMatrices[0] = ll;
    lawsMatrices[1] = le;
    lawsMatrices[2] = ls;
    lawsMatrices[3] = lw;
    lawsMatrices[4] = lr;
    
    lawsMatrices[5] = el;
    lawsMatrices[6] = ee;
    lawsMatrices[7] = es;
    lawsMatrices[8] = ew;
    lawsMatrices[9] = er;

    lawsMatrices[10] = sl;
    lawsMatrices[11] = se;
    lawsMatrices[12] = ss;
    lawsMatrices[13] = sw;
    lawsMatrices[14] = sr;
    
    lawsMatrices[15] = wl;
    lawsMatrices[16] = we;
    lawsMatrices[17] = ws;
    lawsMatrices[18] = ww;
    lawsMatrices[19] = wr;

    lawsMatrices[20] = rl;
    lawsMatrices[21] = re;
    lawsMatrices[22] = rs;
    lawsMatrices[23] = rw;
    lawsMatrices[24] = rr;
     
    
    Matrix lawsEnergyMeasures[25];

    Matrix imgMat = image2Matrix(img);
    
    for(int i = 0; i < 25; i++){
        lawsEnergyMeasures[i] = convolve(imgMat, lawsMatrices[i]);
    }
    
    Matrix processedLawsEnergyMeasures[14];

    processedLawsEnergyMeasures[0] = lawsEnergyMeasures[0]; //ll
    processedLawsEnergyMeasures[1] = lawsEnergyMeasures[12]; //ss
    processedLawsEnergyMeasures[2] = lawsEnergyMeasures[18]; //ww
    processedLawsEnergyMeasures[3] = lawsEnergyMeasures[24]; //rr
    processedLawsEnergyMeasures[4] = averageMatrices(lawsEnergyMeasures[1], lawsEnergyMeasures[5]); // avg le el
    processedLawsEnergyMeasures[5] = averageMatrices(lawsEnergyMeasures[2], lawsEnergyMeasures[10]); // avg ls sl
    processedLawsEnergyMeasures[6] = averageMatrices(lawsEnergyMeasures[3], lawsEnergyMeasures[15]); // avg lw wl
    processedLawsEnergyMeasures[7] = averageMatrices(lawsEnergyMeasures[4], lawsEnergyMeasures[20]); // avg lr rl
    processedLawsEnergyMeasures[8] = averageMatrices(lawsEnergyMeasures[7], lawsEnergyMeasures[11]); // avg es se
    processedLawsEnergyMeasures[9] = averageMatrices(lawsEnergyMeasures[8], lawsEnergyMeasures[16]); // avg ew we
    processedLawsEnergyMeasures[10] = averageMatrices(lawsEnergyMeasures[9], lawsEnergyMeasures[21]); // avg er re
    processedLawsEnergyMeasures[11] = averageMatrices(lawsEnergyMeasures[13], lawsEnergyMeasures[17]); // avg sw ws
    processedLawsEnergyMeasures[12] = averageMatrices(lawsEnergyMeasures[14], lawsEnergyMeasures[22]); // avg sr rs
    processedLawsEnergyMeasures[13] = averageMatrices(lawsEnergyMeasures[19], lawsEnergyMeasures[23]); // avg rwr rw

    

    
    
    
    avgNeighborhood(processedLawsEnergyMeasures, 16);
    
    double maxFeatures[14];
    
    for (int i = 0; i < 14; i++) {
        maxFeatures[i] = maxMatrix(processedLawsEnergyMeasures[i]);
    }
    

    Point** featureVectors = (Point**)malloc(img.height * sizeof(Point*));

    for (int i = 0; i < img.height; i++) {
        featureVectors[i] = (Point*)malloc(img.width * sizeof(Point));
    }
  

    
    
    
    
    
    
    //create array of points that match the image
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++){
            for(int i = 0; i < 14; i++){
                if (i == 0)
                    featureVectors[y][x].feature[i] = processedLawsEnergyMeasures[i].map[y][x] / maxFeatures[i] * 2;
                else
                    featureVectors[y][x].feature[i] = processedLawsEnergyMeasures[i].map[y][x] / maxFeatures[i];
            }
        }
    }
    
  
    
    srand(0);
    
    Point* centroids = (Point*)malloc(clusters * sizeof(Point));
    
    //centroid initialization
    for (int i = 0 ; i < clusters; i++) {
        int clusterX = rand() % img.width;
        int clusterY = rand() % img.height;
        
        centroids[i] = featureVectors[clusterY][clusterX];
    }
    
    
    int **assignments;
    assignments = malloc(sizeof(int*) * img.height);
   
    for (int i = 0; i < img.height; i++) {
        assignments[i] = malloc(img.width * sizeof(int));
    }
    
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {

            assignments[y][x] = 0;
        }
    }

    
    //we create assignments here if they are background do not consider them?
    int changed = 0;
    int numReassigned = 0;
    int iterations = 0;
    do{
        changed = 0;
        numReassigned = 0;

        for (int x = 0; x < img.width; x++) {
            for (int y = 0; y < img.height; y++) {
                int assignment = assignments[y][x];
                if (assignment == -1) {
                    continue;
                }
                float minDist = distanceSquared(featureVectors[y][x], centroids[assignment]);
                for (int k = 0; k < clusters; k++) {
                    float dist = distanceSquared(featureVectors[y][x], centroids[k]);
                    if (dist < minDist) {
                        minDist = dist;
                        assignment = k;
                        changed = 1;
                        numReassigned++;
                    }
                    
                    assignments[y][x] = assignment;
                }
            }
        }
        
        printf("%d \n", numReassigned);
        //recalculate centroids
        for(int k = 0; k < clusters; k++){
            Point sum = { {0} };
            double count = 0;
            for(int x = 0; x < img.width; x++){
                for(int y = 0; y < img.height; y++){
                    if (assignments[y][x] == k) {
                        sum = addPoint(sum, featureVectors[y][x]);
                        count = count + 1;
                    }
                }
            }
            centroids[k] = multPointVal(sum, (double)(1.0/count));
        }
  
        iterations++;
    }while(iterations < 2 );
    
    
    Image segmentedImage = createImage(img.height, img.width);
    
    float redTable[clusters];
    float blueTable[clusters];
    float greenTable[clusters];
    float intensityTable[clusters];
    for (int i = 0; i < clusters; i++) {
        redTable[i] = rand() % 255;
        blueTable[i] = rand() % 255;
        greenTable[i] = rand() % 255;
        intensityTable[i] = rand() % 255;
    }

    for (int x = 0; x < img.width; x++) {
        for(int y = 0; y < img.height; y++){
            if (assignments[y][x] == -1) {
                segmentedImage.map[y][x].r = 0;
                segmentedImage.map[y][x].g = 0;
                segmentedImage.map[y][x].b = 0;
                segmentedImage.map[y][x].i = 0;
                continue;
            }
            
            
            segmentedImage.map[y][x].r = redTable[assignments[y][x]];
            segmentedImage.map[y][x].g = greenTable[assignments[y][x]];
            segmentedImage.map[y][x].b = blueTable[assignments[y][x]];
            segmentedImage.map[y][x].i = intensityTable[assignments[y][x]];
        }
    }

    
    for(int i=0;i<img.height;i++)
    {
        free(featureVectors[i]);
    }
    free(featureVectors);
    
    free(centroids);
    
    for(int i=0;i<img.height;i++)
    {
        free(assignments[i]);
    }
    free(assignments);
    
    //create colors and color based on the clustering 2darray
    return segmentedImage;
}
