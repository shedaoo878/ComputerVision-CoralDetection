//
//  main.c
//  CS136PROJ
//
//  Created by An Ho on 11/14/24.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "netpbm.h"
#include <stdio.h>
#include "textureDetection.h"
#include <time.h>



int main(int argc, const char * argv[]) {
    clock_t start, end;
    double cpu_time_used;
    
    start = clock();
    Image timepoint0 = readImage("/Users/anho/Desktop/CS136PROJ/images/coral.pgm");
    Image result = imageSegmentation(timepoint0, 4);
    writeImage(result, "/Users/anho/Desktop/CS136PROJ/writeImages/timepoint0.ppm");
    
    Image timepoint1 = readImage("/Users/anho/Desktop/CS136PROJ/images/coral2.pgm");
    Image result2 = imageSegmentation(timepoint1, 4);
    writeImage(result2, "/Users/anho/Desktop/CS136PROJ/writeImages/timepoint1.ppm");
     
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("%f", cpu_time_used);
    deleteImage(timepoint0);
    deleteImage(result);
    //deleteImage(timepoint1);
    //deleteImage(result2);
    return 0;
    //test
}




