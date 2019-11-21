#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>
#include "common.h"

int main (int argc, char* argv[]) {
    int matrixCreationFlag = 0;
    int i, j;

    int rows = 0;
    int cols = 0;

    char outputFileName[1000];

    for(i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-createMatrix") == 0) {
            if(argc < 3) {
                printf("Not enough arguments. Usage: ./util -createMatrix [rows] [cols] [outputFileName]\n");
                return 0;
            }
            else {
                matrixCreationFlag = 1;
            }
        }
    }

    if(matrixCreationFlag == 1) {
        rows = atoi(argv[2]);
        cols = atoi(argv[3]);
        strcpy(outputFileName, argv[4]);
        FILE *fptr;
        fptr = fopen(outputFileName, "w");

        fprintf(fptr, "%d\n", rows);
        fprintf(fptr, "%d\n", cols);

        for(i = 0; i < rows; i++) {
            for(j = 0; j < cols; j ++) {
                fprintf(fptr, "%f ", (float)(rand())/(float)(RAND_MAX));
            }

            fprintf(fptr, "\n");
        }

        fclose(fptr);
    }

    return 0;
}