#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>
#include "common.h"

int main (int argc, char* argv[]) {
    int matrixCreationFlag = 0;
    int matrixCreationFGLFlag = 0;
    int intFlag = 0;
    int percentageFlag = 0;
    int percentageFactor = 4;
    int i, j;

    int rows = 0;
    int cols = 0;

    char outputFileName[1000];

    for(i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-createMatrix") == 0) {
            if(argc < 3) {
                printf("Not enough arguments. Usage: ./util -createMatrix [rows] [cols] [outputFileName]\n");
                return 0;
            } else {
                matrixCreationFlag = 1;
            } 
        }

        if(strcmp(argv[i], "-createMatrixFGL") == 0) {
            if(argc < 3) {
                printf("Not enough arguments. Usage: ./util -createMatrixFGL [rows] [cols] [outputFileName]\n");
                return 0;
            } else {
                matrixCreationFGLFlag = 1;
            } 
        }

        if(strcmp(argv[i], "-int") == 0) {
            intFlag = 1;
        }

        if(strcmp(argv[i], "-percent") == 0) {
            i++;
            percentageFactor = atoi(argv[i]);
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
                if(intFlag == 0) {
                    fprintf(fptr, "%f ", (float)(rand())/(float)(RAND_MAX));
                } else {
                    fprintf(fptr, "%f ", (float)((int)(rand() * 20000000)));
                }                
            }

            fprintf(fptr, "\n");
        }

        fclose(fptr);
        printf("Done\n");
    }

    if(matrixCreationFGLFlag == 1) {
        rows = atoi(argv[2]);
        cols = atoi(argv[3]);
        strcpy(outputFileName, argv[4]);
        FILE *fptr;
        fptr = fopen(outputFileName, "w");

        fprintf(fptr, "%d\n", rows);
        fprintf(fptr, "%d\n", cols);

        for(i = 0; i < rows; i++) {
            for(j = 0; j < cols; j ++) {
                
                if(j > (i - 4)) {
                    int flag = (int)(rand() % percentageFactor);

                    if(flag == 1) {
                        if(intFlag == 0) {
                            fprintf(fptr, "%f ", (float)(rand())/(float)(RAND_MAX));
                        } else {
                            fprintf(fptr, "%f ", (float)((int)(rand() * 20000000)));
                        }   
                    } else {
                        fprintf(fptr, "%f ", (float)(0));
                    }  
                } else {
                    fprintf(fptr, "%f ", (float)(0));
                }
            }

            fprintf(fptr, "\n");
        }

        fclose(fptr);
        printf("Done\n");
    }

    return 0;
}