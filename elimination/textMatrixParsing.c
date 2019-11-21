#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include "common.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

float** parseInputMatrix(FILE *f, int maxLineSize, int* resultRows, int* resultCols) {
    float** resultMatrix;

    int i = 0;
    int j = 0;
    int rows = 0;
    int cols = 0;
    char str[maxLineSize];

    if(fgets(str, maxLineSize, f) != NULL) {
        rows = atoi(str);
    }

    if(fgets(str, maxLineSize, f) != NULL) {
        cols = atoi(str);
    }

    resultMatrix = (float**) malloc (rows * sizeof(float*));

    for(i = 0; i < rows; i++) {
        resultMatrix[i] = malloc (cols * sizeof(float));        
    }

    for(i = 0; i < rows; i ++) {
        for(j = 0; j < cols; j++) {
            fscanf(f, "%f", &resultMatrix[i][j]);
        }        
    }

    *resultRows = rows;
    *resultCols = cols;

    return resultMatrix;
}

void printMatrix(float **input, int rows, int cols) {
    int i = 0;
    int j = 0;

    printf("Input Matrix:\n");
    printf("Rows: %d\n", rows);
    printf("Columns: %d\n", cols);

    for(i = 0; i < rows; i ++) {
        for(j = 0; j < cols; j++) {
            printf("%-15f ", input[i][j]);
        }
        printf("\n");
    }

    printf("\n");
}

void printMatrixWithLimits(float **input, int rows, int cols, int limit) {
    int i = 0;
    int j = 0;

    printf("Matrix (Display Limit = 12 rows/columns):\n");
    printf("Rows: %d\n", rows);
    printf("Columns: %d\n", cols);

    if(rows > limit) {
        rows = limit;
    }

    if(cols > limit) {
        cols = limit;
    }

    for(i = 0; i < rows; i ++) {
        for(j = 0; j < cols; j++) {
            printf("%-11f ", input[i][j]);
        }
        printf("\n");
    }

    printf("\n");
}

void printCublasMatrixArray(float * input, int length) {
    int i;
    int counter = 0;
    printf("CUBLAS Matrix Array (Column Major Storage): \n");   
    
    for(i = 0; i < length; i++) {
        printf("%-15f ", input[i]);
        counter++;

        if((counter % 7) == 0) {
            printf("\n");
        }
    }

    printf("\n\n");
}

void printCublasMatrixArrayConverted (float* input, int rows, int cols) {
    int i, j;

    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            printf("%f ", input[IDX2C(i,j,rows)]);
        }

        printf("\n");
    }

    printf("\n\n");
}