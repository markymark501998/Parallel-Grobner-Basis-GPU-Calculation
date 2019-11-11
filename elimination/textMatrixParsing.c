#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include "common.h"

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

    printf("Rows: %d\n", rows);
    printf("Columns: %d\n", cols);

    for(i = 0; i < rows; i ++) {
        for(j = 0; j < cols; j++) {
            printf("%-15f ", input[i][j]);
        }
        printf("\n");
    }
}