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

double** parseInputMatrixDouble(FILE *f, int maxLineSize, int* resultRows, int* resultCols) {
    double** resultMatrix;

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

    resultMatrix = (double**) malloc (rows * sizeof(double*));

    for(i = 0; i < rows; i++) {
        resultMatrix[i] = malloc (cols * sizeof(double));        
    } 

    for(i = 0; i < rows; i ++) {
        for(j = 0; j < cols; j++) {
            fscanf(f, "%lf", &resultMatrix[i][j]);
            //printf("#: %lf\n", resultMatrix[i][j]);
        }        
    }

    *resultRows = rows;
    *resultCols = cols;    

    printf("Checkpoint D\n");	

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

    printf("Matrix (Display Limit = %d rows/columns):\n", limit);
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

void printMatrixWithLimitsDouble(double **input, int rows, int cols, int limit) {
    int i = 0;
    int j = 0;

    printf("Matrix (Display Limit = %d rows/columns):\n", limit);
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
            printf("%-11lf ", input[i][j]);
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

void printStandardIntArray(int* input, int length) {
    int i;
    int counter = 0;
    printf("Integer Array: \n");   
    
    for(i = 0; i < length; i++) {
        printf("%-5d ", input[i]);
        counter++;

        if((counter % 10) == 0) {
            printf("\n");
        }
    }

    printf("\n\n");
}

void printCublasMatrixArrayConverted (float* input, int rows, int cols) {
    int i, j;

    printf("Integer Array (converted): \n");
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            printf("%f ", input[IDX2C(i,j,rows)]);
        }

        printf("\n");
    }

    printf("\n\n");
}

void printSparseMatrixArray (float** input, int rows, int cols, int lineLimit) {
    int i, j;
    if(lineLimit == 0) {
        lineLimit = 160;
    }

    printf("Sparse Matrix(lineLimit = %d): \n", lineLimit);
    for(i = 0; i < rows; i++) {
        if(i >= lineLimit) {
            break;
        }

        printf("%.5d ", i);

        for(j = 0; j < cols; j++) {
            if(j >= lineLimit) {
                break;
            } 
            
            /*
            if(input[i][j] == +0.0 || input[i][j] == -0.0) {
                printf("-");
            } else {
                printf("*");
            }
            */

            if(input[i][j] > 0) {
                if(input[i][j] > 0.00001) {
                    printf("*");
                } else {
                    printf("-");
                }
            } else {
                if(input[i][j] < -0.00001) {
                    printf("*");
                } else {
                    printf("-");
                }
            }
        }

        printf("\n");
    }

    printf("\n\n");
}

void printSparseMatrixArrayDouble (double** input, int rows, int cols, int lineLimit) {
    int i, j;
    if(lineLimit == 0) {
        lineLimit = 160;
    }

    printf("Sparse Matrix(lineLimit = %d): \n", lineLimit);
    for(i = 0; i < rows; i++) {
        if(i >= lineLimit) {
            break;
        }

        printf("%.5d ", i);

        for(j = 0; j < cols; j++) {
            if(j >= lineLimit) {
                break;
            } 
            
            /*
            if(input[i][j] == +0.0 || input[i][j] == -0.0) {
                printf("-");
            } else {
                printf("*");
            }
            */

            if(input[i][j] > 0) {
                if(input[i][j] > 0.00001) {
                    printf("*");
                } else {
                    printf("-");
                }
            } else {
                if(input[i][j] < -0.00001) {
                    printf("*");
                } else {
                    printf("-");
                }
            }
        }

        printf("\n");
    }

    printf("\n\n");
}

void printSparseMatrixArrayConverted (float* input, int rows, int cols, int lineLimit) {
    int i, j;
    if(lineLimit == 0) {
        lineLimit = 100;
    }

    printf("Sparse Matrix(converted)(lineLimit = %d): \n", lineLimit);
    for(i = 0; i < rows; i++) {
        if(i >= lineLimit) {
            break;
        }

        printf("%.5d ", i);
        
        for(j = 0; j < cols; j++) {
            if(j >= lineLimit) {
                break;
            }

            if(input[IDX2C(i,j,rows)] > 0) {
                if(input[IDX2C(i,j,rows)] > 0.00001) {
                    printf("*");
                } else {
                    printf("-");
                }
            } else {
                if(input[IDX2C(i,j,rows)] < -0.00001) {
                    printf("*");
                } else {
                    printf("-");
                }
            }
        }

        printf("\n");
    }

    printf("\n\n");
}

void printSparseMatrixArrayConvertedDouble (double* input, int rows, int cols, int lineLimit) {
    int i, j;
    if(lineLimit == 0) {
        lineLimit = 100;
    }

    printf("Sparse Matrix(converted)(lineLimit = %d): \n", lineLimit);
    for(i = 0; i < rows; i++) {
        if(i >= lineLimit) {
            break;
        }

        printf("%.5d ", i);
        
        for(j = 0; j < cols; j++) {
            if(j >= lineLimit) {
                break;
            }

            if(input[IDX2C(i,j,rows)] > 0) {
                if(input[IDX2C(i,j,rows)] > 0.00001) {
                    printf("*");
                } else {
                    printf("-");
                }
            } else {
                if(input[IDX2C(i,j,rows)] < -0.00001) {
                    printf("*");
                } else {
                    printf("-");
                }
            }
        }

        printf("\n");
    }

    printf("\n\n");
}