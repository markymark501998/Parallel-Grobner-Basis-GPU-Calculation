#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define new_max(x,y) ((x) >= (y)) ? (x) : (y)

extern "C" {
    #include "common.h"
}

extern "C"
int F4_5_GuassianElimination (double ** inputMatrix, int rows, int cols, int dontPrint, int checkRef) {
    double *hostMatrix = 0;
    double *deviceMatrix = 0;
    cudaError_t cudaStat;
	cublasStatus_t stat;
    cublasHandle_t handle;

    int* rPiv;
    int i, j, c, r, k;
    int nPiv = 0;

    rPiv = (int *)malloc(cols * sizeof(int));

    if(dontPrint == 0) {
        printf("F4-5 Guassian Elimination\n");
        printf("====================================================================================================================================\n");
        printf("                                                        Start Algorithm\n");
        printf("====================================================================================================================================\n");
    }
    
    for (i = 0; i < cols; i++) {
        rPiv[i] = -1;
    }
    
    if(dontPrint == 0) {
        printf("rPiv:\n");
        printStandardIntArray(rPiv, cols);
        printf("nPiv: %d\n\n\n", nPiv);
    }

    //Initialize Array for the Host Matrix
    hostMatrix = (double *)malloc(rows * cols * sizeof(double));
    if (!hostMatrix) {
        printf ("host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    //Populate the Host Matrix Array in CuBLAS format
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            hostMatrix[IDX2C(i,j,rows)] = inputMatrix[i][j];
        }
    }
    
    //Allocate memory for Device Matrix Array
    cudaStat = cudaMalloc ((void**) &deviceMatrix, rows * cols * sizeof(*hostMatrix));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    //Initialize CuBLAS object
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    //Push data to the Device
    stat = cublasSetMatrix (rows, cols, sizeof(*hostMatrix), hostMatrix, rows, deviceMatrix, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Data download failed\n");
        cudaFree (deviceMatrix);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //Algorithm Starts Here
    double *tempVector = (double *)malloc(rows* sizeof(double));
    double *tempAxpyScal = (double *) malloc (sizeof(double));
    double *inverseRounder = (double *)malloc(sizeof(double));
    double scalar = 0.0f;
    *inverseRounder = 1;

    for (c = 0; c < cols; c++) {
        //Download the Vector
        stat = cublasGetVector(rows, sizeof(double), &deviceMatrix[IDX2C(0,c,rows)], 1, tempVector, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("Data Vector download failed");
            cudaFree (deviceMatrix);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        
        for (r = 0; r < rows; r++) {
            if(tempVector[r] != 0.0f && rPiv[r] == -1) {
                rPiv[r] = r;

                scalar = tempVector[r];    
                scalar = powf(scalar, -1);  

                stat = cublasDscal (handle, cols, &scalar, &deviceMatrix[IDX2C(r,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (row scalar * inverse of leading term)\n");
                    return EXIT_FAILURE;
                }                
                
                //Copy 1 to location where the LT should be 1 because floats/doubles are not accurate enough
                cudaMemcpy(&deviceMatrix[IDX2C(r,c,rows)], inverseRounder, sizeof(double), cudaMemcpyHostToDevice);

                for (i = r + 1; i < rows; i++) {
                    if (tempVector[i] != 0.0f) {
                        *tempAxpyScal = -(tempVector[i]);
                        stat = cublasDaxpy(handle, cols, tempAxpyScal, &deviceMatrix[IDX2C(r,0,rows)], rows, &deviceMatrix[IDX2C(i,0,rows)], rows);
                        if (stat != CUBLAS_STATUS_SUCCESS) {
                            printf ("Device operation failed (dAxpy)\n");
                            return EXIT_FAILURE;
                        }
                    }
                }

                break;
            }
        }
    }

    //Download Matrix from the Device -> Host
    stat = cublasGetMatrix (rows, cols, sizeof(*hostMatrix), deviceMatrix, rows, hostMatrix, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Data upload failed");
        cudaFree (deviceMatrix);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //Sync up the device (not necessary with CuBLAS but here for good measure)
    cudaDeviceSynchronize();

    if (checkRef == 1) {
        printf("Checking if NaN/Infinite rows are present...\n");
    }

    //Bring Data back to Main function through inputMatrix variable
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {        
            inputMatrix[i][j] = hostMatrix[IDX2C(i,j,rows)];            

            if (checkRef == 1) {
                if(!isfinite(inputMatrix[i][j])) {
                    printf("NaN/Infinite Value Detected\n");
                }
            }
        }
    } 

    if (checkRef == 1) {
        printf("Checking if illogical values are present...\n\n");
        
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                if (inputMatrix[i][j] != 0.0f) {                     
                    for (k = i + 1; k < rows; k++) {
                        if (inputMatrix[k][j] != 0.0f) {
                            printf("Illogical Value Detected\n");
                        }
                    }

                    break;
                }
            }
        }
    }

    //Free all the memory used
    cudaFree (deviceMatrix);
    cublasDestroy(handle);

    free(hostMatrix);
    free(rPiv);
    free(inverseRounder);
    free(tempVector);

    return 0;
}