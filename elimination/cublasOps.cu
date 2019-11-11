#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

extern "C" {
    #include "common.h"
}

/*
static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}
*/

extern "C"
int GuassianEliminationV1 (float** inputMatrix, int rows, int cols) {
    float *hostMatrix = 0;
	float *deviceMatrix = 0;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

    int i, j;

    hostMatrix = (float *)malloc(rows * cols * sizeof(float));
    if (!hostMatrix) {
        printf ("host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            hostMatrix[IDX2C(i,j,rows)] = inputMatrix[i][j];
        }
    }
    
    cudaStat = cudaMalloc ((void**) &deviceMatrix, rows * cols * sizeof(*hostMatrix));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    stat = cublasSetMatrix (rows, cols, sizeof(*hostMatrix), hostMatrix, rows, deviceMatrix, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Data download failed\n");
        cudaFree (deviceMatrix);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }






    //If you are gonna use this command, uncomment this function "modify" at the top of this file, else it won't work
    //modify (handle, deviceMatrix, rows, cols, 1, 2, 16.0f, 12.0f);    

    float scalar = 10.0f;
    cublasSscal (handle, rows, &scalar, &deviceMatrix[IDX2C(0,2,rows)], 1);




    stat = cublasGetMatrix (rows, cols, sizeof(*hostMatrix), deviceMatrix, rows, hostMatrix, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Data upload failed");
        cudaFree (deviceMatrix);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaDeviceSynchronize();

    cudaFree (deviceMatrix);
    cublasDestroy(handle);

    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            inputMatrix[i][j] = hostMatrix[IDX2C(i,j,rows)];
        }
    }

    free(hostMatrix);

    return 0;
}