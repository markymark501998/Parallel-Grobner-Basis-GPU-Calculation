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


static __inline__ void test_inline_function (cublasHandle_t handle, float *m, int rows, float scalar){
    cublasSscal (handle, rows, &scalar, &m[IDX2C(0,2,rows)], 1);
}
*/

/*
static __inline__ int GuassianEliminationDeviceFunction(cublasHandle_t handle, float *deviceMatrix, int rows, int cols) {
    cublasStatus_t stat;
    int i, j;
    int rank = 0;
    float scalar = 0.0f;

    for(i = 0; i < cols; i++) {
        int piv_found = 0;
        int rankPlusOne = rank + 1; 
        
        for (j = rankPlusOne; j < rows; j++) {
            if(deviceMatrix[IDX2C(j,i,rows)] != 0.0f) {
                rank++;    

                stat = cublasSswap(handle, cols, &deviceMatrix[IDX2C(j,0,rows)], rows, &deviceMatrix[IDX2C(rank,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (vector swap)\n");
                    return EXIT_FAILURE;
                }

                printf("debug: rank[%d] i[%d]\n", rank, i);
                printf("value @ [rank, i]: %f\n", deviceMatrix[IDX2C(rank, i, rows)]);
                scalar = deviceMatrix[IDX2C(rank, i, rows)];
                scalar = powf(scalar, -1);

                stat = cublasSscal (handle, cols, &scalar, &deviceMatrix[IDX2C(rank,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (row scalar x inverse)\n");
                    return EXIT_FAILURE;
                }

                piv_found = 1;
            }
        }

        if(piv_found == 1) {
            
        }
    }

    return 0;
}
*/

extern "C"
int GuassianEliminationV1 (float** inputMatrix, int rows, int cols) {
    float *hostMatrix = 0;
	float *deviceMatrix = 0;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

    int i, j, k;

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



    //float scalar = 10.0f;
    //cublasSscal (handle, cols, &scalar, &deviceMatrix[IDX2C(1,0,rows)], rows);
    //test_inline_function(handle, deviceMatrix, rows, scalar);

    //cublasSswap(handle, cols, &deviceMatrix[IDX2C(1,0,rows)], rows, &deviceMatrix[IDX2C(2,0,rows)], rows);    
    
    //GuassianEliminationDeviceFunction(handle, deviceMatrix, rows, cols);

    int rank = 0;
    float scalar = 0.0f;
    float *Aji;
    float *Aki;
    float *Aranki;

    for(i = 0; i < cols; i++) {
        int piv_found = 0;
        
        for(j = (rank + 1); j < rows; j++) {            
        //for(j = (rank); j < rows; j++) {            
            //Download A[j,i]
            Aji = (float *) malloc (sizeof(float));
            cudaMemcpy(Aji, &deviceMatrix[IDX2C(j,i,rows)], sizeof(float), cudaMemcpyDeviceToHost);
            //printf("Aji: %f\n", *Aji);            
            
            if(*Aji != 0.0f) {
                rank++;    

                stat = cublasSswap(handle, cols, &deviceMatrix[IDX2C(j,0,rows)], rows, &deviceMatrix[IDX2C(rank,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (vector swap)\n");
                    return EXIT_FAILURE;
                }
                
                //Download A[rank, i]
                Aranki = (float *) malloc (sizeof(float));
                cudaMemcpy(Aranki, &deviceMatrix[IDX2C(rank,i,rows)], sizeof(float), cudaMemcpyDeviceToHost);

                scalar = *Aranki;
                scalar = powf(scalar, -1);

                stat = cublasSscal (handle, cols, &scalar, &deviceMatrix[IDX2C(rank,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (row scalar x inverse)\n");
                    return EXIT_FAILURE;
                }

                piv_found = 1;
                free(Aranki);
                free(Aji);
                break;
            }
        }

        /*        
        printf("piv_found: %d\n", piv_found);
        stat = cublasGetMatrix (rows, cols, sizeof(*hostMatrix), deviceMatrix, rows, hostMatrix, rows);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("Data upload failed");
            cudaFree (deviceMatrix);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        printCublasMatrixArrayConverted(hostMatrix, rows, cols);
        */

        if(piv_found == 1) {
            for(k = (rank + 1); k < rows; k++) {   
            //for(k = (rank); k < rows; k++) {   
                //Download A[j,i]
                Aki = (float *) malloc (sizeof(float));
                cudaMemcpy(Aki, &deviceMatrix[IDX2C(k,i,rows)], sizeof(float), cudaMemcpyDeviceToHost);
                *Aki = -*Aki;
                
                stat = cublasSaxpy(handle, cols, Aki, &deviceMatrix[IDX2C(rank,0,rows)], rows, &deviceMatrix[IDX2C(k,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (axpy)\n");
                    return EXIT_FAILURE;
                }

                free(Aki);
            }
        }
    }


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

            if(inputMatrix[i][j] == -0.0f) {
                inputMatrix[i][j] = 0.0f;
            }
        }
    }

    free(hostMatrix);

    return 0;
}