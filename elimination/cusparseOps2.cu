#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX2C_ROW(i,j,ld) (((i)*(ld))+(j))
#define new_max(x,y) ((x) >= (y)) ? (x) : (y)

extern "C" {
    #include "common.h"
}

struct cooInfo {
    int index;
    int rowNNZ;
};

extern "C"
int F4_5_GuassianEliminationCuSparseMHVersion (double ** inputMatrix, int rows, int cols, int dontPrint, int checkRef) {
    double *hostMatrix = 0;
    double *deviceMatrix = 0;
    
    double *cooHostMatrix = 0;
    int *cooRowIndHost = 0;
    int *cooColIndHost = 0;

    double *cooDeviceMatrix = 0;
    int *cooRowIndDevice = 0;
    int *cooColIndDevice = 0;
    
    cudaError_t cudaStat, cudaStat2;
    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    //cusparseMatDescr_t descr = 0;

    cublasHandle_t blas_handle;
	cublasStatus_t blas_stat;

    int* rPiv;
    int* cPiv;
    int i, j, k, c;
    int nnz = 0;

    struct cooInfo* cooInfoTable;
    cooInfoTable = (struct cooInfo *)malloc(rows * sizeof(struct cooInfo));

    rPiv = (int *)malloc(rows * sizeof(int));
    cPiv = (int *)malloc(cols * sizeof(int));

    if(dontPrint == 0) {
        printf("F4-5 Guassian Elimination\n");
        printf("====================================================================================================================================\n");
        printf("                                                        Start Algorithm\n");
        printf("====================================================================================================================================\n");
    }
    
    for (i = 0; i < cols; i++) {
        cPiv[i] = -1;
    }
    
    for (i = 0; i < rows; i++) {
        rPiv[i] = -1;
        cooInfoTable[i].index = -1;
        cooInfoTable[i].rowNNZ = 0;
    }

    /* Analysis */
    for(j = 0; j < cols; j++) {
        for (i = 0; i < rows; i++) {
            if (inputMatrix[i][j] != 0.0) {
                nnz++;
            }
            
            if (inputMatrix[i][j] != 0.0 && rPiv[i] == -1 && cPiv[j] == -1) {
                cPiv[j] = i;
                rPiv[i] = j;
            }
        }
    }
    
    if(dontPrint == 0) {
        printf("rPiv:\n");
        printStandardIntArray(rPiv, cols);
        printf("cPiv:\n");
        printStandardIntArray(cPiv, cols);
    }

    /* initialize cusparse library */
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE Library initialization failed");
        return 1;
    }
    
    //Initialize CuBLAS object
    blas_stat = cublasCreate(&blas_handle);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    //Initialize Array for the Host Matrix
    hostMatrix = (double *)malloc(rows * cols * sizeof(double));
    if (!hostMatrix) {
        printf("host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    //Initialize Array for the Host Matrix (COO Format)
    cooHostMatrix = (double *)malloc(nnz * sizeof(double));
    if (!cooHostMatrix) {
        printf("host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    cooRowIndHost = (int *)malloc(nnz * sizeof(int));
    cooColIndHost = (int *)malloc(nnz * sizeof(int));

    //Populate the Host Matrix Array in CuSPARSE format
    c = 0;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {        
            hostMatrix[IDX2C_ROW(i,j,rows)] = inputMatrix[i][j];

            if(inputMatrix[i][j] != 0.0) {
                cooHostMatrix[c] = inputMatrix[i][j];
                cooRowIndHost[c] = i;
                cooColIndHost[c] = j;

                if (cooInfoTable[i].index == -1) {
                    cooInfoTable[i].index = c;
                }
                cooInfoTable[i].rowNNZ++;

                c++;
            }
        }
    }
    
    for(i = 0; i < rows; i++) {
        printf("[%d, %d]\n", cooInfoTable[i].index, cooInfoTable[i].rowNNZ);
    }
    
    //Allocate memory for Device Matrix Arrays
    cudaStat = cudaMalloc ((void**) &deviceMatrix, (rows * cols * sizeof(*hostMatrix)));    
    cudaStat2 = cudaMalloc ((void**) &cooDeviceMatrix, nnz * sizeof(*cooDeviceMatrix));
    if (cudaStat != cudaSuccess || cudaStat2 != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    //Allocate memory for Device Matrix Array Indice Pointers(COO Format)
    cudaStat = cudaMalloc ((void**) &cooRowIndDevice, nnz * sizeof(int));
    cudaStat2 = cudaMalloc ((void**) &cooColIndDevice, nnz * sizeof(int));
    if (cudaStat != cudaSuccess || cudaStat2 != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    cudaMemcpy(&deviceMatrix[0], &hostMatrix[0], rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&cooDeviceMatrix[0], &cooHostMatrix[0], nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&cooRowIndDevice[0], &cooRowIndHost[0], nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&cooColIndDevice[0], &cooColIndHost[0], nnz * sizeof(double), cudaMemcpyHostToDevice);


    printf("cooHostMatrix:\n");
    printStandardDoubleArray(cooHostMatrix, nnz);
    printf("cooRowIndHost:\n");
    printStandardIntArray(cooRowIndHost, nnz);
    printf("cooRowColHost:\n");
    printStandardIntArray(cooColIndHost, nnz);



    /*
    MORE TESTING
    double scalar = 2.0f;
    blas_stat = cublasDscal(blas_handle, 10, &scalar, &cooDeviceMatrix[4], 1);
    if (blas_stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Device operation failed (row scalar * inverse of leading term)\n");
        return EXIT_FAILURE;
    }  
    */


    //OLD STUFF BELOW
    /*
    status= cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Matrix descriptor initialization failed");
        return 1;
    }
    
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    */


    //ALGORITHM HERE

    for (i = (rows-1); i > 0; i--) {
        int row_nnz = cooInfoTable[i].rowNNZ;
        int ref_index = cooInfoTable[i].index;

        for (j = ref_index; j < (ref_index + row_nnz); j++) {
            //int matrix_i = cooRowIndHost[j];
            //int matrix_j = cooColIndHost[j];

            
        }
    }


    
    cudaMemcpy(&hostMatrix[0], &deviceMatrix[0], rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    
    //Testing Only
    //cudaMemcpy(&cooHostMatrix[0], &cooDeviceMatrix[0], nnz * sizeof(double), cudaMemcpyDeviceToHost);

    printf("cooHostMatrix (AFTER DOWNLOAD):\n");
    printStandardDoubleArray(cooHostMatrix, nnz);

    if (checkRef == 1) {
        printf("Checking if NaN/Infinite rows are present...\n");
    }

    //Bring Data back to Main function through inputMatrix variable
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {        
            inputMatrix[i][j] = hostMatrix[IDX2C_ROW(i,j,rows)];            

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
    cudaFree (cooDeviceMatrix);
    cudaFree (cooColIndDevice);
    cudaFree (deviceMatrix);
    if(handle) { cusparseDestroy(handle); }

    free(hostMatrix);
    free(cooHostMatrix);
    free(cooRowIndHost);
    free(cooColIndHost);
    free(rPiv);

    return 0;
}