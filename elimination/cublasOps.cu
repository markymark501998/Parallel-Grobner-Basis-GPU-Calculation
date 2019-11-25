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

//Test Code
//==================================================================================================================================
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
int GuassianEliminationV1 (float** inputMatrix, int rows, int cols, int dontPrint) {
    float *hostMatrix = 0;
	float *deviceMatrix = 0;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

    int i, j, k;

    //Initialize Array for the Host Matrix
    hostMatrix = (float *)malloc(rows * cols * sizeof(float));
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


    //This is all test code
    //==================================================================================================================================
    //If you are gonna use this command, uncomment this function "modify" at the top of this file, else it won't work
    //modify (handle, deviceMatrix, rows, cols, 1, 2, 16.0f, 12.0f);    



    //float scalar = 10.0f;
    //cublasSscal (handle, cols, &scalar, &deviceMatrix[IDX2C(1,0,rows)], rows);
    //test_inline_function(handle, deviceMatrix, rows, scalar);

    //cublasSswap(handle, cols, &deviceMatrix[IDX2C(1,0,rows)], rows, &deviceMatrix[IDX2C(2,0,rows)], rows);    
    
    //GuassianEliminationDeviceFunction(handle, deviceMatrix, rows, cols);

    //cublasSaxpy(handle, cols, &scalar, &deviceMatrix[IDX2C(1,0,rows)], rows, &deviceMatrix[IDX2C(2,0,rows)], rows);
    //==================================================================================================================================
    
    int rank = -1;
    float scalar = 0.0f;
    float *Aji;
    float *Aki;
    float *Aranki;

    //The FGL Paper says to start this at 1 and I don't know if that is correct. The first column starts at index "0"
    for(i = 0; i < cols; i++) {
    //for(i = 1; i < cols; i++) {
        int piv_found = 0;
        
        for(j = (rank + 1); j < rows; j++) {            
        //for(j = (rank); j < rows; j++) {            
            //Download A[j,i]
            Aji = (float *) malloc (sizeof(float));
            cudaMemcpy(Aji, &deviceMatrix[IDX2C(j,i,rows)], sizeof(float), cudaMemcpyDeviceToHost);
            
            if(dontPrint == 0)
                printf("Aji: %f\n", *Aji);
            
            if(*Aji != 0.0f) {
                rank++;    

                stat = cublasSswap(handle, cols, &deviceMatrix[IDX2C(rank,0,rows)], rows, &deviceMatrix[IDX2C(j,0,rows)], rows);
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

                //This break was never in the FGL algorithm. I added it because it just was not working before (still logically not 
                //working but hey its a work in progress).
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

    cudaFree (deviceMatrix);
    cublasDestroy(handle);

    //Bring Data back to Main function through inputMatrix variable
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            inputMatrix[i][j] = hostMatrix[IDX2C(i,j,rows)];
            
            //Get rid of -0.0 's, they look weird in the console
            if(inputMatrix[i][j] == -0.000000f) {
                inputMatrix[i][j] = 0.0f;
            }
        }
    }

    free(hostMatrix);

    return 0;
}

extern "C"
int GuassianEliminationV1Rref (float** inputMatrix, int rows, int cols, int dontPrint) {
    float *hostMatrix = 0;
	float *deviceMatrix = 0;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

    int i, j, k;

    //Initialize Array for the Host Matrix
    hostMatrix = (float *)malloc(rows * cols * sizeof(float));
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
    
    int rank = -1;
    float scalar = 0.0f;
    float *Aji;
    float *Aki;
    float *Aranki;

    //The FGL Paper says to start this at 1 and I don't know if that is correct. The first column starts at index "0"
    for(i = 0; i < cols; i++) {
        int piv_found = 0;
        
        for(j = (rank + 1); j < rows; j++) {         
            //Download A[j,i]
            Aji = (float *) malloc (sizeof(float));
            cudaMemcpy(Aji, &deviceMatrix[IDX2C(j,i,rows)], sizeof(float), cudaMemcpyDeviceToHost);
            
            if(dontPrint == 0)
                printf("Aji: %f\n", *Aji);
            
            if(*Aji != 0.0f) {
                rank++;    

                stat = cublasSswap(handle, cols, &deviceMatrix[IDX2C(rank,0,rows)], rows, &deviceMatrix[IDX2C(j,0,rows)], rows);
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

                //This break was never in the FGL algorithm. I added it because it just was not working before (still logically not 
                //working but hey its a work in progress).
                break;
            }
        }     

        if(piv_found == 1) {
            //for(k = (rank + 1); k < rows; k++) { 
            for(k = 0; k < rows; k++) { 
                if(k != rank) {
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

    cudaFree (deviceMatrix);
    cublasDestroy(handle);

    //Bring Data back to Main function through inputMatrix variable
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            inputMatrix[i][j] = hostMatrix[IDX2C(i,j,rows)];
            
            //Get rid of -0.0 's, they look weird in the console
            if(inputMatrix[i][j] == -0.000000f) {
                inputMatrix[i][j] = 0.0f;
            }
        }
    }

    free(hostMatrix);

    return 0;
}

extern "C"
int FGL_Algorithm (float** inputMatrix, int rows, int cols, int dontPrint) {
    float *hostMatrix = 0;
	float *deviceMatrix = 0;
	cudaError_t cudaStat;
	cublasStatus_t stat;
    cublasHandle_t handle;
    
    int* cPiv;
    int* rPiv;

    int i, j;
    int rowColMax = new_max(rows, cols);
    printf("RowColMax: %d\n", rowColMax);
    int nPiv;

    //Initialize Array for the Host Matrix
    hostMatrix = (float *)malloc(rows * cols * sizeof(float));
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

    //FLG Analysis Phase (Algorithm 1.5 in the FGL paper)
    cPiv = (int *)malloc(rowColMax * sizeof(int));
    rPiv = (int *)malloc(rowColMax * sizeof(int));
    nPiv = -1;

    for (i = 0; i < cols; i++) {
        for (j = 0; j < rows; j++) {
            if (hostMatrix[IDX2C(j,i,rows)] != 0) {
                nPiv++;

                if(nPiv < rowColMax) {
                    cPiv[nPiv] = i;
                    rPiv[nPiv] = j;
                }                
                
                break;
            }
        }
    }

    if(dontPrint == 0) {
        printf("cPiv:\n");
        printStandardIntArray(cPiv, nPiv);
        printf("rPiv:\n");
        printStandardIntArray(rPiv, nPiv);
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

    //FGL Here
    //
     
    
    


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

    cudaFree (deviceMatrix);
    cublasDestroy(handle);

    //Bring Data back to Main function through inputMatrix variable
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            inputMatrix[i][j] = hostMatrix[IDX2C(i,j,rows)];
            
            //Get rid of -0.0 's, they look weird in the console
            if(inputMatrix[i][j] == -0.000000f) {
                inputMatrix[i][j] = 0.0f;
            }
        }
    }

    free(hostMatrix);

    return 0;
}