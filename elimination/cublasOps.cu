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
int GuassianEliminationV1Rref (float** inputMatrix, int rows, int cols, int dontPrint, int roundFactor) {
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
            int floatFactor = pow(10, roundFactor);            
            hostMatrix[IDX2C(i,j,rows)] = roundf(inputMatrix[i][j] * floatFactor) / floatFactor;
            //hostMatrix[IDX2C(i,j,rows)] = inputMatrix[i][j];
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
                //scalar = 1.0f / scalar;
                scalar = powf(scalar, -1);



                //printf("New Rank: %.50f\n", (scalar * (*Aranki)));

                

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
                    //Download A[k,i]
                    Aki = (float *) malloc (sizeof(float));
                    cudaMemcpy(Aki, &deviceMatrix[IDX2C(k,i,rows)], sizeof(float), cudaMemcpyDeviceToHost);
                    *Aki = (*Aki) * -1.0f;



                    float * tempRank = (float *) malloc (sizeof(float));
                    cudaMemcpy(tempRank, &deviceMatrix[IDX2C(rank,i,rows)], sizeof(float), cudaMemcpyDeviceToHost);
                    float temp1 = *Aki;



                    if(*Aki != 0.0f) {
                        stat = cublasSaxpy(handle, cols, Aki, &deviceMatrix[IDX2C(rank,0,rows)], rows, &deviceMatrix[IDX2C(k,0,rows)], rows);
                        if (stat != CUBLAS_STATUS_SUCCESS) {
                            printf ("Device operation failed (axpy)\n");
                            return EXIT_FAILURE;
                        }
                    }



                    cudaMemcpy(Aki, &deviceMatrix[IDX2C(k,i,rows)], sizeof(float), cudaMemcpyDeviceToHost);
                    float temp2 = *Aki;
                    if(temp2 != 0.0f) {
                        //printf("Before/After Aki: %.40f  /  %.40f | rank: %.40f\n", temp1, temp2, *tempRank);
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

            int floatFactor = pow(10, roundFactor);
            inputMatrix[i][j] = roundf(inputMatrix[i][j] * floatFactor) / floatFactor;
        }
    }

    free(hostMatrix);

    return 0;
}

extern "C"
int FGL_Algorithm (float** inputMatrix, int rows, int cols, int dontPrint, int roundFactor) {
    float *hostMatrix = 0;
	float *deviceMatrix = 0;
	cudaError_t cudaStat;
	cublasStatus_t stat;
    cublasHandle_t handle;
    
    int* cPiv;
    int* rPiv;
    int* chosenPivots;

    float** abcdMatrixWhole;
    int* aColPivLocations;

    //float* testVectorHost;
    //float* testVectorDevice;

    int i, j, k;
    //int rowColMax = new_max(rows, cols);
    int nPiv = 0;

    //Hojnacki FGL
    //=============================================================================================================
    //FGL Analysis Phase (Algorithm 1.5 in the FGL paper)
    cPiv = (int *)malloc(cols * sizeof(int));
    rPiv = (int *)malloc(cols * sizeof(int));
    aColPivLocations = (int *)malloc(rows * sizeof(int));
    chosenPivots = (int *)malloc(rows * sizeof(int));

    for (i = 0; i < cols; i++) {
        cPiv[i] = -1;
        rPiv[i] = -1;
    }

    for (k = 0; k < rows; k++) {
        chosenPivots[k] = 0;
        aColPivLocations[k] = -1;
    }

    //Identify the pivots in the matrix
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if(inputMatrix[i][j] != 0.0) {
                if(cPiv[j] == -1) {
                    //No pivot yet since the entry is -1, make this entry a pivot col/row

                    cPiv[j] = j;
                    rPiv[j] = i;
                    chosenPivots[i] = 1;

                    nPiv += 1;
                }

                break;
            }
        }
    }

    

    if(dontPrint == 0) {
        printf("====================================================================================================================================\n");
        printf("                                              Begin Analysis and Reconstruct ABCD Matrix\n");
        printf("====================================================================================================================================\n");
        printf("cPiv:\n");
        printStandardIntArray(cPiv, cols);
        printf("rPiv:\n");
        printStandardIntArray(rPiv, cols);
        printf("nPiv: %d\n\n\n", nPiv);
        printf("Chosen Pivots: \n");
        printStandardIntArray(chosenPivots, rows);
    }

    //ABCD Matrix Initialization
    abcdMatrixWhole = (float**) malloc (rows * sizeof(float*));

    for(i = 0; i < rows; i++) {
        abcdMatrixWhole[i] = (float *) malloc (cols * sizeof(float));        
    }

    int counter = 0;

    //ABCD Matrix Construction

    //  A B
    //  C D

    for(i = 0; i < cols; i++) {
        if(rPiv[i] != -1) {
            aColPivLocations[counter] = cPiv[i];
            for(j = 0; j < cols; j++) {
                abcdMatrixWhole[counter][j] = inputMatrix[(rPiv[i])][j];
            }

            counter++;
        } 
    }

    for(i = 0; i < cols; i++) {
        if(counter >= rows) {
            break;
        }
        
        if(chosenPivots[i] == 0) {
            for(j = 0; j < cols; j++) {
                abcdMatrixWhole[counter][j] = inputMatrix[i][j];
            }

            counter++;
        }
    }

    if(dontPrint == 0) {
        printMatrixWithLimits(abcdMatrixWhole, rows, cols, 16);
        printSparseMatrixArray(abcdMatrixWhole, rows, cols, 160);
        printf("A Column Pivot Locations: \n");
        printStandardIntArray(aColPivLocations, rows);	
        printf("====================================================================================================================================\n");
        printf("                                                     Reduce A (read only) and do AXPY on B\n");
        printf("====================================================================================================================================\n");
    }

    //Initialize Array for the Host Matrix
    hostMatrix = (float *)malloc(rows * cols * sizeof(float));
    if (!hostMatrix) {
        printf ("host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    //Populate the Host Matrix Array in CuBLAS format
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            hostMatrix[IDX2C(i,j,rows)] = abcdMatrixWhole[i][j];
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

    //Scale every pivot row so each leading term in submatrix A has a leading term of 1
    float scalar = 0.0f;
    for(i = (nPiv - 1); i >= 0; i--) {
        scalar = abcdMatrixWhole[i][(aColPivLocations[i])];    
        scalar = powf(scalar, -1);    
        

        stat = cublasSscal (handle, cols, &scalar, &deviceMatrix[IDX2C(i,0,rows)], rows);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("Device operation failed (row scalar * inverse of leading term)\n");
            return EXIT_FAILURE;
        }

        //Because floats suck with inverses it will work to a certain degree of error to just set the pivot value to 1
        float *inverseRounder = (float *)malloc(sizeof(float));
        *inverseRounder = 1;
        cudaMemcpy(&deviceMatrix[IDX2C(i,aColPivLocations[i],rows)], inverseRounder, sizeof(float), cudaMemcpyHostToDevice);
    }

    //Reduce submatrix A by itself starting from the bottom up
    float *tempAxpyScal = (float *) malloc (sizeof(float));
    for(i = (nPiv - 1); i > 0; i--) {
        float *tempVector = (float *)malloc(rows * sizeof(float));

        stat = cublasGetVector(rows, sizeof(float), &deviceMatrix[IDX2C(0,(aColPivLocations[i]),rows)], 1, tempVector, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("Data Vector download failed");
            cudaFree (deviceMatrix);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        
        for(j = i - 1; j >= 0; j--) {
            if(tempVector[j] != 0.0f) {
                *tempAxpyScal = -(tempVector[j]);

                stat = cublasSaxpy(handle, cols, tempAxpyScal, &deviceMatrix[IDX2C(i,0,rows)], rows, &deviceMatrix[IDX2C(j,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (axpy)\n");
                    return EXIT_FAILURE;
                }
            }
        }

        free(tempVector);
    }

    free(tempAxpyScal);
    

    //Reduce C by A
    
    /* Here */



    //Analyze Whole Matrix and Reconstruct into ABCD 2.0
    

    

    //Haven't gotten this far yet lol



    

    /*
    if(dontPrint == 0) {
        printf("cPiv:\n");
        printStandardIntArray(cPiv, nPiv + 1);
        printf("rPiv:\n");
        printStandardIntArray(rPiv, nPiv + 1);
    }
    */

    

    //FGL algorithm here
    //





     
    /*
    //Test Vector Download
    testVectorHost = (float *)malloc(rows * sizeof(float));

    //Download Vector from the Device -> Host
    stat = cublasGetVector(rows, sizeof(float), &deviceMatrix[IDX2C(0,6,rows)], 1, testVectorHost, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Data Vector download failed");
        cudaFree (deviceMatrix);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    for(i = 0; i < rows; i++) {
        printf("Vector[%d]: %f\n", i, testVectorHost[i]);
    }
    */




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
    free(cPiv);
    free(rPiv);
    free(chosenPivots);
    free(aColPivLocations);

    for(i = 0; i < rows; i++) {
        free(abcdMatrixWhole[i]);
    }

    free(abcdMatrixWhole);

    return 0;
}