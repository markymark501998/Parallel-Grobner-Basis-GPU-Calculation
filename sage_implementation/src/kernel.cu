#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void __global__ kernel_add_one(int* a, int length) {
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    while(gid < length) {
    	a[gid] += 1;
        gid += blockDim.x*gridDim.x;
    }
}

int Inverse_Integer_Mod (int x, int n) {
    int tmp = 0;
    int a = 0;
    int b = 0;
    int last_t = 0;
    int t = 0;
    int next_t = 0;
    int q = 0;

    if (n == 1) {
        return 0;
    }

    a = n;
    b = x;
    t = 0;
    next_t = 1;

    while (b != 0) {
        if (b == 1) {
            next_t = next_t % n;
            if (next_t < 0) {
                next_t = next_t + n;
            }
            return next_t;
        }

        q = a / b;
        tmp = b;
        b = a % b;
        a = tmp;
        last_t = t;
        t = next_t;
        next_t = last_t - q * t;
    }

    return 0;
}

int __device__ Add_Integer_Mod (int n, int m, int p) {
    return (n + m) % p;
}

int __device__ Mul_Integer_Mod (int n, int m, int p) {
    return (n * m) % p;
}

//Call: intModDAXPY <<<(N+255)/256, 256>>> (cols, scalar, devicematrix*, devicematrix*, rows, field_size);
void __global__ intModDAXPY (int n, int scalar, double* x, double* y, int inc, int p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < n) {
        int j = i * inc;

        int aX = Mul_Integer_Mod(scalar, (int)x[j], p);
        y[j] = (double)Add_Integer_Mod(aX, (int)y[j], p);
    }
}

//Call: intModDScale <<<(N+255)/256, 256>>> (cols, scalar, devicematrix*, rows, field_size)
void __global__ intModDScale (int n, int scalar, double* x, int inc, int p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < n) {
        int j = i * inc;

        x[j] = (double)Mul_Integer_Mod(scalar, x[j], p);
    }
}

int F4_5_GuassianElimination_Finite (double * inputMatrix, int rows, int cols, int dontPrint, int checkRef, int field_size) {
    double *hostMatrix = 0;
    double *deviceMatrix = 0;
    cudaError_t cudaStat;
	cublasStatus_t stat;
    cublasHandle_t handle;

    int* rPiv;
    int i, j, c, r;

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

    //Initialize Array for the Host Matrix
    hostMatrix = (double *)malloc(rows * cols * sizeof(double));
    if (!hostMatrix) {
        printf ("host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /*
    //Populate the Host Matrix Array in CuBLAS format
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            hostMatrix[IDX2C(i,j,rows)] = inputMatrix[i][j];
        }
    }
    */

    //hostMatrix = inputMatrix;
    
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            hostMatrix[IDX2C(i,j,rows)] = inputMatrix[IDX2C(i,j,rows)];
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
    
    double *tempVector = (double *)malloc(rows* sizeof(double));
    double *tempAxpyScal = (double *) malloc (sizeof(double));
    double *inverseRounder = (double *)malloc(sizeof(double));
    int int_scalar = 0;
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

                int_scalar = Inverse_Integer_Mod((int)tempVector[r], field_size);
                //intModDScale <<<(N+255)/256, 256>>> (cols, scalar, devicematrix*, rows, field_size)
                intModDScale<<<(cols+255)/256, 256>>>(cols, int_scalar, &deviceMatrix[IDX2C(r,0,rows)], rows, field_size);
                cudaDeviceSynchronize();

                for (i = r + 1; i < rows; i++) {
                    if (tempVector[i] != 0.0f) {
                        int_scalar = field_size - (int)tempVector[i];

                        //intModDAXPY <<<(N+255)/256, 256>>> (cols, scalar, devicematrix*, devicematrix*, rows, field_size);
                        intModDAXPY<<<(cols+255)/256, 256>>>(cols, int_scalar, &deviceMatrix[IDX2C(r,0,rows)], &deviceMatrix[IDX2C(i,0,rows)], rows, field_size);
                        cudaDeviceSynchronize();
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

    //inputMatrix = hostMatrix;;
    
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            inputMatrix[IDX2C(i,j,rows)] = hostMatrix[IDX2C(i,j,rows)];
        }
    }

    /*
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
    */

    //Free all the memory used
    cudaFree (deviceMatrix);
    cublasDestroy(handle);

    free(hostMatrix);
    free(rPiv);
    free(inverseRounder);
    free(tempVector);

    return 0;
}

int F4_5_GuassianElimination (double * inputMatrix, int rows, int cols, int dontPrint, int checkRef) {
    double *hostMatrix = 0;
    double *deviceMatrix = 0;
    cudaError_t cudaStat;
	cublasStatus_t stat;
    cublasHandle_t handle;

    int* rPiv;
    int i, j, c, r;

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

    //Initialize Array for the Host Matrix
    hostMatrix = (double *)malloc(rows * cols * sizeof(double));
    if (!hostMatrix) {
        printf ("host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /*
    //Populate the Host Matrix Array in CuBLAS format
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            hostMatrix[IDX2C(i,j,rows)] = inputMatrix[i][j];
        }
    }
    */

    //hostMatrix = inputMatrix;
    
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            hostMatrix[IDX2C(i,j,rows)] = inputMatrix[IDX2C(i,j,rows)];
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

    //inputMatrix = hostMatrix;;
    
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            inputMatrix[IDX2C(i,j,rows)] = hostMatrix[IDX2C(i,j,rows)];
        }
    }

    /*
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
    */

    //Free all the memory used
    cudaFree (deviceMatrix);
    cublasDestroy(handle);

    free(hostMatrix);
    free(rPiv);
    free(inverseRounder);
    free(tempVector);

    return 0;
}