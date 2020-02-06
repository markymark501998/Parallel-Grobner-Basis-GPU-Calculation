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
int F5_GuassianElimination (double ** inputMatrix, int rows, int cols, int dontPrint, int checkRef) {
    double *hostMatrix = 0;
    double *deviceMatrix = 0;
    cudaError_t cudaStat;
	cublasStatus_t stat;
    cublasHandle_t handle;

    int* cPiv;
    int* rPiv;
    int* chosenPivots;
    int* aColPivLocations;

    int i, j, k, c, r;
    int nPiv = 0;

    cPiv = (int *)malloc(cols * sizeof(int));
    rPiv = (int *)malloc(cols * sizeof(int));
    aColPivLocations = (int *)malloc(rows * sizeof(int));
    chosenPivots = (int *)malloc(rows * sizeof(int)); 

    if(dontPrint == 0) {
        printf("F4-5 Guassian Elimination\n");
        printf("====================================================================================================================================\n");
        printf("                                              Begin Analysis and Construct ABCD Matrix\n");
        printf("====================================================================================================================================\n");
    }

    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //REDO THE ANALYSIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
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

                //Instead of breaking here...set a flag and check for future pivots
                break;
            }
        }
    }

    if(dontPrint == 0) {
        printf("cPiv:\n");
        printStandardIntArray(cPiv, cols);
        printf("rPiv:\n");
        printStandardIntArray(rPiv, cols);
        printf("nPiv: %d\n\n\n", nPiv);
        printf("Chosen Pivots: \n");
        printStandardIntArray(chosenPivots, rows);
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

    for (c = 0; c < cols; c++) {
        for (r = 0; r < rows; r++) {

        }
    }
    

    //Free all the memory used
    cudaFree (deviceMatrix);
    cublasDestroy(handle);

    free(hostMatrix);
    free(cPiv);
    free(rPiv);
    free(chosenPivots);
    free(aColPivLocations);

    return 0;
}

extern "C"
int FGL_Algorithm_Double_NewScalingMethod (double** inputMatrix, int rows, int cols, int dontPrint, int roundFactor, int checkRref) {
    double *hostMatrix = 0;
	double *deviceMatrix = 0;
	cudaError_t cudaStat;
	cublasStatus_t stat;
    cublasHandle_t handle;
    
    int* cPiv;
    int* rPiv;
    int* chosenPivots;

    double** abcdMatrixWhole;
    int* aColPivLocations;

    //double* testVectorHost;
    //double* testVectorDevice;

    int i, j, k;
    //int rowColMax = new_max(rows, cols);
    int nPiv = 0;   

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
        printf("                                              Begin Analysis and Construct ABCD Matrix\n");
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
    abcdMatrixWhole = (double**) malloc (rows * sizeof(double*));

    for(i = 0; i < rows; i++) {
        abcdMatrixWhole[i] = (double *) malloc (cols * sizeof(double));        
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

    for(i = 0; i < rows; i++) {
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
        printMatrixWithLimitsDouble(abcdMatrixWhole, rows, cols, 16);
        printSparseMatrixArrayDouble(abcdMatrixWhole, rows, cols, 160);
        printf("A Matrix - Column Pivot Locations: \n");
        printStandardIntArray(aColPivLocations, rows);	
        printf("====================================================================================================================================\n");
        printf("                                           Reduce A & B by all of A's leading terms (AXPY operations)\n");
        printf("====================================================================================================================================\n");
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
    double scalar = 0.0f;
    for(i = (nPiv - 1); i >= 0; i--) {
        scalar = abcdMatrixWhole[i][(aColPivLocations[i])];    
        scalar = powf(scalar, -1);  

        stat = cublasDscal (handle, cols, &scalar, &deviceMatrix[IDX2C(i,0,rows)], rows);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("Device operation failed (row scalar * inverse of leading term)\n");
            return EXIT_FAILURE;
        }

        //Because floats/doubles suck with inverses it will work to a certain degree of error to just set the pivot value to 1 so there are not cascading errors
        double *inverseRounder = (double *)malloc(sizeof(double));
        *inverseRounder = 1;
        cudaMemcpy(&deviceMatrix[IDX2C(i,aColPivLocations[i],rows)], inverseRounder, sizeof(double), cudaMemcpyHostToDevice);
    }

    //Reduce submatrix A by itself starting from the bottom up
    double *tempAxpyScal = (double *) malloc (sizeof(double));
    for(i = (nPiv - 1); i > 0; i--) {
        double *tempVector = (double *)malloc(rows* sizeof(double));

        stat = cublasGetVector(rows, sizeof(double), &deviceMatrix[IDX2C(0,(aColPivLocations[i]),rows)], 1, tempVector, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("Data Vector download failed");
            cudaFree (deviceMatrix);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }

        for(j = i - 1; j >= 0; j--) {
            if(tempVector[j] != 0.0f) {
                *tempAxpyScal = -(tempVector[j]);

                if(i == 14) {
                    stat = cublasGetMatrix (rows, cols, sizeof(*hostMatrix), deviceMatrix, rows, hostMatrix, rows);
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        printf ("Data upload failed");
                        cudaFree (deviceMatrix);
                        cublasDestroy(handle);
                        return EXIT_FAILURE;
                    } 

                    printf("#==================================\n");
                    printf("tempAxpyScal: %lf, i: %d, jth row val: %lf\n", *tempAxpyScal, i, hostMatrix[IDX2C(j,26,rows)]);
                    printf("#==================================\n");
                }

                stat = cublasDaxpy(handle, cols, tempAxpyScal, &deviceMatrix[IDX2C(i,0,rows)], rows, &deviceMatrix[IDX2C(j,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (axpy)\n");
                    return EXIT_FAILURE;
                }
            }
        }

        free(tempVector);
    }

    if(dontPrint == 0) {
        printf("A Matrix - Column Pivot Locations: \n");
        printStandardIntArray(aColPivLocations, rows);	
        printf("====================================================================================================================================\n");
        printf("                                           Reduce C & D subsequently by all of A's rows (AXPY operations)\n");
        printf("====================================================================================================================================\n");
    }

    for (i = 0; i < nPiv; i++) {
        double *tempVector = (double *)malloc(rows * sizeof(double));

        stat = cublasGetVector(rows, sizeof(double), &deviceMatrix[IDX2C(0,(aColPivLocations[i]),rows)], 1, tempVector, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("Data Vector download failed");
            cudaFree (deviceMatrix);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }

        for (j = nPiv; j < rows; j++) {
        //for (j = (i+1); j < rows; j++) {
            if(tempVector[j] != 0.0f) {
                *tempAxpyScal = -(tempVector[j]);

                stat = cublasDaxpy(handle, cols, tempAxpyScal, &deviceMatrix[IDX2C(i,0,rows)], rows, &deviceMatrix[IDX2C(j,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (axpy)\n");
                    return EXIT_FAILURE;
                }
            }
        }

        free(tempVector);
    }

    free(tempAxpyScal);

    int doneFlag = 0;
    while(doneFlag == 0) {
        if(dontPrint == 0) {
            printf("====================================================================================================================================\n");
            printf("                                                Iterate through algorithm until matrix is fully reduced\n");
            printf("====================================================================================================================================\n");
        }
        
        //Download Matrix from the Device -> Host
        stat = cublasGetMatrix (rows, cols, sizeof(*hostMatrix), deviceMatrix, rows, hostMatrix, rows);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("Data upload failed");
            cudaFree (deviceMatrix);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }

        for(i = 0; i < rows; i++) {
            for(j = 0; j < cols; j++) {
                inputMatrix[i][j] = hostMatrix[IDX2C(i,j,rows)];
            }
        }

        for (i = 0; i < cols; i++) {
            cPiv[i] = -1;
            rPiv[i] = -1;
        }

        for (k = 0; k < rows; k++) {
            chosenPivots[k] = 0;
            aColPivLocations[k] = -1;
        }
        
        int oldNPiv = nPiv;
        nPiv = 0;
        //Set this flag preemptively to see if non-pivot rows exist
        doneFlag = 1;

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
                    } else {
                        doneFlag = 0;
                    }

                    break;
                }
            }
        }

        if(doneFlag == 1 || oldNPiv == nPiv) {
            printf("======================================================================\nDO FINAL ITERATION\n======================================================================\n");
        }

        if(dontPrint == 0) {
            printf("====================================================================================================================================\n");
            printf("                                              Begin Analysis and Construct ABCD Matrix\n");
            printf("====================================================================================================================================\n");
            printf("cPiv:\n");
            printStandardIntArray(cPiv, cols);
            printf("rPiv:\n");
            printStandardIntArray(rPiv, cols);
            printf("nPiv: %d\n\n\n", nPiv);
            printf("Chosen Pivots: \n");
            printStandardIntArray(chosenPivots, rows);
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

        for(i = 0; i < rows; i++) {
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
            printf("A Matrix Column Pivot Locations\n");
            printStandardIntArray(aColPivLocations, rows);
            //printSparseMatrixArray(abcdMatrixWhole, rows, cols, 160);
            printSparseMatrixArrayConvertedDouble(hostMatrix, rows, cols, 160);
            printf("ABCD Matrix: \n");
            printSparseMatrixArrayDouble(abcdMatrixWhole, rows, cols, 160);
        }
        
        //====================================================================================================================================
        //                                           Reduce A & B by all of A's leading terms (AXPY operations)
        //====================================================================================================================================

        //Populate the Host Matrix Array in CuBLAS format
        for(j = 0; j < cols; j++) {
            for(i = 0; i < rows; i++) {
                hostMatrix[IDX2C(i,j,rows)] = abcdMatrixWhole[i][j];
            }
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
        double scalar = 0.0f;
        for(i = (nPiv - 1); i >= 0; i--) {
            scalar = abcdMatrixWhole[i][(aColPivLocations[i])];    
            scalar = powf(scalar, -1);    
            
            if(scalar != 1.0f) {
                stat = cublasDscal (handle, cols, &scalar, &deviceMatrix[IDX2C(i,0,rows)], rows);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    printf ("Device operation failed (row scalar * inverse of leading term)\n");
                    return EXIT_FAILURE;
                }
        
                //Because floats suck with inverses it will work to a certain degree of error to just set the pivot value to 1 so there are not cascading errors
                double *inverseRounder = (double *)malloc(sizeof(double));
                *inverseRounder = 1;
                cudaMemcpy(&deviceMatrix[IDX2C(i,aColPivLocations[i],rows)], inverseRounder, sizeof(double), cudaMemcpyHostToDevice);
            }            
        }

        if(dontPrint == 0) {
            //Download Matrix from the Device -> Host
            stat = cublasGetMatrix (rows, cols, sizeof(*hostMatrix), deviceMatrix, rows, hostMatrix, rows);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("Data upload failed");
                cudaFree (deviceMatrix);
                cublasDestroy(handle);
                return EXIT_FAILURE;
            }

            printf("Post Scaling - Resultant Matrix:\n");
            printSparseMatrixArrayConvertedDouble(hostMatrix, rows, cols, 160);
        }
    
        //Reduce submatrix A by itself starting from the bottom up
        double *tempAxpyScal = (double *) malloc (sizeof(double));
        for(i = (nPiv - 1); i > 0; i--) {
            double *tempVector = (double *)malloc(rows * sizeof(double));
    
            stat = cublasGetVector(rows, sizeof(double), &deviceMatrix[IDX2C(0,(aColPivLocations[i]),rows)], 1, tempVector, 1);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("Data Vector download failed");
                cudaFree (deviceMatrix);
                cublasDestroy(handle);
                return EXIT_FAILURE;
            }  
            
            for(j = i - 1; j >= 0; j--) {
                if(tempVector[j] != 0.0f) {
                    *tempAxpyScal = -(tempVector[j]);
    
                    stat = cublasDaxpy(handle, cols, tempAxpyScal, &deviceMatrix[IDX2C(i,0,rows)], rows, &deviceMatrix[IDX2C(j,0,rows)], rows);
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        printf ("Device operation failed (axpy)\n");
                        return EXIT_FAILURE;
                    }
                }
            }
    
            free(tempVector);
        }

        if(dontPrint == 0) {
            //Download Matrix from the Device -> Host
            stat = cublasGetMatrix (rows, cols, sizeof(*hostMatrix), deviceMatrix, rows, hostMatrix, rows);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("Data upload failed");
                cudaFree (deviceMatrix);
                cublasDestroy(handle);
                return EXIT_FAILURE;
            }

            printf("Post A Submatrix Reduction - Resultant Matrix:\n");
            printSparseMatrixArrayConvertedDouble(hostMatrix, rows, cols, 160);
        }

        for (i = 0; i < nPiv; i++) {
            double *tempVector = (double *)malloc(rows * sizeof(double));
    
            stat = cublasGetVector(rows, sizeof(double), &deviceMatrix[IDX2C(0,(aColPivLocations[i]),rows)], 1, tempVector, 1);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("Data Vector download failed");
                cudaFree (deviceMatrix);
                cublasDestroy(handle);
                return EXIT_FAILURE;
            }
    
            for (j = nPiv; j < rows; j++) {
            //for (j = (i+1); j < rows; j++) {
                if(tempVector[j] != 0.0f) {
                    *tempAxpyScal = -(tempVector[j]);
    
                    stat = cublasDaxpy(handle, cols, tempAxpyScal, &deviceMatrix[IDX2C(i,0,rows)], rows, &deviceMatrix[IDX2C(j,0,rows)], rows);
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        printf ("Device operation failed (axpy)\n");
                        return EXIT_FAILURE;
                    }
                }
            }
    
            free(tempVector);
        }
    
        free(tempAxpyScal);

        if(doneFlag == 1 || oldNPiv == nPiv) {
        //if(doneFlag == 1) {
            break;
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

    //Bring Data back to Main function through inputMatrix variable
    for(j = 0; j < cols; j++) {
        for(i = 0; i < rows; i++) {
            inputMatrix[i][j] = hostMatrix[IDX2C(i,j,rows)];
            
            /*
            //Get rid of -0.0 's, they look weird in the console
            if(inputMatrix[i][j] == -0.000000f) {
                inputMatrix[i][j] = 0.0f;
            }
            */
        }
    }

    if(checkRref == 1) {
        printf("=================================================================\n");
        printf("                        CHECKING RREF FORM\n");
        printf("=================================================================\n");
        printf("Result: ");

        int isRref = 0;

        for(i = 0; i < rows; i++) {
            if(isRref > 100) {
                break;
            }

            for(j = 0; j < cols; j++) {
                if(inputMatrix[i][j] != 0) {
                    if(isRref > 100) {
                        break;
                    }
                    
                    for(k = 0; k < rows; k++) {
                        if(inputMatrix[k][j] != 0 && k != i) {
                            isRref += 1;
                            printf("Fail Coordinates: [%d][%d] Value: %f\n", k, j, inputMatrix[k][j]);

                            if(isRref > 100) {
                                break;
                            }
                        }
                    }

                    break;
                }
            }
        }

        if(isRref == 0) {
            printf("Success!\n\n\n");
        } else {
            printf("Fail\n\n\n");
        }
    }

    //Free all the memory used
    cudaFree (deviceMatrix);
    cublasDestroy(handle);

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