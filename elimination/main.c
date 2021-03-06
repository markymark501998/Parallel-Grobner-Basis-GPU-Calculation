#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include "common.h"
#include <time.h>
#define MAXCHAR 30000

void CreateExecutionLog (double totalTime, double parsingTime, double deviceTime) {
	FILE *fptr;

	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	char outputFileName[1000];
	char buffer[100];
	strcpy(outputFileName, "ExecutionLogs//ElimData");
	snprintf(buffer, 50, "_%d_%d_%d:%d:%d", tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
	strcat(outputFileName, buffer);
	fptr = fopen(outputFileName, "w");

	fprintf(fptr, "====================================================================================\n");
	fprintf(fptr, "Parsing Execution Time: %f seconds\n", parsingTime);
	fprintf(fptr, "Device (GPU) Execution Time: %f seconds\n\n", deviceTime);
	fprintf(fptr, "Total Execution Time: %f seconds\n", totalTime);
	fprintf(fptr, "====================================================================================\n");

	fclose(fptr);
}

int main(int argc, char* argv[]) {
	printf("\n");
	printf("Senior Design Guassian Elimination Testing\n\nNote: Each line in the input file can only have 100000 characters\n");
	printf("============================================================================================================================================\n");
	printf("Input Matrix file must be in this form:\n\n[rows (int)]\n[columns (int)]\n[float + 'space' + float + 'space' + ...]\n[......]\n\n");
	printf("Note: the Matrix on the host will exist in the 'CUDA' way (just one array) and in float** format\n");
	printf("Of course the final format will not follow this as it 'doubles up' on storage space on the host which is uneccessary [get Mark H. to fix it when this goes to implementation]\n\n");
	printf("Options:\n-printDebug               (doesn't print extra debugging. Large matrices will overflow the console with repeated logging)\n");
	printf("-rref                     (reduced row echelon form)\n");
	printf("-fgl                      (faugere-lachartre guassian elimination)\n");
	printf("-output                   (outputs execution data to the 'ExecutionLogs' directory)\n");
	printf("-round [precision(int)]   (floating point precision rounding factor)\n");
	printf("-sparsePrint              (prints the matrix in a denser form to see structure of matrix)\n");
	printf("-checkRREF                (checks matrix entry by entry to ensure RREF form [WARNING: may cause severe slowdowns])\n");
	printf("-cuSparse                 (executes Michael D.'s version of the cuSPARSE F4-5 Guassian Elimination algorithm)\n");
	printf("-cuSparse2                (executes Mark H.'s version of the cuSPARSE F4-5 Guassian Elimination algorithm)\n\n");

	double time_elapsed = 0.0;
	double time_elapsed_parse = 0.0;
	double time_elapsed_device = 0.0;
	int i, j;
	float** inputMatrix;
	double** inputMatrixDouble;
	int rows = 0;
	int cols = 0;
	int dontPrint = 1;
	int rrefForm = 0;
	int fgl = 0;
	int outputExecutionData = 0;
	int sparsePrint = 0;
	int roundFactor = 6;
	int checkRref = 0;
	int cuSparseFlag = 0;

	printf("------------------------------------------------------------\n");

	for(i = 0; i < argc; i++) {
		if(strcmp(argv[i], "-printDebug") == 0) {
			dontPrint = 0;
		}

		if(strcmp(argv[i], "-rref") == 0) {
			rrefForm = 1;			
		}

		if(strcmp(argv[i], "-fgl") == 0) {
			fgl = 1;			
		}

		if(strcmp(argv[i], "-output") == 0) {
			outputExecutionData = 1;			
		}

		if(strcmp(argv[i], "-sparsePrint") == 0) {
			sparsePrint = 1;			
		}

		if(strcmp(argv[i], "-checkRREF") == 0) {
			checkRref = 1;			
		}

		if(strcmp(argv[i], "-round") == 0) {
			i++;
			roundFactor = atoi(argv[i]);			
		}

		if(strcmp(argv[i], "-cuSparse") == 0) {
			cuSparseFlag = 1;			
		}

		if(strcmp(argv[i], "-cuSparse2") == 0) {
			cuSparseFlag = 2;			
		}
	}
	
	printf("-             [Don't Print Debug Info: %d]\n", dontPrint);
	printf("-             [Floating Point Precision: %d]\n", roundFactor);
	printf("-             [Reduced Row Echelon Form: %d]\n", rrefForm);
	printf("-             [Faugere-Lachartre Guassian Elimination: %d]\n", fgl);
	printf("-             [Output Execution Data -> ExecutionLogs: %d]\n", outputExecutionData);
	printf("-             [Check RREF Form: %d]\n", checkRref);

	printf("------------------------------------------------------------\n\n");
	
	if ( argc <= 1 )
	{
		//Error
		printf("Please enter a filename in the command line: ./main [filename] [other arguments]\n");
	}
	else
	{		
		//FILE *file = fopen( argv[1], "r" );
		FILE *fileDouble = fopen( argv[1], "r" );

		if ( fileDouble == 0 )
		{
			printf( "Could not open the file\n" );
		}
		else
		{
			clock_t begin = clock();
			clock_t beginParse = clock();
			
			//inputMatrix = parseInputMatrix(file, MAXCHAR, &rows, &cols);
			inputMatrixDouble = parseInputMatrixDouble(fileDouble, MAXCHAR, &rows, &cols);
			
			//fclose(file);	
			fclose(fileDouble);	

			clock_t endParse = clock();	
			clock_t beginDeivce = clock(); 

			printf("Input:\n");
			if (sparsePrint == 0) {
				printMatrixWithLimitsDouble(inputMatrixDouble, rows, cols, 16);	
			} else {
				printSparseMatrixArrayDouble(inputMatrixDouble, rows, cols, 160);
			}

			if(rrefForm == 1) {
				//GuassianEliminationV1Rref(inputMatrix, rows, cols, dontPrint, roundFactor);
				printf("Not supported at the moment...Please use a different option\n");
			} else if(fgl == 1) {
				//FGL_Algorithm_Double(inputMatrixDouble, rows, cols, dontPrint, roundFactor, checkRref);
				F4_5_GuassianElimination(inputMatrixDouble, rows, cols, dontPrint, checkRref);
			} else if (cuSparseFlag == 1) {
				F4_5_GuassianEliminationCuSparse(inputMatrixDouble, rows, cols, dontPrint, checkRref);
			} else if (cuSparseFlag == 2) {
				F4_5_GuassianEliminationCuSparseMHVersion(inputMatrixDouble, rows, cols, dontPrint, checkRref);
			} else {
				//GuassianEliminationV1(inputMatrix, rows, cols, dontPrint);
				printf("Not supported at the moment...Please use a different option\n");
			}

			clock_t endDeivce = clock();
			printf("Output:\n");
			if (sparsePrint == 0) {
				printMatrixWithLimitsDouble(inputMatrixDouble, rows, cols, 16);	
			} else {
				printSparseMatrixArrayDouble(inputMatrixDouble, rows, cols, 160);
			}

			clock_t end = clock();
			printf("Done\n=============================================================\n");
			
			time_elapsed += (double)(end - begin) / CLOCKS_PER_SEC;
			time_elapsed_parse += (double)(endParse - beginParse) / CLOCKS_PER_SEC;
			time_elapsed_device += (double)(endDeivce - beginDeivce) / CLOCKS_PER_SEC;
			printf("Parsing Execution Time: %f seconds\n", time_elapsed_parse);
			printf("Device (GPU) Execution Time: %f seconds\n\n", time_elapsed_device);
			printf("Total Execution Time: %f seconds\n", time_elapsed);

			printf("\n\n");

			if(outputExecutionData) {
				CreateExecutionLog(time_elapsed, time_elapsed_parse, time_elapsed_device);
			}
		}
	}

	return 0;
}