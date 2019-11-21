#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include "common.h"
#include <time.h>
#define MAXCHAR 100000

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
	printf("Options:\n-dontPrint    (doesn't print extra debugging. Large matrices will overflow the console with repeated logging)\n");
	printf("-rref         (reduced row echelon form)\n");
	printf("-fgl          (faugere-lachartre guassian elimination)\n");
	printf("-output       (outputs execution data to the 'ExecutionLogs' directory)\n\n");

	double time_elapsed = 0.0;
	double time_elapsed_parse = 0.0;
	double time_elapsed_device = 0.0;
	int i, j;
	float** inputMatrix;
	int rows = 0;
	int cols = 0;
	int dontPrint = 0;
	int rrefForm = 0;
	int fgl = 0;
	int outputExecutionData = 0;

	printf("------------------------------------------------------------\n");

	for(i = 0; i < argc; i++) {
		if(strcmp(argv[i], "-dontPrint") == 0) {
			dontPrint = 1;
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
	}
	
	printf("-             [Don't Print: %d]\n", dontPrint);
	printf("-             [Reduced Row Echelon Form: %d]\n", rrefForm);
	printf("-             [Faugere-Lachartre Guassian Elimination: %d]\n", fgl);
	printf("-             [Output Execution Data -> ExecutionLogs: %d]\n", outputExecutionData);

	printf("------------------------------------------------------------\n\n");
	
	if ( argc <= 1 )
	{
		//Error
		printf("Please enter a filename in the command line: ./main [filename] [other arguments]\n");
	}
	else
	{
		FILE *file = fopen( argv[1], "r" );

		if ( file == 0 )
		{
			printf( "Could not open the file\n" );
		}
		else
		{
			clock_t begin = clock();
			clock_t beginParse = clock();
			
			inputMatrix = parseInputMatrix(file, MAXCHAR, &rows, &cols);
			fclose(file);	

			clock_t endParse = clock();	
			clock_t beginDeivce = clock();	 

			printf("Input:\n");
			printMatrixWithLimits(inputMatrix, rows, cols, 12);	

			if(rrefForm == 1) {
				GuassianEliminationV1Rref(inputMatrix, rows, cols, dontPrint);
			} else if(fgl == 1) {
				FGL_Algorithm(inputMatrix, rows, cols, dontPrint);
			} else {
				GuassianEliminationV1(inputMatrix, rows, cols, dontPrint);
			}

			clock_t endDeivce = clock();
			printf("Output:\n");	
			printMatrixWithLimits(inputMatrix, rows, cols, 12);

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