#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include "common.h"
#include <time.h>
#define MAXCHAR 100000

int main(int argc, char* argv[]) {
	printf("\n");
	printf("Senior Design Guassian Elimination Testing\n\nNote: Each line in the input file can only have 100000 characters\n");
	printf("==========================================================================================================================\n");
	printf("Input Matrix file must be in this form:\n\n[rows (int)]\n[columns (int)]\n[float + 'space' + float + 'space' + ...]\n[......]\n\n");
	printf("Note: the Matrix on the host will exist in the 'CUDA' way (just one array) and in float** format\n");
	printf("Of course the final format will not follow this as it 'doubles up' on storage space on the host which is uneccessary [get Mark H. to fix it when this goes to implementation]\n\n");

	double time_elapsed = 0.0;
	int i, j;
	float** inputMatrix;
	int rows = 0;
	int cols = 0;
	int dontPrint = 0;

	for(i = 0; i < argc; i++) {
		if(strcmp(argv[i], "-dontPrint") == 0) {
			dontPrint = 1;
		}
	}
	
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
			
			inputMatrix = parseInputMatrix(file, MAXCHAR, &rows, &cols);
			fclose(file);			

			if(dontPrint == 0)
				printMatrixWithLimits(inputMatrix, rows, cols, 12);	

			GuassianEliminationV1(inputMatrix, rows, cols);

			if(dontPrint == 0)
				printMatrixWithLimits(inputMatrix, rows, cols, 12);

			printf("Done\n");

			clock_t end = clock();
			time_elapsed += (double)(end - begin) / CLOCKS_PER_SEC;
			printf("Execution Time: %f seconds\n", time_elapsed);
		}
	}

	return 0;
}
