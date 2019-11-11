#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include "common.h"
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define MAXCHAR 100000

int main(int argc, char* argv[]) {
	printf("\n");
	printf("Senior Design Guassian Elimination Testing\n\nNote: Each line in the input file can only have 100000 characters\n");
	printf("==========================================================================================================================\n");
	printf("Input Matrix file must be in this form:\n\n[rows (int)]\n[columns (int)]\n[float + 'space' + float + 'space' + ...]\n[......]\n\n");

	float** inputMatrix;
	int rows = 0;
	int cols = 0;
	
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
			inputMatrix = parseInputMatrix(file, MAXCHAR, &rows, &cols);			
			printMatrix(inputMatrix, rows, cols);
			
			
			
			
			

			fclose(file);
		}
	}

	return 0;
}
