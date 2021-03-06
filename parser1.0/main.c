#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "poly.h"
#include "parser.h"

int main(int argc, char* argv[]) {
	int i = 0;

	printf("\n");
	printf("\n");
	printf("Senior Design Parser Testing\n\nNote: Each line in the input can only have 100000 characters\n");
	printf("Rules: ([Coeff(float)]x[0...N]^[0...N] (0 to many ->) x[0...N]^[0...N]) + (Polynomial Term) + ...\n\n");
	printf("=============================================================\n");

	int debug = 0;
	int mono_order = 0;		//default: 0-grevlex, 1-grlex, 2-lex
	int failState = 0;
	char failMessage[1000];
	struct PolynomialSystem *system;

	for (i = 2; i < argc; i++) {
		if (strcmp(argv[i], "-d") == 0) {
			debug = 1;
			continue;
		} else if (strcmp(argv[i], "-grevlex") == 0) {
			mono_order = 0;
		} else if (strcmp(argv[i], "-grlex") == 0) {
			mono_order = 1;
		} else if (strcmp(argv[i], "-lex") == 0) {
			mono_order = 2;
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
			printf( "Could not open file\n" );
		}
		else
		{
			printf("Building system.");
			system = buildPolySystem(file, mono_order);

			//Build system from file
			printf("Printing the System...\n");
			printPolySystem(system);
			//check failState
			if(failState == 1)
			{
				printf("Failed to parse file. Message: %s\n", failMessage);
			}
			else
			{
				printf("Parsed Successfully!\n");
			}

			fclose(file);
		}
	}

	printf("\n");

	return 0;
}
