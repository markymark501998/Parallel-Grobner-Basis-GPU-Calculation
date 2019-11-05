#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "poly.h"
#include "parser.h"

#define MAXCHAR 100000

int main(int argc, char* argv[]) {
	char str[MAXCHAR];
	int i = 0;

	printf("\n");
	printf("\n");
	printf("Senior Design Parser Testing\n\nNote: Each line in the input can only have 100000 characters\n");
	printf("Rules: ([Coeff(float)]x[0...N]^[0...N] (0 to many ->) x[0...N]^[0...N]) + (Polynomial Term) + ...\n\n");
	printf("=============================================================\n");

	int debug = 0;
	int failState = 0;
	char failMessage[1000];
	struct PolynomialSystem system;
	int size = 0;

	for (i = 2; i < argc; i++) {
		if (strcmp(argv[i], "-d") == 0) {
			debug = 1;
			continue;
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
			printf("File Contents:\n");

			//Build system from file
			while (fgets(str, MAXCHAR, file) != NULL) {
				printf("%s\n", str);

				struct Polynomial *poly = parsePoly(str);

				if (size == 0) {
					system.head = poly;
					system.tail = poly;
				} else {
					poly->prev = system.tail;
					system.tail->next = poly;
					system.tail = poly;
				}

				size++;
			}
			printf("Printing the System...\n");
			struct Polynomial *poly = system.head;
			int i;
			for(i=0; i<size; i++) {
				printf("%d : ", i);
				printPoly(poly);

				poly = poly->next;
			}

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
