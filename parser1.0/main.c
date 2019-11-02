#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utility.h"

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
			
			//Every line in the parser
			while (fgets(str, MAXCHAR, file) != NULL) {
				printf("%s\n", str);
				
				char* buffer;
				int startSearchIndex = 0;
				
				struct Polynomial poly;
				int doneParsingPoly = 0;
				int firstTermFlag = 1;				
				
				struct PolyTerm* currentTerm;
				
				while(doneParsingPoly == 0) {
					currentTerm = (struct PolyTerm*) malloc (sizeof(struct PolyTerm));	
					
					if(firstTermFlag == 1) {
						firstTermFlag = 0;						
						poly.head = currentTerm;
					}
					else {
						if(str[startSearchIndex] != '+' && str[startSearchIndex+1] != '+') {
							if(str[startSearchIndex] == '\n' || str[startSearchIndex+1] == '\n') {
								break;
							}
						}
						
						poly.tail->next = currentTerm;
					}
					
					poly.tail = currentTerm;
					
					int indexOfCoeffStart = indexOfStart(str, '[', startSearchIndex);
					int indexOfCoeffEnd = indexOfStart(str, ']', startSearchIndex);
					startSearchIndex = indexOfCoeffEnd;
					
					int coeffLen = indexOfCoeffEnd - indexOfCoeffStart - 1;
					
					buffer = (char*) malloc(coeffLen * sizeof(char));
					substring(str, buffer, (indexOfCoeffStart + 1), coeffLen);
					
					currentTerm->coeff = strtof(buffer, NULL);
					
					int doneFlag = 0;
					int firstFlag = 0;
					
					struct VarItem* nextItem;
										
					while (doneFlag == 0) {				
						startSearchIndex++;
						if(str[startSearchIndex] != 'x') {
							if(str[startSearchIndex] != '*' && str[startSearchIndex + 1] != 'x') {
								break;
							}						
						}
						
						nextItem = (struct VarItem*) malloc (sizeof(struct VarItem));					
						
						if(firstFlag == 0) {
							firstFlag = 1;
							
							currentTerm->head = nextItem;
						}
						else {
							currentTerm->tail->next = nextItem;
						}
						
						currentTerm->tail = nextItem;
						
						int bracketStart = indexOfStart(str, '[', startSearchIndex);
						int bracketEnd = indexOfStart(str, ']', startSearchIndex);
						startSearchIndex = bracketEnd + 1;					
						int numLength = bracketEnd - bracketStart - 1;
						
						free(buffer);
						buffer = (char*) malloc(numLength * sizeof(char));
						substring(str, buffer, (bracketStart + 1), numLength);					
						nextItem->varNum = atoi(buffer);
						
						if(str[startSearchIndex] != '^') {
							break;						
						}
						
						bracketStart = indexOfStart(str, '[', startSearchIndex);
						bracketEnd = indexOfStart(str, ']', startSearchIndex);
						startSearchIndex = bracketEnd + 1;					
						numLength = bracketEnd - bracketStart - 1;
						
						free(buffer);
						buffer = (char*) malloc(numLength * sizeof(char));
						substring(str, buffer, (bracketStart + 1), numLength);					
						nextItem->varPow = atoi(buffer);			
						
						if((str[startSearchIndex-1] == ']' && str[startSearchIndex] == '\0')
						|| (str[startSearchIndex-1] == ']' && (str[startSearchIndex] == '+' || str[startSearchIndex+1] == '+'))) {
							doneFlag = 1;
						}
					}
					
					if(str[startSearchIndex+1] != '[') {
						break;
					}					
				}
				
				//At this point the variable "poly" has a linked list of terms (not ordered)
				//for a single polynomial (aka: a single row of the input text file)
				
				//Print the Polynomial
				struct PolyTerm* currentTermTemp;
				currentTermTemp = poly.head;
				
				while (currentTermTemp != NULL) {
					printf("coeff: %f ", currentTermTemp->coeff);
					
					struct VarItem* currentVarItem;
					currentVarItem = currentTermTemp->head;
					
					int firstTermTemp = 1;
					while (currentVarItem != NULL) {
						if(firstTermTemp == 1) {
							firstTermTemp = 0;
							printf("x[%d]^[%d]", currentVarItem->varNum, currentVarItem->varPow);
						} else
							printf("*x[%d]^[%d]", currentVarItem->varNum, currentVarItem->varPow);
							
						
						currentVarItem = currentVarItem->next;
					}
					
					printf("\n");					
					currentTermTemp = currentTermTemp->next;
				}
			}
			
			if(failState == 1) {
				printf("Failed to parse file. Message: %s\n", failMessage);
			}
			else {
				printf("Parsed Successfully!\n");
			}
				
			fclose(file);
        }
    }
	
	printf("\n");
	printf("\n");
	
	return 0;
}
