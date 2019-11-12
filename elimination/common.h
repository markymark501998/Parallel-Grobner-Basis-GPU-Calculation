int GuassianEliminationV1 (float**, int, int, int);

float** parseInputMatrix(FILE *, int, int*, int*);
void printMatrix(float **, int, int);
void printMatrixWithLimits(float **, int, int, int);
void printCublasMatrixArray(float * , int);
void printCublasMatrixArrayConverted (float*, int, int);