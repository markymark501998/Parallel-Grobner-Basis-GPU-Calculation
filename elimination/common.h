int GuassianEliminationV1 (float**, int, int, int);
int GuassianEliminationV1Rref (float**, int, int, int, int);
int FGL_Algorithm (float**, int, int, int, int);

float** parseInputMatrix(FILE *, int, int*, int*);

void printMatrix(float **, int, int);
void printMatrixWithLimits(float **, int, int, int);

void printCublasMatrixArray(float * , int);
void printCublasMatrixArrayConverted (float*, int, int);

void printStandardIntArray(int*, int);

void printSparseMatrixArray (float**, int, int, int);
void printSparseMatrixArrayConverted (float*, int, int, int);